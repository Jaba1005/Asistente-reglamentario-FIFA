import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tavily import TavilyClient
from typing import Dict, Any
from embeddings_onnx import MultilingualOnnxEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-120b"
INDICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indice")


def log(mensaje):
    # flush=True para que aparezca en los logs de Render al instante
    print(f"[rag {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


if not GROQ_API_KEY:
    log("ADVERTENCIA: falta la variable de entorno GROQ_API_KEY")
if not os.getenv("TAVILY_API_KEY"):
    log("ADVERTENCIA: falta la variable de entorno TAVILY_API_KEY")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,
    api_key=GROQ_API_KEY,
    timeout=60,
    max_retries=1
)

tavily_client = TavilyClient()

# Debe ser el mismo modelo con el que se indexó la base (ver paso4.py)
log("Cargando modelo de embeddings...")
embeddings_model = MultilingualOnnxEmbeddings()
# Primer embedding al arrancar, así la primera pregunta no espera la carga
embeddings_model.embed_query("fuera de juego")
log("Modelo de embeddings listo")

# Índice exportado por paso4.py. Con unos cientos de fragmentos, comparar contra
# todos con numpy es instantáneo; el motor de Chroma se congelaba en Render.
vectores = np.load(os.path.join(INDICE_DIR, "vectores.npy"))
with open(os.path.join(INDICE_DIR, "fragmentos.json"), encoding="utf-8") as f:
    fragmentos = json.load(f)
log(f"Índice listo: {len(fragmentos)} fragmentos")

SYSTEM_PROMPT = '''Eres un consultor experto en las Reglas de Juego del fútbol (IFAB/FIFA). Cuando el contexto lo permita, indica el número y nombre de la regla (por ejemplo: Regla 12 – Faltas e incorrecciones).
Para responder preguntas sobre las Reglas de Juego, usa la herramienta buscar_reglamentos. Puedes llamarla varias veces con consultas distintas si lo necesitas.
Hoy es {fecha}. Tu conocimiento tiene una fecha de corte anterior, así que para cualquier pregunta sobre hechos actuales o recientes usa siempre la herramienta web_search antes de responder.
Responde usando ÚNICAMENTE la información que devuelvan las herramientas. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en la información recuperada, indica exactamente: "No encontré información sobre esto en la base de conocimientos."
Si el usuario solo saluda o conversa, responde con normalidad sin usar las herramientas.'''

MAX_ITERACIONES = 5  # límite de rondas de tools por pregunta, para evitar bucles infinitos


def buscar(pregunta, k):
    # Los vectores están normalizados: el producto punto es la similitud coseno
    vector = np.asarray(embeddings_model.embed_query(pregunta), dtype=np.float32)
    similitudes = vectores @ vector
    mejores = np.argsort(-similitudes)[:k]
    return [fragmentos[i] for i in mejores]


def context(docs):
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {d['fuente']} — Pág. {d['pagina']}]\n{d['texto']}" for d in docs
    )
    return contexto


@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(
        query=query,
        include_answer='basic',
        max_results=5,
        start_date="2026-01-27",
        end_date="2026-10-27"
        )


def crear_tools(k):
    """Crea las tools que el LLM puede usar, conectadas al índice de las Reglas de Juego."""

    @tool
    def buscar_reglamentos(consulta: str) -> str:
        """Busca en las Reglas de Juego del fútbol (IFAB/FIFA) fragmentos relevantes para la consulta.
        Úsala siempre que la pregunta trate sobre reglas, faltas, sanciones o procedimientos del juego."""
        docs = buscar(consulta, k)
        return context(docs)

    return [buscar_reglamentos, web_search]


def rag_pipeline(pregunta, k=10):
    """Ejecuta el bucle de tool calling hasta que el LLM da una respuesta final."""
    log(f"Pregunta recibida: {pregunta[:80]!r}")

    # Cargar el LLM y conectarle las tools
    tools = crear_tools(k)
    tools_por_nombre = {t.name: t for t in tools}
    llm_con_tools = llm.bind_tools(tools)

    mensajes = [SystemMessage(SYSTEM_PROMPT.format(fecha=time.strftime('%Y-%m-%d'))), HumanMessage(pregunta)]

    for _ in range(MAX_ITERACIONES):
        respuesta = llm_con_tools.invoke(mensajes)
        mensajes.append(respuesta)

        # Si el LLM no pide ninguna tool, esta es la respuesta final
        if not respuesta.tool_calls:
            log("Groq respondió")
            return respuesta.content

        # Ejecutar cada tool que pidió el LLM y devolverle el resultado
        for tool_call in respuesta.tool_calls:
            log(f"[tool] {tool_call['name']}({tool_call['args']})")
            herramienta = tools_por_nombre[tool_call["name"]]
            resultado = herramienta.invoke(tool_call["args"])
            mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))

    return "No pude completar la respuesta: se alcanzó el límite de llamadas a herramientas."
