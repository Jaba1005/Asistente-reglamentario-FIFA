import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
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

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,
    api_key=GROQ_API_KEY,
    timeout=60,
    max_retries=1
)

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

PROMPT_TEMPLATE = '''Eres un consultor experto en las Reglas de Juego del fútbol (IFAB/FIFA). Cuando el contexto lo permita, indica el número y nombre de la regla (por ejemplo: Regla 12 – Faltas e incorrecciones).
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def buscar(pregunta, k):
    # Los vectores están normalizados: el producto punto es la similitud coseno
    vector = np.asarray(embeddings_model.embed_query(pregunta), dtype=np.float32)
    similitudes = vectores @ vector
    mejores = np.argsort(-similitudes)[:k]
    return [fragmentos[i] for i in mejores]


def rag_pipeline(pregunta, k=10):
    log(f"Pregunta recibida: {pregunta[:80]!r}")
    docs = buscar(pregunta, k)
    log(f"1/2 búsqueda lista: {len(docs)} fragmentos")

    contexto = "\n\n---\n\n".join(
        f"[Fuente: {d['fuente']} — Pág. {d['pagina']}]\n{d['texto']}" for d in docs
    )

    prompt_con_rag = prompt_template.invoke({"context": contexto, "question": pregunta})
    respuesta = llm.invoke(prompt_con_rag).content
    log("2/2 Groq respondió")
    return respuesta
