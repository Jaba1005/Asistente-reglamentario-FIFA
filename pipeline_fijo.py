"""Pipeline fijo del Avance 2 (sin tools), conservado para compararlo con el agente.

Siempre: buscar k=10 fragmentos -> pegarlos en el prompt -> 1 llamada al LLM.
Lo usa evaluar_agente.py; la app desplegada usa el agente (agente.py).
"""
import time
from langchain_core.prompts import ChatPromptTemplate

from busqueda import buscar, log
from agente import llm  # mismo modelo y temperatura que el agente, para comparar en igualdad

K_FIJO = 10

PROMPT_TEMPLATE = '''Eres un consultor experto en las Reglas de Juego del fútbol (IFAB/FIFA). Cuando el contexto lo permita, indica el número y nombre de la regla (por ejemplo: Regla 12 – Faltas e incorrecciones).
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def rag_pipeline_fijo(pregunta, k=K_FIJO):
    inicio = time.time()
    docs = buscar(pregunta, k)
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {d['fuente']} — Pág. {d['pagina']}]\n{d['texto']}" for d in docs
    )
    respuesta = llm.invoke(prompt_template.invoke({"context": contexto, "question": pregunta}))
    uso = getattr(respuesta, "usage_metadata", None) or {}
    log("Pipeline fijo: Groq respondió")
    return {
        "respuesta": respuesta.content,
        "contextos": [d["texto"] for d in docs],
        "llamadas_llm": 1,
        "tokens_total": uso.get("input_tokens", 0) + uso.get("output_tokens", 0),
        "tiempo_s": round(time.time() - inicio, 2),
    }
