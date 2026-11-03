from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Importamos las variables de tu paso anterior
from paso5 import documentos_recuperados, pregunta

PROMPT_TEMPLATE = '''Eres un consultor experto en las Reglas de Juego del fútbol (IFAB/FIFA). Cuando el contexto lo permita, indica el número y nombre de la regla (por ejemplo: Regla 12 – Faltas e incorrecciones).
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

contexto = "\n\n---\n\n".join(
    f"[Fuente: {os.path.basename(doc.metadata.get('source', '?'))} — Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
    for i, doc in enumerate(documentos_recuperados)
)

prompt_aumentado = prompt_template.invoke({
    "context": contexto,
    "question": pregunta
})

print("Prompt aumentado construido correctamente.")
print(f"  Fragmentos en el contexto:       {len(documentos_recuperados)}")
print(f"  Caracteres totales del contexto: {len(contexto)}")
print(f"  Tokens aproximados del contexto: ~{len(contexto)//4}")

print("=" * 65)
print("ESTRUCTURA DEL PROMPT AUMENTADO (RAG)")
print("=" * 65)

print("\n[SISTEMA — Instrucciones para el LLM]")
print("  \"Eres un consultor experto en las Reglas de Juego...\"")

print("\n[CONTEXTO — Fragmentos recuperados del vector store]")
for i, doc in enumerate(documentos_recuperados, 1):
    fuente = os.path.basename(doc.metadata.get("source", "?"))
    pagina = doc.metadata.get("page", "?")
    print(f"  [{i}] {fuente} Pág.{pagina} — {len(doc.page_content)} chars")
    print(f"       \"{doc.page_content[:80]}...\"")

print(f"\n[PREGUNTA DEL USUARIO]")
print(f"  \"{pregunta}\"")
print("\n[RESPUESTA]")
print("  (será generada por Groq en el siguiente paso)")
print("=" * 65)
