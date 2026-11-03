import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from paso5 import retriever
from paso6 import pregunta, prompt_aumentado, prompt_template

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Esta variable DEBE quedar afuera porque la necesitamos en el pipeline
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=GROQ_API_KEY
)

# ==========================================
# TODO LO QUE ESTÉ AQUÍ ABAJO SE IGNORARÁ CUANDO INICIES FLASK
# ==========================================

if __name__ == '__main__':
    print("[OK] LLM configurado (vía Groq API)")
    print("\nEnviando prompt aumentado a Groq...")
    print(f"Pregunta: \"{pregunta}\"\n")

    # --- RESPUESTA CON RAG ---
    respuesta = llm.invoke(prompt_aumentado)
    texto_respuesta = respuesta.content

    print("=" * 60)
    print("RESPUESTA DEL LLM (basada en el contexto RAG):")
    print("=" * 60)
    print(texto_respuesta)
    print("\n" + "=" * 60 + "\n")

    # --- PRUEBA DE COMPARACIÓN: RAG vs SIN RAG ---
    pregunta_test = "¿Qué estudiantes están matriculados en el curso?"

    print("--- INICIANDO PRUEBA DE COMPARACIÓN ---")
    print(f"Pregunta: \"{pregunta_test}\"\n")

    # A. Con RAG (Consultando tu documento)
    docs_test = retriever.invoke(pregunta_test)
    contexto_test = "\n\n---\n\n".join(
        f"[Fragmento {i+1}]\n{d.page_content}" for i, d in enumerate(docs_test)
    )
    prompt_con_rag = prompt_template.invoke({"context": contexto_test, "question": pregunta_test})
    texto_con_rag = llm.invoke(prompt_con_rag).content

    # B. Sin RAG (Solo conocimiento general del LLM)
    texto_sin_rag = llm.invoke([HumanMessage(content=pregunta_test)]).content

    print("[CON RAG — basado en el documento de la Konrad Lorenz]")
    print(texto_con_rag)
    print("\n------------------------------------------------------------\n")
    print("[SIN RAG — solo conocimiento general del LLM]")
    print(texto_sin_rag)