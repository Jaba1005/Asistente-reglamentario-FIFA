import os

# 1. Importamos la base de datos de tu archivo anterior
from paso4 import vector_store

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

pregunta = "cuando se sanciona un fuera de juego"

print(f"Consulta: \"{pregunta}\"")
print(f"Recuperando los 5 fragmentos más relevantes...\n")

documentos_recuperados = retriever.invoke(pregunta)

print(f"Fragmentos recuperados: {len(documentos_recuperados)}")
print("=" * 60)
for i, doc in enumerate(documentos_recuperados, 1):
    fuente = os.path.basename(doc.metadata.get("source", "desconocido"))
    pagina = doc.metadata.get("page", "?")
    print(f"\n[{i}] {fuente} — Pág. {pagina} ({len(doc.page_content)} chars)")
    print(f"    {doc.page_content[:250]}...")
