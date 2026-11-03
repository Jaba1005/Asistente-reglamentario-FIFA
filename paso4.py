import os
import re
import json
import shutil
import numpy as np
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings_onnx import MultilingualOnnxEmbeddings

# Modelo multilingüe: entiende español (all-MiniLM-L6-v2 solo inglés)
# Debe ser EL MISMO que usa pipelineRag.py para consultar
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2 (ONNX int8)"

PDF_DIR = "pdfs"
PERSIST_DIR = "./chroma"
INDICE_DIR = "./indice"
COLLECTION_NAME = "reglas_de_juego"

# 1. Cargar los PDFs
documents = []
for pdf_file in sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")):
    pages = PyPDFLoader(os.path.join(PDF_DIR, pdf_file)).load()
    seccion = ""
    for p in pages:
        # Solo el nombre del archivo: la ruta de Windows ("pdfs\x.pdf") no sirve en Render (Linux)
        p.metadata["source"] = pdf_file
        # La cabecera "Reglas de Juego 2026/27 | Regla 11 | El fuera de juego" solo sale
        # en algunas páginas: se arrastra a las siguientes hasta que cambie
        cabecera = re.search(r"Reglas de Juego \d{4}/\d{2}\s*\|\s*([^\n]+)", p.page_content)
        if cabecera:
            seccion = re.sub(r"\s*\|\s*", " – ", cabecera.group(1)).strip()
        p.metadata["seccion"] = seccion
    documents.extend(pages)
    print(f"  {pdf_file}: {len(pages)} páginas cargadas")

# 2. Partir en fragmentos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)
# Anteponer la regla al texto: sin esto, "fuera de juego" (Regla 11) se confunde
# con "fuera del terreno de juego" al buscar
for c in chunks:
    if c.metadata["seccion"]:
        c.page_content = f"{c.metadata['seccion']}\n{c.page_content}"
print(f"Fragmentos generados: {len(chunks)}")

# 3. Embeddings locales
embeddings_model = MultilingualOnnxEmbeddings()

# 4. Borrar la base anterior y crear una nueva con los fragmentos
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"Base de datos anterior eliminada: {PERSIST_DIR}")

print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    collection_metadata={"hnsw:space": "cosine"}
)

coleccion = vector_store._collection
print()
print("[OK] Base vectorial creada exitosamente!")
print(f"  Ubicación:             {PERSIST_DIR}/")
print(f"  Colección:             {coleccion.name}")
print(f"  Fragmentos indexados:  {coleccion.count()}")
print(f"  Modelo embeddings:     {EMBEDDING_MODEL} (local, CPU)")

# 5. Exportar los vectores para el servidor (pipelineRag.py busca con numpy:
#    el motor de Chroma se quedaba congelado en Render)
datos = coleccion.get(include=["embeddings", "documents", "metadatas"])
os.makedirs(INDICE_DIR, exist_ok=True)
np.save(os.path.join(INDICE_DIR, "vectores.npy"), np.asarray(datos["embeddings"], dtype=np.float32))
fragmentos = [
    {
        "texto": texto,
        "fuente": meta.get("source", "?"),
        "pagina": meta.get("page_label", meta.get("page", "?")),
    }
    for texto, meta in zip(datos["documents"], datos["metadatas"])
]
with open(os.path.join(INDICE_DIR, "fragmentos.json"), "w", encoding="utf-8") as f:
    json.dump(fragmentos, f, ensure_ascii=False)
print(f"  Índice para el servidor: {INDICE_DIR}/ ({len(fragmentos)} fragmentos)")
