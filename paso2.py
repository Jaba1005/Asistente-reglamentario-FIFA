# 1. Importamos la variable 'documents' de tu primer archivo
from langchain_huggingface import HuggingFaceEmbeddings

from paso1 import documents 

# 2. Importamos las librerías necesarias
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3. Inicializamos el modelo de embeddings (Este es el "Paso 0" que te faltaba)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- DESDE AQUÍ EMPIEZA TU CÓDIGO ORIGINAL ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)

print(f"Documentos originales (páginas): {len(documents)}")
print(f"Fragmentos generados:            {len(chunks)}")
print(f"Factor de expansión:             {len(chunks)/len(documents):.1f}x")
print()
print("Ejemplo — Fragmento #20:")
print(f"  Fuente: {chunks[20].metadata.get('source', '?')}")
print(f"  Página: {chunks[20].metadata.get('page', '?')}")
print(f"  Longitud: {len(chunks[20].page_content)} caracteres")
print(f"  Contenido: {chunks[20].page_content}")

# El modelo ya fue cargado arriba — lo reutilizamos
embeddings_model = embeddings

sample_text = chunks[20].page_content
sample_vector = embeddings_model.embed_query(sample_text)

print("\nTexto de muestra:")
print(f"  {sample_text[:120]}...")
print()
print("Embedding generado:")
print(f"  Dimensiones del vector: {len(sample_vector)}")
print(f"  Rango de valores:       [{min(sample_vector):.4f},  {max(sample_vector):.4f}]")
print(f"  Primeros 8 valores:     {[round(v, 4) for v in sample_vector[:8]]}")