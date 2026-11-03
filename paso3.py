# 1. Importamos la variable 'chunks' desde tu segundo archivo
from paso2 import chunks

# 2. Importamos la librería para los embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Inicializamos el modelo (Este es el famoso "Paso 0" que mencionaba tu código)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Reutilizamos el modelo
embeddings_model = embeddings

sample_text = chunks[20].page_content
sample_vector = embeddings_model.embed_query(sample_text)

print("Texto de muestra:")
print(f"  {sample_text[:120]}...")
print()
print("Embedding generado:")
print(f"  Dimensiones del vector: {len(sample_vector)}")
print(f"  Rango de valores:       [{min(sample_vector):.4f},  {max(sample_vector):.4f}]")
print(f"  Primeros 8 valores:     {[round(v, 4) for v in sample_vector[:8]]}")