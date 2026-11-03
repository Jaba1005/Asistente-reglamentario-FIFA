from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modelo de embeddings local — corre en CPU, se descarga ~120 MB la primera vez
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"[OK] Embeddings locales cargados: {EMBEDDING_MODEL}")

# Corpus de ejemplo
reglas = [
    "Un jugador estará en fuera de juego si está más cerca de la línea de meta contraria que el balón y el penúltimo adversario.",
    "Se concederá un tiro penal si un jugador comete una falta sancionable con tiro libre directo dentro de su propia área.",
    "El partido se compone de dos tiempos de 45 minutos cada uno.",
    "El árbitro podrá mostrar la tarjeta roja a un jugador por conducta violenta.",
]

corpus_embeddings = embeddings.embed_documents(reglas)
query = "cuándo pitan un penalti"
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], corpus_embeddings)
indices_ordenados = np.argsort(scores[0])[::-1]

print(f"\nConsulta: \"{query}\"\n")
for idx in indices_ordenados:
    barra = "█" * int(scores[0][idx] * 20)
    print(f"  {scores[0][idx]:.4f} {barra}")
    print(f"  {reglas[idx]}\n")