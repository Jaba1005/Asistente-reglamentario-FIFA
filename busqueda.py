"""Recuperación sobre el índice vectorial de las Reglas de Juego (igual que en el Avance 2).

La usan el agente (a través de la tool buscar_en_base) y el pipeline fijo del Avance 2
(pipeline_fijo.py, para comparar en la evaluación). El corpus y los vectores nunca salen
de aquí: al LLM solo le llegan los fragmentos recuperados para cada consulta.
"""
import os
import json
import time
import numpy as np
from embeddings_onnx import MultilingualOnnxEmbeddings

INDICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indice")


def log(mensaje):
    # flush=True para que aparezca en los logs de Render al instante
    print(f"[rag {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


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


def seccion_de(texto):
    """paso4.py antepone la sección ('Regla 11 – El fuera de juego') como primera línea."""
    primera = texto.split("\n", 1)[0].strip()
    return primera if len(primera) < 80 else ""


def buscar(consulta, k):
    """Devuelve los k fragmentos más parecidos a la consulta, con su similitud."""
    # Los vectores están normalizados: el producto punto es la similitud coseno
    vector = np.asarray(embeddings_model.embed_query(consulta), dtype=np.float32)
    similitudes = vectores @ vector
    mejores = np.argsort(-similitudes)[:k]
    return [dict(fragmentos[i], similitud=float(similitudes[i])) for i in mejores]
