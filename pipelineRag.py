"""Compatibilidad: el agente ahora vive en agente.py (Avance 3).

Se conserva rag_pipeline() con la misma firma porque lo usa taller_multimodal/parte2_audio.py.
"""
from agente import ejecutar_agente


def rag_pipeline(pregunta, k=10):
    """Devuelve solo el texto de la respuesta del agente (k se ignora: el límite está en tools.py)."""
    return ejecutar_agente(pregunta)["respuesta"]
