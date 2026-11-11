"""Funciones compartidas por las dos partes del taller."""
import os
import re
import sys
import json
import base64
from dotenv import load_dotenv
from groq import Groq

CARPETA = os.path.dirname(os.path.abspath(__file__))
# El .env está en la raíz del proyecto (un nivel arriba)
load_dotenv(os.path.join(CARPETA, "..", ".env"))
load_dotenv()

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "whisper-large-v3-turbo")
TEXT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

IMAGENES = os.path.join(CARPETA, "imagenes")
AUDIOS = os.path.join(CARPETA, "audios")
RESULTADOS = os.path.join(CARPETA, "resultados")


def cliente():
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta la variable de entorno GROQ_API_KEY (ponla en el .env de la raíz del proyecto).")
    return Groq()


def imagen_base64(nombre, mime="image/png"):
    """Lee una imagen local y la devuelve como data URL en base64 (igual que multimodal.py)."""
    with open(os.path.join(IMAGENES, nombre), "rb") as f:
        datos = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{datos}"


def bloque_imagen(nombre, mime="image/png"):
    return {"type": "image_url", "image_url": {"url": imagen_base64(nombre, mime)}}


def limpiar(texto):
    """Quita el razonamiento <think>...</think> que a veces devuelven los modelos Qwen."""
    return re.sub(r"<think>.*?</think>", "", texto or "", flags=re.S).strip()


def guardar(nombre, datos):
    os.makedirs(RESULTADOS, exist_ok=True)
    ruta = os.path.join(RESULTADOS, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        if isinstance(datos, str):
            f.write(datos)
        else:
            json.dump(datos, f, ensure_ascii=False, indent=2)
    print(f"   -> guardado en resultados/{nombre}")


def titulo(texto):
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)
