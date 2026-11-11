"""Parte 2 del taller: audio -> transcripción (Whisper en Groq) -> respuesta del modelo.

Uso:
    python parte2_audio.py                  # transcribe los audios de audios/audios.json y responde
    python parte2_audio.py --solo-transcribir
    python parte2_audio.py --vocabulario    # pasa a Whisper una lista de términos del dominio (prompt)
    python parte2_audio.py --modo simple    # responde con el LLM directo, sin la base de conocimientos
    python parte2_audio.py --solo-responder # usa resultados/2_1_transcripciones.json sin volver a transcribir

Por defecto la respuesta la da el pipeline del proyecto (pipelineRag.rag_pipeline), es decir,
el mismo asistente de la app con la base de conocimientos de las Reglas de Juego.
Antes de ejecutar, copiar los audios a audios/ y completar audios/audios.json
(archivo, hablante, condición y lo que se dijo).
"""
import os
import sys
import json
import argparse
import time
import subprocess
from groq import RateLimitError

from comun import cliente, limpiar, guardar, titulo, AUDIO_MODEL, TEXT_MODEL, AUDIOS, CARPETA

# Términos del dominio para ayudar a Whisper (solo con --vocabulario)
VOCABULARIO = ("Reglas de Juego, IFAB, fuera de juego, saque de meta, saque de banda, saque de esquina, "
               "guardameta, área penal, tiro libre directo, tiro libre indirecto, tiro penal, VAR, "
               "área de revisión, amonestación, expulsión, balón a tierra")
MAX_MB = 25  # límite del plan gratuito de Groq
# El plan gratuito permite 8.000 tokens por minuto en openai/gpt-oss-120b y cada pregunta
# del asistente usa varios miles: se espera entre preguntas y se reintenta si llega el error 429
PAUSA_ENTRE_PREGUNTAS = 30
ESPERA_REINTENTO = 60
MAX_REINTENTOS = 3


def duracion_ffprobe(ruta):
    try:
        salida = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=nw=1:nk=1", ruta], capture_output=True, text=True, timeout=30)
        return round(float(salida.stdout.strip()), 1)
    except Exception:
        return None


def transcribir(client, ruta, vocabulario):
    with open(ruta, "rb") as f:
        t = client.audio.transcriptions.create(
            file=(os.path.basename(ruta), f.read()),
            model=AUDIO_MODEL,
            language="es",
            response_format="verbose_json",  # trae también la duración
            temperature=0.0,
            prompt=VOCABULARIO if vocabulario else None,
        )
    extra = getattr(t, "model_extra", None) or {}
    duracion = getattr(t, "duration", None) or extra.get("duration")
    return t.text.strip(), (round(float(duracion), 1) if duracion else duracion_ffprobe(ruta))


def responder_simple(client, pregunta):
    r = client.chat.completions.create(
        model=TEXT_MODEL, temperature=0.0,
        messages=[{"role": "system", "content": "Eres un experto en las Reglas de Juego del fútbol 2026/27 de la IFAB. "
                                                "Responde en español, breve y preciso."},
                  {"role": "user", "content": pregunta}])
    return limpiar(r.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo-transcribir", action="store_true")
    parser.add_argument("--solo-responder", action="store_true",
                        help="no transcribir: usar resultados/2_1_transcripciones.json")
    parser.add_argument("--vocabulario", action="store_true", help="enviar términos del dominio a Whisper")
    parser.add_argument("--modo", default="proyecto", choices=["proyecto", "simple"])
    args = parser.parse_args()
    client = cliente()

    with open(os.path.join(AUDIOS, "audios.json"), encoding="utf-8") as f:
        audios = json.load(f)

    sufijo = "_vocabulario" if args.vocabulario else ""
    if args.solo_responder:
        with open(os.path.join(CARPETA, "resultados", f"2_1_transcripciones{sufijo}.json"), encoding="utf-8") as f:
            filas = json.load(f)
        print(f"Usando {len(filas)} transcripciones guardadas en resultados/2_1_transcripciones{sufijo}.json")
    else:
        filas = transcribir_todos(client, audios, args.vocabulario)
        guardar(f"2_1_transcripciones{sufijo}.json", filas)

    if args.solo_transcribir or not filas:
        return
    responder_todos(client, filas, args.modo, sufijo)


def transcribir_todos(client, audios, vocabulario):
    titulo(f"2.1 Transcripción desde archivo ({AUDIO_MODEL}{', con vocabulario' if vocabulario else ''})")
    filas = []
    for a in audios:
        ruta = os.path.join(AUDIOS, a["archivo"])
        if not os.path.exists(ruta):
            print(f"AVISO: no existe audios/{a['archivo']}; se omite.")
            continue
        tam = os.path.getsize(ruta) / 1_048_576
        if tam > MAX_MB:
            print(f"AVISO: {a['archivo']} pesa {tam:.1f} MB (> {MAX_MB} MB); se omite.")
            continue
        texto, duracion = transcribir(client, ruta, vocabulario)
        print(f"\n[{a['archivo']}] {a['hablante']} · {a['condicion']} · {duracion} s")
        print(f"  Lo que se dijo:  {a['lo_que_se_dijo']}")
        print(f"  Transcripción:   {texto}")
        filas.append(dict(a, duracion_s=duracion, transcripcion=texto))
        if duracion and not 10 <= duracion <= 30:
            print(f"  AVISO: dura {duracion} s; el enunciado pide audios de 10 a 30 segundos.")
    return filas


def responder_todos(client, filas, modo, sufijo):
    titulo(f"2.2 Respuesta del modelo a partir de la transcripción (modo {modo})")
    if modo == "proyecto":
        # Se reutiliza el asistente del proyecto sin modificarlo
        sys.path.insert(0, os.path.join(CARPETA, ".."))
        from pipelineRag import rag_pipeline
    archivo = f"2_2_respuestas{sufijo}_{modo}.json"
    for i, fila in enumerate(filas):
        if i > 0:
            print(f"(esperando {PAUSA_ENTRE_PREGUNTAS} s por el límite de tokens por minuto de Groq)")
            time.sleep(PAUSA_ENTRE_PREGUNTAS)
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                if modo == "proyecto":
                    fila["respuesta"] = rag_pipeline(fila["transcripcion"], k=10)
                else:
                    fila["respuesta"] = responder_simple(client, fila["transcripcion"])
                break
            except RateLimitError:
                print(f"Límite de Groq (429). Reintento {intento}/{MAX_REINTENTOS} en {ESPERA_REINTENTO} s...")
                time.sleep(ESPERA_REINTENTO)
        else:
            fila["respuesta"] = "ERROR: límite de tokens por minuto de Groq (429) tras varios reintentos."
        print(f"\n[{fila['archivo']}] Transcripción enviada: {fila['transcripcion']}")
        print(f"Respuesta:\n{fila['respuesta']}")
        guardar(archivo, filas[:i + 1])  # se guarda después de cada audio


if __name__ == "__main__":
    main()