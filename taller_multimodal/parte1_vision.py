"""Parte 1 del taller: visión sobre imágenes del dominio (Reglas de Juego 2026/27).

Uso:
    python parte1_vision.py            # ejecuta los 4 puntos
    python parte1_vision.py pregunta   # 1.1 pregunta sobre una imagen
    python parte1_vision.py extraccion # 1.2 extracción estructurada + porcentaje de aciertos
    python parte1_vision.py varias     # 1.3 varias imágenes en una solicitud
    python parte1_vision.py error      # 1.4 error provocado (por defecto: 4 imágenes)
    python parte1_vision.py error --tipo sin_vision   # otras variantes: demasiadas, sin_vision, mime

Las imágenes son páginas del PDF de las Reglas de Juego que usa el proyecto.
"""
import json
import argparse
import unicodedata
from pydantic import BaseModel, Field, ValidationError

from comun import (cliente, bloque_imagen, limpiar, guardar, titulo,
                   VISION_MODEL, TEXT_MODEL, CARPETA)
import os


def preguntar(client, contenido, modelo=VISION_MODEL, temperatura=0.0):
    respuesta = client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": contenido}],
        temperature=temperatura,
    )
    return limpiar(respuesta.choices[0].message.content), respuesta.usage


# ---------------------------------------------------------------------------
# 1.1 Pregunta sobre una imagen
# ---------------------------------------------------------------------------
def punto_1_1(client):
    titulo("1.1 Pregunta sobre una imagen")
    imagen = "01_terreno_de_juego.png"
    pregunta = ("Esta imagen es el diagrama del terreno de juego de las Reglas de Juego 2026/27 de la IFAB. "
                "Según el diagrama, ¿a qué distancia de la línea de meta está el punto penal, cuál es el radio "
                "del círculo central y cuánto mide la portería entre los postes? Da los valores en metros y en yardas.")
    texto, uso = preguntar(client, [{"type": "text", "text": pregunta}, bloque_imagen(imagen)])
    print(f"Imagen:   imagenes/{imagen}\nPregunta: {pregunta}\n\nRespuesta del modelo:\n{texto}")
    print(f"\nTokens: entrada={uso.prompt_tokens} salida={uso.completion_tokens}")
    guardar("1_1_pregunta.json", {"imagen": imagen, "pregunta": pregunta, "respuesta": texto,
                                  "valor_real": "Punto penal 11 m (12 yd); radio del círculo central 9.15 m (10 yd); "
                                                "portería 7.32 m (8 yd)",
                                  "tokens_entrada": uso.prompt_tokens, "tokens_salida": uso.completion_tokens})


# ---------------------------------------------------------------------------
# 1.2 Extracción estructurada
# ---------------------------------------------------------------------------
class PaginaRegla(BaseModel):
    """Esquema de extracción: datos de una página del reglamento."""
    numero_regla: int = Field(description="Número de la regla, según el pie de página (Regla N)")
    nombre_regla: str = Field(description="Nombre de la regla (título grande de la página)")
    numero_pagina: int = Field(description="Número de página impreso en el pie de página")
    temporada: str = Field(description="Temporada de las Reglas de Juego, formato AAAA/AA")
    primer_apartado: str = Field(description="Título del apartado numerado 1")
    cantidad_apartados: int = Field(description="Cantidad de apartados numerados (1., 2., 3., ...) visibles en la página")


ESQUEMA_TEXTO = json.dumps({k: v["description"] for k, v in PaginaRegla.model_json_schema()["properties"].items()},
                           ensure_ascii=False, indent=2)
IMAGENES_EXTRACCION = ["02_regla2_balon.png", "03_regla3_jugadores.png", "04_regla7_duracion.png"]


def extraer_json(texto):
    """Toma el primer objeto JSON del texto (por si el modelo agrega texto o ```json)."""
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fin == -1:
        raise ValueError("la respuesta no contiene un objeto JSON")
    return json.loads(texto[inicio:fin + 1])


def normalizar(valor):
    """Para comparar: minúsculas, sin tildes, sin espacios ni puntuación de sobra."""
    texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode().lower()
    return " ".join(texto.replace(".", " ").replace(",", " ").split())


def punto_1_2(client):
    titulo("1.2 Extracción estructurada")
    with open(os.path.join(CARPETA, "valores_reales.json"), encoding="utf-8") as f:
        reales = json.load(f)

    print("Esquema:\n" + ESQUEMA_TEXTO)
    prompt = ("Extrae de esta página de las Reglas de Juego los siguientes campos y responde SOLO con un objeto "
              f"JSON válido con exactamente estas claves (sin texto adicional):\n{ESQUEMA_TEXTO}")

    filas, detalle = [], []
    for imagen in IMAGENES_EXTRACCION:
        print(f"\n--- {imagen}")
        texto, _ = preguntar(client, [{"type": "text", "text": prompt}, bloque_imagen(imagen)])
        print("Respuesta cruda:", texto)
        registro = {"imagen": imagen, "respuesta_cruda": texto, "valido": False, "errores": None, "extraido": None}

        # Validación: si no es JSON o no cumple el esquema, se informa y se sigue con la siguiente imagen
        try:
            crudo = extraer_json(texto)
            registro["extraido"] = crudo  # aunque no cumpla el esquema, se comparan los campos que sí trae
            datos = PaginaRegla.model_validate(crudo)
            registro.update(valido=True, extraido=datos.model_dump())
            print("JSON válido según el esquema.")
        # ValidationError va primero: en Pydantic 2 también es un ValueError
        except ValidationError as e:
            registro["errores"] = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
            print("AVISO: el JSON no cumple el esquema (se comparan solo los campos que trae):")
            for err in registro["errores"]:
                print("   -", err)
        except (ValueError, json.JSONDecodeError) as e:
            registro["errores"] = f"No es JSON: {e}"
            print(f"AVISO: la respuesta no es un JSON válido ({e}). Se cuentan sus campos como incorrectos.")
        detalle.append(registro)

        extraido = registro["extraido"] or {}
        for campo, real in reales[imagen].items():
            valor = extraido.get(campo, "(sin dato)")
            correcto = normalizar(valor) == normalizar(real)
            filas.append({"imagen": imagen, "campo": campo, "valor_real": real,
                          "valor_extraido": valor, "correcto": correcto})

    aciertos, total = sum(f["correcto"] for f in filas), len(filas)
    porcentaje = round(100 * aciertos / total, 1)
    print("\n| Imagen | Campo | Valor real | Valor extraído | ¿Correcto? |\n|---|---|---|---|---|")
    for f in filas:
        print(f"| {f['imagen']} | {f['campo']} | {f['valor_real']} | {f['valor_extraido']} | "
              f"{'Sí' if f['correcto'] else 'No'} |")
    print(f"\nPorcentaje de aciertos: {aciertos} de {total} campos = {porcentaje} %")
    guardar("1_2_extraccion.json", {"esquema": PaginaRegla.model_json_schema(), "detalle": detalle,
                                    "comparacion": filas, "aciertos": aciertos, "total": total,
                                    "porcentaje": porcentaje})


# ---------------------------------------------------------------------------
# 1.3 Varias imágenes en una solicitud
# ---------------------------------------------------------------------------
def punto_1_3(client):
    titulo("1.3 Varias imágenes en una solicitud")
    imagenes = ["05_senales_arbitro_a.png", "06_senales_arbitro_b.png", "07_senales_asistente.png"]
    pregunta = ("Estas tres imágenes son ilustraciones de señales arbitrales de las Reglas de Juego 2026/27. "
                "Las dos primeras son del árbitro y la tercera del árbitro asistente. "
                "1) Enumera todas las señales que aparecen, indicando en qué imagen está cada una. "
                "2) Compara la señal de tiro libre directo con la de tiro libre indirecto. "
                "3) Explica en qué se diferencia cómo señala un tiro libre el árbitro y cómo lo señala el asistente.")
    contenido = [{"type": "text", "text": pregunta}] + [bloque_imagen(i) for i in imagenes]
    texto, uso = preguntar(client, contenido)
    print(f"Imágenes: {', '.join(imagenes)}\nPregunta: {pregunta}\n\nRespuesta del modelo:\n{texto}")
    print(f"\nTokens: entrada={uso.prompt_tokens} (≈2.048 por imagen) salida={uso.completion_tokens}")
    guardar("1_3_varias_imagenes.json", {
        "imagenes": imagenes, "pregunta": pregunta, "respuesta": texto,
        "senales_reales": {
            "05_senales_arbitro_a.png": ["Tiro libre indirecto", "Tiro libre directo", "Ventaja (1)", "Ventaja (2)"],
            "06_senales_arbitro_b.png": ["Saque de esquina", "Saque de meta", "Tarjeta roja o amarilla",
                                         "Chequeo (dedo a la oreja, otro brazo extendido)"],
            "07_senales_asistente.png": ["Sustitución", "Tiro libre para el atacante", "Tiro libre para el defensor"],
        },
        "tokens_entrada": uso.prompt_tokens, "tokens_salida": uso.completion_tokens})


# ---------------------------------------------------------------------------
# 1.4 Error provocado
# ---------------------------------------------------------------------------
def punto_1_4(client, tipo):
    titulo(f"1.4 Error provocado ({tipo})")
    texto = {"type": "text", "text": "¿Qué señal arbitral aparece en la imagen?"}
    if tipo == "demasiadas":
        explicacion = "Se envían 4 imágenes en un mensaje; el límite del modelo de visión es 3 por solicitud."
        modelo = VISION_MODEL
        contenido = [texto] + [bloque_imagen(i) for i in ["01_terreno_de_juego.png", "05_senales_arbitro_a.png",
                                                          "06_senales_arbitro_b.png", "07_senales_asistente.png"]]
    elif tipo == "sin_vision":
        explicacion = f"Se envía una imagen a {TEXT_MODEL}, un modelo solo de texto."
        modelo = TEXT_MODEL
        contenido = [texto, bloque_imagen("05_senales_arbitro_a.png")]
    else:  # mime
        explicacion = "Se envía un PNG declarado como 'image/tiff' en la data URL (tipo de imagen mal declarado)."
        modelo = VISION_MODEL
        contenido = [texto, bloque_imagen("05_senales_arbitro_a.png", mime="image/tiff")]

    print("Qué se hizo:", explicacion)
    resultado = {"tipo": tipo, "que_se_hizo": explicacion, "modelo": modelo}
    try:
        respuesta, _ = preguntar(client, contenido, modelo=modelo)
        print("La API NO devolvió error. Respuesta:\n", respuesta)
        resultado["error"] = None
        resultado["respuesta"] = respuesta
    except Exception as e:
        # Los errores del SDK de Groq traen el código HTTP y el cuerpo con el mensaje de la API
        codigo = getattr(e, "status_code", None)
        cuerpo = getattr(e, "body", None)
        print(f"Excepción: {type(e).__name__}\nCódigo HTTP: {codigo}\nMensaje de la API: {cuerpo or e}")
        resultado.update(error=type(e).__name__, codigo_http=codigo, mensaje=cuerpo or str(e))
    guardar(f"1_4_error_{tipo}.json", resultado)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("punto", nargs="?", default="todo",
                        choices=["todo", "pregunta", "extraccion", "varias", "error"])
    parser.add_argument("--tipo", default="demasiadas", choices=["demasiadas", "sin_vision", "mime"],
                        help="variante del error provocado (punto 1.4)")
    args = parser.parse_args()
    client = cliente()
    print(f"Modelo de visión: {VISION_MODEL}")

    if args.punto in ("todo", "pregunta"):
        punto_1_1(client)
    if args.punto in ("todo", "extraccion"):
        punto_1_2(client)
    if args.punto in ("todo", "varias"):
        punto_1_3(client)
    if args.punto in ("todo", "error"):
        punto_1_4(client, args.tipo)


if __name__ == "__main__":
    main()
