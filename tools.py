"""Tools del agente (Avance 3).

Tres tools, cada una con sus límites escritos AQUÍ, en el código (no en el prompt):

  1. buscar_en_base   (búsqueda) -> fragmentos de las Reglas de Juego con documento, página y sección
  2. verificar_medida (tarea)    -> convierte unidades y verifica si una medida cumple las Reglas 1 y 2
  3. buscar_en_web    (web)      -> Tavily, solo en sitios oficiales y solo si la base no tiene la respuesta

Las tools devuelven datos estructurados (listas/diccionarios). El agente se los pasa al LLM
como JSON y además los guarda para mostrarlos en la interfaz, en el registro y en Ragas.
"""
import os
from typing import Literal
from langchain_core.tools import tool

from busqueda import buscar, seccion_de

# ---------------------------------------------------------------------------
# Límites (en código: el LLM no los puede cambiar)
# ---------------------------------------------------------------------------
MAX_FRAGMENTOS = 5              # fragmentos máximos por llamada a buscar_en_base
MAX_CARACTERES_FRAGMENTO = 600  # longitud máxima de cada fragmento devuelto
MAX_CARACTERES_CONSULTA = 200   # longitud máxima de la consulta (base y web)

MAX_RESULTADOS_WEB = 3          # resultados máximos de buscar_en_web
MAX_CARACTERES_EXTRACTO = 300   # longitud máxima del extracto de cada resultado web
# Lista blanca de sitios: fijada aquí, el LLM no puede pasar otros dominios
SITIOS_PERMITIDOS = [
    "theifab.com",        # IFAB: organismo que redacta las Reglas de Juego
    "fifa.com",           # FIFA: organismo rector del fútbol mundial
    "inside.fifa.com",    # FIFA: circulares, reglamentos y noticias institucionales
    "conmebol.com",       # Confederación Sudamericana de Fútbol
    "uefa.com",           # Confederación Europea de Fútbol
]

DOCUMENTO = "Reglas-de-Juego-2026-27-IFAB.pdf"


# ---------------------------------------------------------------------------
# 1. Tool de búsqueda en la base de conocimientos
# ---------------------------------------------------------------------------
@tool
def buscar_en_base(consulta: str) -> list:
    """Busca en las Reglas de Juego 2026/27 de la IFAB (la base de conocimientos) los fragmentos
    más relevantes para la consulta.

    Úsala SIEMPRE que la pregunta trate sobre reglas, faltas, sanciones, reanudaciones,
    procedimientos del árbitro, el VAR, el equipamiento o cualquier contenido del reglamento.
    Si la pregunta tiene varios temas, llámala una vez por tema con una consulta corta y concreta.
    NO la uses para saludos o conversación casual.

    Devuelve como máximo 5 fragmentos; cada uno trae documento, página y sección para citar.
    """
    consulta = consulta.strip()[:MAX_CARACTERES_CONSULTA]
    resultados = []
    for frag in buscar(consulta, MAX_FRAGMENTOS):
        resultados.append({
            "origen": "base",
            "documento": frag["fuente"],
            "pagina": str(frag["pagina"]),
            "seccion": seccion_de(frag["texto"]),
            "texto": frag["texto"][:MAX_CARACTERES_FRAGMENTO],
        })
    return resultados


# ---------------------------------------------------------------------------
# 2. Tool de tarea del dominio: verificar medidas reglamentarias
# ---------------------------------------------------------------------------
# Rangos copiados de las Reglas de Juego 2026/27 (Regla 1 págs. 51-54 y Regla 2 pág. 59).
# (mínimo, máximo, unidad reglamentaria, regla, página). None = sin límite en ese lado.
MEDIDAS = {
    "general": {
        "terreno_largo": (90.0, 120.0, "m", "Regla 1 – El terreno de juego (línea de banda)", "51"),
        "terreno_ancho": (45.0, 90.0, "m", "Regla 1 – El terreno de juego (línea de meta)", "51"),
    },
    "internacional": {
        "terreno_largo": (100.0, 110.0, "m", "Regla 1 – Dimensiones en partidos internacionales (línea de banda)", "51"),
        "terreno_ancho": (64.0, 75.0, "m", "Regla 1 – Dimensiones en partidos internacionales (línea de meta)", "51"),
    },
    "comun": {
        "porteria_ancho":       (7.32, 7.32, "m", "Regla 1 – Porterías (distancia entre postes)", "54"),
        "porteria_alto":        (2.44, 2.44, "m", "Regla 1 – Porterías (altura del travesaño)", "54"),
        "ancho_lineas":         (None, 0.12, "m", "Regla 1 – Marcación del terreno (anchura de las líneas)", "51"),
        "altura_banderin":      (1.5, None, "m", "Regla 1 – Postes de banderines", "52"),
        "balon_circunferencia": (68.0, 70.0, "cm", "Regla 2 – El balón (circunferencia)", "59"),
        "balon_peso":           (410.0, 450.0, "g", "Regla 2 – El balón (peso al comienzo del partido)", "59"),
        "balon_presion":        (0.6, 1.1, "atm", "Regla 2 – El balón (presión al nivel del mar)", "59"),
    },
}
# Las porterías tienen una medida exacta: se acepta la diferencia de redondeo a 1 cm
TOLERANCIA_EXACTA = 0.005

# Factores para pasar cada unidad a la unidad reglamentaria
CONVERSIONES = {
    "m":   {"m": 1, "cm": 0.01, "yd": 0.9144, "ft": 0.3048, "in": 0.0254},
    "cm":  {"cm": 1, "m": 100, "in": 2.54, "ft": 30.48},
    "g":   {"g": 1, "kg": 1000, "oz": 28.349523125, "lb": 453.59237},
    "atm": {"atm": 1, "psi": 1 / 14.6959488, "bar": 1 / 1.01325, "kpa": 1 / 101.325, "g/cm2": 1 / 1033.227},
}
SINONIMOS_UNIDAD = {
    "metros": "m", "metro": "m", "centimetros": "cm", "centímetros": "cm", "yardas": "yd", "yarda": "yd",
    "pies": "ft", "pie": "ft", "pulgadas": "in", "pulgada": "in", "gramos": "g", "kilogramos": "kg",
    "onzas": "oz", "onza": "oz", "libras": "lb", "libra": "lb", "atmosferas": "atm", "atmósferas": "atm",
    "gcm2": "g/cm2", "g/cm²": "g/cm2",
}

Elemento = Literal[
    "terreno_largo", "terreno_ancho", "porteria_ancho", "porteria_alto", "ancho_lineas",
    "altura_banderin", "balon_circunferencia", "balon_peso", "balon_presion",
]


@tool
def verificar_medida(elemento: Elemento, valor: float, unidad: str,
                     tipo_partido: Literal["general", "internacional"] = "general") -> dict:
    """Verifica si una medida del terreno de juego, la portería o el balón cumple las Reglas de Juego,
    convirtiendo antes la unidad si hace falta. Haz el cálculo con esta tool, nunca de memoria.

    Úsala cuando el usuario dé un valor numérico y pregunte si es reglamentario, o pida convertirlo.
    Si da varias medidas (por ejemplo largo y ancho), llámala una vez por medida.

    Args:
        elemento: qué se mide (terreno_largo = línea de banda, terreno_ancho = línea de meta,
            porteria_ancho, porteria_alto, ancho_lineas, altura_banderin,
            balon_circunferencia, balon_peso, balon_presion).
        valor: número que dio el usuario.
        unidad: unidad del valor (m, cm, yd, ft, in, g, kg, oz, lb, atm, psi, bar, kpa, g/cm2).
        tipo_partido: "internacional" solo si el usuario habla de partidos internacionales;
            si no, "general". Solo cambia los límites del terreno.
    """
    unidad = SINONIMOS_UNIDAD.get(unidad.strip().lower(), unidad.strip().lower())
    # El terreno tiene límites distintos en partidos internacionales; el resto es común
    tabla = MEDIDAS[tipo_partido] if elemento in MEDIDAS[tipo_partido] else MEDIDAS["comun"]
    minimo, maximo, unidad_regla, regla, pagina = tabla[elemento]

    factores = CONVERSIONES[unidad_regla]
    if unidad not in factores:
        return {"error": f"Unidad '{unidad}' no válida para {elemento}. Usa una de: {', '.join(factores)}"}
    convertido = round(valor * factores[unidad], 4)

    if minimo is not None and minimo == maximo:
        cumple = abs(convertido - minimo) <= TOLERANCIA_EXACTA
        rango = f"exactamente {minimo} {unidad_regla}"
    else:
        cumple = (minimo is None or convertido >= minimo) and (maximo is None or convertido <= maximo)
        partes = []
        if minimo is not None:
            partes.append(f"mínimo {minimo} {unidad_regla}")
        if maximo is not None:
            partes.append(f"máximo {maximo} {unidad_regla}")
        rango = ", ".join(partes)

    return {
        "origen": "tarea",
        "elemento": elemento,
        "tipo_partido": tipo_partido if elemento.startswith("terreno") else "cualquiera",
        "valor_dado": f"{valor} {unidad}",
        "valor_convertido": f"{convertido} {unidad_regla}",
        "rango_reglamentario": rango,
        "cumple": cumple,
        "regla": regla,
        "documento": DOCUMENTO,
        "pagina": pagina,
    }


# ---------------------------------------------------------------------------
# 3. Tool de búsqueda web controlada
# ---------------------------------------------------------------------------
_tavily = None


def _cliente_tavily():
    # Se crea al primer uso: si falta TAVILY_API_KEY la app arranca igual
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily


@tool
def buscar_en_web(consulta: str) -> list:
    """Busca información PÚBLICA o ACTUAL en sitios oficiales del fútbol (IFAB, FIFA, CONMEBOL, UEFA).

    Úsala SOLO si se cumplen las dos condiciones:
      1. buscar_en_base ya se consultó y no tiene la respuesta, y
      2. la pregunta trata de información pública o actual (una circular o norma reciente,
         un cargo actual, una fecha, una competición vigente).
    NUNCA la uses para preguntas que responde el reglamento, ni copies texto de la base
    en la consulta: escribe solo una consulta corta con las palabras clave del usuario.

    Devuelve como máximo 3 resultados con título, URL y un extracto corto, marcados con origen "web".
    Los sitios permitidos están fijados en el código y no se pueden cambiar.
    """
    consulta = consulta.strip()[:MAX_CARACTERES_CONSULTA]
    try:
        respuesta = _cliente_tavily().search(
            query=consulta,
            max_results=MAX_RESULTADOS_WEB,
            include_domains=SITIOS_PERMITIDOS,
            search_depth="basic",
            include_answer=False,      # sin resumen generado por Tavily: solo resultados citables
            include_raw_content=False,
        )
    except Exception as e:
        return [{"origen": "web", "error": f"La búsqueda web no está disponible: {type(e).__name__}"}]

    resultados = []
    for r in respuesta.get("results", [])[:MAX_RESULTADOS_WEB]:
        url = r.get("url", "")
        # Segunda barrera: se descarta cualquier resultado fuera de la lista blanca
        dominio = url.split("/")[2] if url.count("/") >= 2 else ""
        if not any(dominio == s or dominio.endswith("." + s) for s in SITIOS_PERMITIDOS):
            continue
        resultados.append({
            "origen": "web",
            "titulo": r.get("title", "")[:150],
            "url": url,
            "extracto": (r.get("content") or "")[:MAX_CARACTERES_EXTRACTO],
        })
    if not resultados:
        return [{"origen": "web", "aviso": "Sin resultados en los sitios permitidos."}]
    return resultados


TOOLS = [buscar_en_base, verificar_medida, buscar_en_web]
