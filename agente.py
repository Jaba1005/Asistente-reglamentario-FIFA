"""Agente del Avance 3: el LLM decide si llama una tool; el programa la ejecuta y le devuelve el resultado.

Bucle manual (sin create_agent) para controlar los límites y registrar cada paso:

    pregunta -> LLM (con tools) -> ¿pidió tools? --sí--> ejecutar tools -> resultado al LLM -> ...
                                               |--no--> respuesta final

Límites por pregunta (en código):
  - MAX_LLAMADAS_LLM: llamadas máximas al LLM. La última se hace SIN tools: se le pasa al LLM
    lo que ya recuperó y debe responder con eso (antes se quedaba sin respuesta).
  - MAX_LLAMADAS_TOOLS: tools máximas ejecutadas por pregunta (evita bucles de búsqueda).

Cada pregunta queda registrada en logs/uso_tools.jsonl (y en el log de Render).
"""
import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import BadRequestError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from busqueda import log
from tools import TOOLS
from prompts import SYSTEM_PROMPT

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_LLAMADAS_LLM = 5
MAX_LLAMADAS_TOOLS = 6
MAX_MENSAJES_HISTORIAL = 6        # últimos mensajes de la conversación que se reenvían al LLM
MAX_CARACTERES_HISTORIAL = 1500   # por mensaje del historial

LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
LOG_FILE = os.path.join(LOG_DIR, "uso_tools.jsonl")

if not os.getenv("GROQ_API_KEY"):
    log("ADVERTENCIA: falta la variable de entorno GROQ_API_KEY")
if not os.getenv("TAVILY_API_KEY"):
    log("ADVERTENCIA: falta TAVILY_API_KEY (buscar_en_web responderá que no está disponible)")

llm = ChatGroq(model=GROQ_MODEL, temperature=0.0, timeout=60, max_retries=3)
llm_con_tools = llm.bind_tools(TOOLS)
# Para la respuesta final forzada se usa el LLM sin tools. Con tool_choice="none" gpt-oss
# a veces intentaba llamar una tool igual y Groq respondía error 400 (tool_use_failed).
llm_sin_tools = llm
MAX_CARACTERES_RESULTADOS_FINAL = 8000  # lo recuperado que se le pasa en la respuesta forzada
TOOLS_POR_NOMBRE = {t.name: t for t in TOOLS}


def _tokens(mensaje):
    uso = getattr(mensaje, "usage_metadata", None) or {}
    return uso.get("input_tokens", 0), uso.get("output_tokens", 0)


def _historial_a_mensajes(historial):
    """Convierte el historial del chat [{'rol': 'user'|'bot', 'texto': ...}] en mensajes."""
    mensajes = []
    for m in (historial or [])[-MAX_MENSAJES_HISTORIAL:]:
        texto = str(m.get("texto", ""))[:MAX_CARACTERES_HISTORIAL]
        if not texto.strip():
            continue
        mensajes.append(HumanMessage(texto) if m.get("rol") == "user" else AIMessage(texto))
    return mensajes


def guardar_registro(registro):
    """Una línea JSON por pregunta: tools, argumentos, llamadas al LLM, tokens y tiempo."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError as e:
        log(f"No se pudo escribir el registro: {e!r}")
    resumen = ", ".join(f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)})" for t in registro["tools"]) or "ninguna"
    log(f"[registro] llamadas_llm={registro['llamadas_llm']} tokens={registro['tokens_total']} "
        f"tiempo={registro['tiempo_s']}s tools: {resumen}")


def _unicos(lista):
    """Quita elementos repetidos conservando el orden."""
    vistos, salida = set(), []
    for x in lista:
        clave = json.dumps(x, sort_keys=True, ensure_ascii=False)
        if clave not in vistos:
            vistos.add(clave)
            salida.append(x)
    return salida


def _respuesta_forzada(pregunta, historial, resultados, fecha):
    """Última llamada sin tools: la pregunta + lo que devolvieron las tools, en un solo mensaje."""
    recuperado = "\n\n".join(resultados)[:MAX_CARACTERES_RESULTADOS_FINAL] or "(no se recuperó nada)"
    mensajes = [SystemMessage(SYSTEM_PROMPT.format(fecha=fecha))]
    mensajes += _historial_a_mensajes(historial)
    mensajes.append(HumanMessage(
        f"{pregunta}\n\n---\nYa no puedes usar herramientas. Responde SOLO con esta información que "
        f"devolvieron las herramientas, citando como indican tus reglas:\n{recuperado}"))
    return llm_sin_tools.invoke(mensajes)


def ejecutar_agente(pregunta, historial=None):
    """Responde una pregunta con el agente. Devuelve la respuesta y todo lo que hizo para llegar a ella."""
    inicio = time.time()
    log(f"Pregunta recibida: {pregunta[:80]!r}")

    fecha = time.strftime("%Y-%m-%d")
    mensajes = [SystemMessage(SYSTEM_PROMPT.format(fecha=fecha))]
    mensajes += _historial_a_mensajes(historial)
    mensajes.append(HumanMessage(pregunta))

    tools_usadas, fuentes_base, fuentes_web, contextos, resultados = [], [], [], [], []
    llamadas_llm = tokens_in = tokens_out = 0
    respuesta_final, limite = None, None

    while respuesta_final is None:
        ultima = llamadas_llm == MAX_LLAMADAS_LLM - 1 or len(tools_usadas) >= MAX_LLAMADAS_TOOLS
        respuesta = None
        if not ultima:
            try:
                respuesta = llm_con_tools.invoke(mensajes)
            except BadRequestError as e:
                # Groq rechaza a veces una llamada a tool mal formada: se responde con lo ya recuperado
                if "tool_use_failed" not in str(e):
                    raise
                log(f"Groq rechazó una llamada a tool ({str(e)[:80]}); se fuerza la respuesta final")
                ultima = True
        if ultima:
            if tools_usadas:
                limite = "Se alcanzó el límite de iteraciones; se respondió con lo ya recuperado."
            respuesta = _respuesta_forzada(pregunta, historial, resultados, fecha)
        llamadas_llm += 1
        t_in, t_out = _tokens(respuesta)
        tokens_in, tokens_out = tokens_in + t_in, tokens_out + t_out
        mensajes.append(respuesta)

        if ultima or not respuesta.tool_calls:
            respuesta_final = respuesta.content or ""
            break

        # Ejecutar cada tool que pidió el LLM y devolverle el resultado
        for llamada in respuesta.tool_calls:
            nombre, args = llamada["name"], llamada["args"]
            herramienta = TOOLS_POR_NOMBRE.get(nombre)
            if herramienta is None or len(tools_usadas) >= MAX_LLAMADAS_TOOLS:
                resultado = {"error": "Herramienta no disponible o límite de herramientas alcanzado."}
            else:
                log(f"[tool] {nombre}({args})")
                try:
                    resultado = herramienta.invoke(args)
                except Exception as e:  # argumentos inválidos, etc.: se le informa al LLM
                    resultado = {"error": f"{type(e).__name__}: {e}"}
            tools_usadas.append({"tool": nombre, "args": args})

            # Guardar fuentes por origen (interfaz) y contextos (Ragas)
            items = resultado if isinstance(resultado, list) else [resultado]
            for r in items:
                if not isinstance(r, dict) or "error" in r:
                    continue
                if r.get("origen") == "base":
                    fuentes_base.append({"documento": r["documento"], "pagina": r["pagina"], "seccion": r["seccion"]})
                    contextos.append(r["texto"])
                elif r.get("origen") == "web" and r.get("url"):
                    fuentes_web.append({"titulo": r["titulo"], "url": r["url"]})
                    contextos.append(f"[web] {r['titulo']}: {r['extracto']}")
                elif r.get("origen") == "tarea":
                    contextos.append(json.dumps(r, ensure_ascii=False))

            texto_resultado = json.dumps(resultado, ensure_ascii=False)
            resultados.append(f"[{nombre}] {texto_resultado}")
            mensajes.append(ToolMessage(content=texto_resultado, tool_call_id=llamada["id"]))

    tiempo = round(time.time() - inicio, 2)
    guardar_registro({
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pregunta": pregunta,
        "tools": tools_usadas,
        "llamadas_tools": len(tools_usadas),
        "llamadas_llm": llamadas_llm,
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "tiempo_s": tiempo,
        "limite": limite,
    })

    return {
        "respuesta": respuesta_final,
        "tools": tools_usadas,
        "fuentes_base": _unicos(fuentes_base),
        "fuentes_web": _unicos(fuentes_web),
        "contextos": contextos,
        "llamadas_llm": llamadas_llm,
        "tokens_total": tokens_in + tokens_out,
        "tiempo_s": tiempo,
        "limite": limite,
    }
