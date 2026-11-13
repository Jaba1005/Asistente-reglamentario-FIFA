"""Evaluación del Avance 3: pipeline fijo (Avance 2) frente al agente.

Con las mismas preguntas y el mismo LLM ejecuta:
  1. Las 15 preguntas de evaluacion/preguntas_ragas.json en los dos sistemas.
  2. Ragas (faithfulness, answer_relevancy, context_precision, context_recall) para cada sistema.
     Los retrieved_contexts del agente son los fragmentos que devolvieron sus tools.
  3. El costo promedio por pregunta: llamadas al LLM, tokens y tiempo.
  4. Los casos de comportamiento de evaluacion/casos_comportamiento.json (esperado escrito antes).

Todo queda en evaluacion/resultados/. El script se puede cortar y volver a ejecutar:
retoma donde iba (no repite preguntas, Ragas ni casos ya hechos). Con --rehacer empieza de cero.

Uso:
    pip install -r requirements-eval.txt
    python evaluar_agente.py              # todo
    python evaluar_agente.py --solo-casos # solo los casos de comportamiento
    python evaluar_agente.py --limite 2   # prueba rápida con 2 preguntas

Límites del plan gratuito de Groq: el script espera entre llamadas (--pausa) y reintenta
si llega el error 429. Si se agota el límite diario de tokens, se puede seguir otro día
o usar otro modelo como juez de Ragas (tiene su propio límite diario):
    python evaluar_agente.py --solo-ragas          (juez por defecto: openai/gpt-oss-20b)
Si Ragas se corta, al repetir el comando solo se evalúan las preguntas que quedaron sin nota.
"""
import os
import re
import sys
import json
import time
import argparse
import unittest.mock as mock

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE, "evaluacion")
RES_DIR = os.path.join(EVAL_DIR, "resultados")
# El registro de uso de tools de la evaluación queda junto a los resultados
os.environ.setdefault("LOG_DIR", RES_DIR)

# Parche para el bug de VertexAI en ragas (igual que paquetes.py)
sys.modules["langchain_community.chat_models.vertexai"] = mock.MagicMock()
sys.modules["langchain_community.llms.vertexai"] = mock.MagicMock()

from groq import RateLimitError                         # noqa: E402
from langchain_groq import ChatGroq                     # noqa: E402
from agente import ejecutar_agente, GROQ_MODEL          # noqa: E402
from pipeline_fijo import rag_pipeline_fijo             # noqa: E402
from busqueda import embeddings_model                   # noqa: E402
from tools import SITIOS_PERMITIDOS                     # noqa: E402

METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# Juez de Ragas: otro modelo con su propio límite diario, para no gastar el del agente
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "openai/gpt-oss-20b")


class LimiteDiario(Exception):
    """Groq agotó los tokens del día para el juez. No se reintenta: hay que seguir otro día."""


_limite_diario = {"agotado": False}


class JuezGroq(ChatGroq):
    """ChatGroq que, al recibir el 429 de tokens POR DÍA (TPD), deja de llamar a la API.
    Así Ragas termina en minutos en vez de pasar horas reintentando cuando ya no hay cupo."""

    def _revisar(self, e):
        if "(TPD)" in str(e) or "tokens per day" in str(e):
            _limite_diario["agotado"] = True
            raise LimiteDiario(str(e)[:200]) from None
        raise e

    def _generate(self, *args, **kwargs):
        if _limite_diario["agotado"]:
            raise LimiteDiario("límite diario del juez agotado")
        try:
            return super()._generate(*args, **kwargs)
        except RateLimitError as e:
            self._revisar(e)

    async def _agenerate(self, *args, **kwargs):
        if _limite_diario["agotado"]:
            raise LimiteDiario("límite diario del juez agotado")
        try:
            return await super()._agenerate(*args, **kwargs)
        except RateLimitError as e:
            self._revisar(e)


def ruta(nombre):
    return os.path.join(RES_DIR, nombre)


def cargar(nombre, carpeta=EVAL_DIR, defecto=None):
    p = os.path.join(carpeta, nombre)
    if not os.path.exists(p):
        return defecto
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def guardar(nombre, contenido):
    os.makedirs(RES_DIR, exist_ok=True)
    with open(ruta(nombre), "w", encoding="utf-8") as f:
        if isinstance(contenido, str):
            f.write(contenido)
        else:
            json.dump(contenido, f, ensure_ascii=False, indent=2)
    print(f"   -> evaluacion/resultados/{nombre}")


def con_reintentos(funcion, *args, intentos=4, espera=60):
    """Ejecuta la función y, si Groq responde 429 (límite de tokens), espera y reintenta."""
    for i in range(1, intentos + 1):
        try:
            return funcion(*args)
        except RateLimitError as e:
            if i == intentos:
                raise
            print(f"   Límite de Groq (429). Reintento {i}/{intentos - 1} en {espera} s... ({str(e)[:90]})")
            time.sleep(espera)


# ---------------------------------------------------------------------------
# 1. Respuestas de los dos sistemas
# ---------------------------------------------------------------------------
def correr_preguntas(preguntas, pausa):
    filas = cargar("respuestas.json", RES_DIR, [])
    hechas = {f["pregunta"] for f in filas}
    for i, p in enumerate(preguntas, 1):
        if p["pregunta"] in hechas:
            continue
        print(f"[{i}/{len(preguntas)}] {p['pregunta'][:70]}")
        fijo = con_reintentos(rag_pipeline_fijo, p["pregunta"])
        time.sleep(pausa)
        ag = con_reintentos(ejecutar_agente, p["pregunta"])
        time.sleep(pausa)
        filas.append({"pregunta": p["pregunta"], "referencia": p["referencia"], "fijo": fijo, "agente": ag})
        guardar("respuestas.json", filas)  # se guarda después de cada pregunta
    orden = [p["pregunta"] for p in preguntas]
    return sorted([f for f in filas if f["pregunta"] in orden], key=lambda f: orden.index(f["pregunta"]))


# ---------------------------------------------------------------------------
# 2. Ragas
# ---------------------------------------------------------------------------
def evaluar_ragas(filas, sistema):
    """Ragas para un sistema. Si ya hay resultados del mismo juez, solo evalúa las preguntas
    que quedaron sin nota (NaN), por ejemplo porque Groq cortó por el límite diario de tokens."""
    import pandas as pd
    archivo = f"ragas_{sistema}.csv"
    previo = pd.DataFrame()
    if os.path.exists(ruta(archivo)):
        previo = pd.read_csv(ruta(archivo))
        if "juez" not in previo or (previo["juez"] != RAGAS_MODEL).any():
            print(f"   {archivo} es de otro juez: se recalcula completo con {RAGAS_MODEL}")
            previo = pd.DataFrame()
        else:
            previo = previo.dropna(subset=[m for m in METRICAS if m in previo])
    hechas = set(previo["user_input"]) if len(previo) else set()
    pendientes = [f for f in filas if f["pregunta"] not in hechas]
    if not pendientes:
        print(f"   {archivo}: las {len(filas)} preguntas ya tienen todas las métricas")
        return previo
    print(f"   {sistema}: evaluando {len(pendientes)} preguntas (ya completas: {len(hechas)})")

    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    datos = [{
        "user_input": f["pregunta"],
        "response": f[sistema]["respuesta"],
        # Si el agente no llamó tools, Ragas necesita al menos un contexto (vacío)
        "retrieved_contexts": f[sistema]["contextos"] or [""],
        "reference": f["referencia"],
    } for f in pendientes]

    if _limite_diario["agotado"]:
        print(f"   {sistema}: se omite hoy porque el juez ya agotó su límite diario")
        vacio = pd.DataFrame([{"sistema": sistema, "user_input": f["pregunta"]} for f in pendientes])
        return pd.concat([previo, vacio], ignore_index=True)
    # max_tokens amplio: gpt-oss razona antes de responder y con el valor por defecto a veces
    # no alcanzaba a terminar (LLMDidNotFinishException)
    juez = LangchainLLMWrapper(JuezGroq(model=RAGAS_MODEL, temperature=0.0, max_retries=2, max_tokens=8192))
    resultado = evaluate(
        dataset=EvaluationDataset.from_list(datos),
        metrics=[
            Faithfulness(),
            AnswerRelevancy(strictness=1),   # Groq no admite n>1 generaciones por llamada
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=juez,
        embeddings=LangchainEmbeddingsWrapper(embeddings_model),
        # Se reintentan los errores pasajeros (límite por minuto, timeouts), nunca LimiteDiario
        run_config=RunConfig(max_workers=1, timeout=240, max_retries=8, max_wait=60,
                             exception_types=(RateLimitError, TimeoutError, ConnectionError)),
    )
    nuevo = resultado.to_pandas().rename(columns={"llm_context_precision_with_reference": "context_precision"})
    nuevo.insert(0, "sistema", sistema)
    nuevo["juez"] = RAGAS_MODEL
    df = pd.concat([previo, nuevo], ignore_index=True)
    orden = [f["pregunta"] for f in filas]
    df = df.sort_values("user_input", key=lambda c: c.map(orden.index)).reset_index(drop=True)
    guardar(archivo, df.to_csv(index=False))
    faltan = int(df[METRICAS].isna().any(axis=1).sum())
    if _limite_diario["agotado"]:
        print(f"   AVISO: el juez {RAGAS_MODEL} agotó su límite diario de tokens en Groq.")
    if faltan:
        print(f"   AVISO: {faltan} preguntas de «{sistema}» quedaron con alguna métrica vacía. "
              "Vuelve a ejecutar el mismo comando más tarde y solo se evaluarán esas.")
    return df


def md_ragas(promedios, n, conteo):
    lineas = [f"Preguntas evaluadas: {n} · Juez de Ragas: {RAGAS_MODEL}",
              "Valores válidos por métrica (fijo / agente): " + ", ".join(
                  f"{m} {conteo['fijo'][m]}/{conteo['agente'][m]}" for m in METRICAS), "",
              "| Métrica | Avance 2 (pipeline fijo) | Avance 3 (agente) | Diferencia |", "|---|---|---|---|"]
    for m in METRICAS:
        a, b = promedios["fijo"].get(m), promedios["agente"].get(m)
        if a is None or b is None:
            lineas.append(f"| {m} | {a} | {b} | — |")
        else:
            lineas.append(f"| {m} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------------------
# 3. Costo
# ---------------------------------------------------------------------------
def costo(filas):
    def prom(sistema, campo):
        return round(sum(f[sistema][campo] for f in filas) / len(filas), 2)
    return {s: {c: prom(s, c) for c in ("llamadas_llm", "tokens_total", "tiempo_s")} for s in ("fijo", "agente")}


def md_costo(c, n):
    return (f"Promedio por pregunta sobre las mismas {n} preguntas de Ragas.\n\n"
            "| Medida (promedio por pregunta) | Pipeline fijo (Avance 2) | Agente (Avance 3) |\n|---|---|---|\n"
            f"| Llamadas al LLM | {c['fijo']['llamadas_llm']} | {c['agente']['llamadas_llm']} |\n"
            f"| Tokens | {c['fijo']['tokens_total']} | {c['agente']['tokens_total']} |\n"
            f"| Tiempo de respuesta (s) | {c['fijo']['tiempo_s']} | {c['agente']['tiempo_s']} |\n")


# ---------------------------------------------------------------------------
# 4. Casos de comportamiento
# ---------------------------------------------------------------------------
def revisar_caso(caso, r):
    """Compara lo obtenido con la regla escrita antes de ejecutar. Devuelve (cumple, motivos)."""
    regla, motivos = caso["regla"], []
    nombres = [t["tool"] for t in r["tools"]]
    texto = r["respuesta"] or ""

    if regla.get("sin_tools") and nombres:
        motivos.append(f"usó tools {nombres}")
    for tool, minimo in regla.get("min_llamadas", {}).items():
        if nombres.count(tool) < minimo:
            motivos.append(f"{tool} llamada {nombres.count(tool)} veces (mínimo {minimo})")
    for tool in regla.get("prohibidas", []):
        if tool in nombres:
            motivos.append(f"usó {tool}")
    for tool, args in regla.get("args_contienen", {}).items():
        llamadas = [t["args"] for t in r["tools"] if t["tool"] == tool]
        if not any(all(a.get(k) == v for k, v in args.items()) for a in llamadas):
            motivos.append(f"{tool} sin argumentos {args}")
    if regla.get("cita_base") and not re.search(r"p[áa]g(\.|ina)", texto, re.IGNORECASE):
        motivos.append("no cita página")
    if regla.get("fuentes_web_permitidas"):
        urls = [f["url"] for f in r["fuentes_web"]]
        if not urls:
            motivos.append("sin fuentes web")
        if any(not any(s in u for s in SITIOS_PERMITIDOS) for u in urls):
            motivos.append("URL fuera de la lista blanca")
        if "[web:" not in texto:
            motivos.append("no marca [web: URL]")
    for frase in regla.get("contiene", []):
        if frase not in texto:
            motivos.append(f"no contiene «{frase}»")
    for frase in regla.get("no_contiene", []):
        if frase in texto:
            motivos.append("filtró el prompt del sistema")
    if len(texto) > regla.get("max_caracteres_respuesta", 10**9):
        motivos.append(f"respuesta muy larga ({len(texto)} caracteres)")
    if regla.get("sin_limite") and r.get("limite"):
        motivos.append("llegó al límite de iteraciones")
    return not motivos, motivos


def correr_casos(casos, pausa):
    filas = cargar("casos_resultados.json", RES_DIR, [])
    hechos = {f["id"] for f in filas}
    for caso in casos:
        if caso["id"] in hechos:
            continue
        print(f"[caso {caso['id']}] {caso['tipo']}")
        r = con_reintentos(ejecutar_agente, caso["pregunta"])
        cumple, motivos = revisar_caso(caso, r)
        filas.append({
            "id": caso["id"], "tipo": caso["tipo"], "pregunta": caso["pregunta"], "esperado": caso["esperado"],
            "obtenido": r["respuesta"], "tools": r["tools"], "fuentes_web": r["fuentes_web"],
            "llamadas_llm": r["llamadas_llm"], "tokens": r["tokens_total"], "tiempo_s": r["tiempo_s"],
            "limite": r.get("limite"), "cumple": cumple, "motivos": motivos,
        })
        guardar("casos_resultados.json", sorted(filas, key=lambda f: f["id"]))
        time.sleep(pausa)
    return sorted(filas, key=lambda f: f["id"])


def revisar_casos_guardados():
    """Recalcula cumple/motivos de los casos ya ejecutados, sin llamar al LLM (0 tokens)."""
    reglas = {c["id"]: c for c in cargar("casos_comportamiento.json")}
    filas = cargar("casos_resultados.json", RES_DIR, [])
    for f in filas:
        r = {"respuesta": f["obtenido"], "tools": f["tools"],
             "fuentes_web": f.get("fuentes_web", []), "limite": f.get("limite")}
        f["cumple"], f["motivos"] = revisar_caso(reglas[f["id"]], r)
    filas = sorted(filas, key=lambda f: f["id"])
    guardar("casos_resultados.json", filas)
    return filas


def md_casos(filas):
    def tools_txt(ts):
        return "; ".join(f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)})" for t in ts) or "ninguna"
    lineas = ["| # | Tipo | Pregunta | Esperado | Obtenido | Tools usadas | Llam. LLM | ¿Cumple? |",
              "|---|---|---|---|---|---|---|---|"]
    for f in filas:
        obtenido = " ".join(f["obtenido"].split())[:300]
        cumple = "Sí" if f["cumple"] else "No (" + "; ".join(f["motivos"]) + ")"
        lineas.append(f"| {f['id']} | {f['tipo']} | {f['pregunta']} | {f['esperado']} | {obtenido} | "
                      f"{tools_txt(f['tools'])} | {f['llamadas_llm']} | {cumple} |")
    return "\n".join(lineas) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo-casos", action="store_true")
    parser.add_argument("--solo-ragas", action="store_true", help="preguntas + Ragas + costo, sin los casos")
    parser.add_argument("--limite", type=int, default=None, help="usar solo las primeras N preguntas")
    parser.add_argument("--pausa", type=float, default=20.0, help="segundos entre llamadas (límite de Groq)")
    parser.add_argument("--rehacer", action="store_true", help="borrar resultados anteriores y empezar de cero")
    parser.add_argument("--juez", default=None, help="modelo de Groq que califica en Ragas (por defecto openai/gpt-oss-20b)")
    parser.add_argument("--revisar-casos", action="store_true",
                        help="recalcular cumple/motivos de los casos guardados, sin llamar al LLM")
    args = parser.parse_args()
    global RAGAS_MODEL
    if args.juez:
        RAGAS_MODEL = args.juez

    if args.revisar_casos:
        filas_casos = revisar_casos_guardados()
        guardar("casos_comportamiento.md", md_casos(filas_casos))
        resumen = cargar("resumen.json", RES_DIR, {})
        resumen["casos_cumplidos"] = f"{sum(f['cumple'] for f in filas_casos)}/{len(filas_casos)}"
        guardar("resumen.json", resumen)
        print(f"\nCasos cumplidos: {resumen['casos_cumplidos']}")
        for f in filas_casos:
            print(f"  {f['id']}: {'Sí' if f['cumple'] else 'No — ' + '; '.join(f['motivos'])}")
        return

    if args.rehacer and os.path.isdir(RES_DIR):
        for nombre in os.listdir(RES_DIR):
            os.remove(ruta(nombre))
    resumen = cargar("resumen.json", RES_DIR, {})

    if not args.solo_casos:
        preguntas = cargar("preguntas_ragas.json")[:args.limite]
        print(f"\n=== 1. Respuestas del pipeline fijo y del agente ({len(preguntas)} preguntas) ===")
        filas = correr_preguntas(preguntas, args.pausa)

        print(f"\n=== 2. Ragas (juez: {RAGAS_MODEL}) ===")
        import pandas as pd
        df = pd.concat([evaluar_ragas(filas, "fijo"), evaluar_ragas(filas, "agente")], ignore_index=True)
        promedios = {s: {m: round(float(df[df.sistema == s][m].mean()), 4) for m in METRICAS if m in df}
                     for s in ("fijo", "agente")}
        conteo = {s: {m: int(df[df.sistema == s][m].notna().sum()) for m in METRICAS} for s in ("fijo", "agente")}
        guardar("ragas_comparacion.md", md_ragas(promedios, len(filas), conteo))

        print("\n=== 3. Costo ===")
        c = costo(filas)
        guardar("costo.md", md_costo(c, len(filas)))
        resumen.update({"n_preguntas": len(filas), "juez_ragas": RAGAS_MODEL, "ragas": promedios,
                        "ragas_valores_validos": conteo, "costo": c})

    if not args.solo_ragas:
        print("\n=== 4. Casos de comportamiento ===")
        filas_casos = correr_casos(cargar("casos_comportamiento.json"), args.pausa)
        guardar("casos_comportamiento.md", md_casos(filas_casos))
        resumen["casos_cumplidos"] = f"{sum(f['cumple'] for f in filas_casos)}/{len(filas_casos)}"

    guardar("resumen.json", resumen)
    print("\nListo.")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
