"""Prueba las 3 tools directamente con .invoke(), SIN el LLM (punto 1.2 del Avance 3).

Uso:  python probar_tools.py
"""
import json
from tools import buscar_en_base, verificar_medida, buscar_en_web


def mostrar(titulo, resultado):
    print("\n" + "=" * 78 + f"\n{titulo}\n" + "=" * 78)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


mostrar('1) buscar_en_base.invoke({"consulta": "cuándo se concede un tiro penal"})',
        buscar_en_base.invoke({"consulta": "cuándo se concede un tiro penal"}))

mostrar('2a) verificar_medida: presión del balón 12 psi',
        verificar_medida.invoke({"elemento": "balon_presion", "valor": 12, "unidad": "psi"}))
mostrar('2b) verificar_medida: largo de 98 m en partido internacional',
        verificar_medida.invoke({"elemento": "terreno_largo", "valor": 98, "unidad": "m",
                                 "tipo_partido": "internacional"}))

mostrar('3) buscar_en_web.invoke({"consulta": "presidente actual de la FIFA"})',
        buscar_en_web.invoke({"consulta": "presidente actual de la FIFA"}))
