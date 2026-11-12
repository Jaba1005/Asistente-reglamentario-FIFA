import os
import sys
import faulthandler
from flask import Flask, render_template, request, jsonify
from agente import ejecutar_agente

# Diagnóstico: si una respuesta se cuelga, escribe en el log qué línea
# está ejecutando cada hilo (funciona aunque el proceso esté bloqueado)
faulthandler.enable()
SEGUNDOS_DIAGNOSTICO = 30

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get('message', ''))
    historial = data.get('history') or []  # [{'rol': 'user'|'bot', 'texto': ...}], lo guarda el navegador

    if not user_message.strip():
        return jsonify({'response': 'Por favor, escribe una pregunta.'}), 400

    faulthandler.dump_traceback_later(SEGUNDOS_DIAGNOSTICO, repeat=True, file=sys.stderr)
    try:
        resultado = ejecutar_agente(user_message, historial if isinstance(historial, list) else [])
        return jsonify({
            'response': resultado['respuesta'],
            'tools': resultado['tools'],
            'sources_base': resultado['fuentes_base'],
            'sources_web': resultado['fuentes_web'],
            'llm_calls': resultado['llamadas_llm'],
            'time_s': resultado['tiempo_s'],
        })
    except Exception as e:
        print(f"Error procesando la pregunta: {e!r}", flush=True)
        return jsonify({'response': 'Lo siento, tuve un problema consultando las Reglas de Juego.'}), 500
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    # Render asigna el puerto mediante la variable de entorno PORT
    puerto_render = int(os.environ.get("PORT", 5000))
    # host="0.0.0.0" permite que la app reciba conexiones desde internet
    app.run(host="0.0.0.0", port=puerto_render)
