import os
import sys
import faulthandler
from flask import Flask, render_template, request, jsonify
from pipelineRag import rag_pipeline

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
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message.strip():
        return jsonify({'response': 'Por favor, escribe una pregunta.'}), 400
    
    faulthandler.dump_traceback_later(SEGUNDOS_DIAGNOSTICO, repeat=True, file=sys.stderr)
    try:
        # Llama a tu IA con la pregunta del usuario
        bot_response = rag_pipeline(user_message, 10)
        return jsonify({'response': bot_response})

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