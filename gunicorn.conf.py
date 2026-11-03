# gunicorn lee este archivo automáticamente al arrancar (Render: gunicorn app:app)

# Un solo worker: cada worker extra carga otra copia del modelo (512 MB en Render gratis)
workers = 1
# Un solo hilo (worker "sync"): el modelo y Chroma se usan en el mismo hilo que los cargó
threads = 1

# Render gratis tiene 0.1 CPU: cargar el modelo y responder puede pasar de 30 s
timeout = 180

# NO usar preload_app: onnxruntime y Chroma crean hilos internos que no
# sobreviven al fork del worker y la app se queda colgada sin responder
preload_app = False

# Mensajes de error de gunicorn y de la app al log de Render
loglevel = "info"
capture_output = True
