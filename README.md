# Consultor de Reglas de Juego FIFA

Asistente conversacional que responde preguntas sobre las **Reglas de Juego 2026/27 de la IFAB** citando la fuente y la página del reglamento. Es un agente con *tool calling*: el modelo decide si busca en la base de conocimientos (el PDF oficial indexado) o, para información actual, en la web.

Proyecto del curso **Desarrollo de Aplicaciones con IA**.

- **App desplegada:** https://asistente-reglamentario-fifa.onrender.com
- **Integrantes:** _(nombres del grupo)_

> En el plan gratuito de Render la app se suspende cuando no se usa: la primera pregunta puede tardar cerca de un minuto mientras arranca.

---

## Cómo funciona

```
Usuario (navegador)
   │  POST /chat
   ▼
app.py (Flask) ──► rag_pipeline()  [pipelineRag.py]
                      │
                      ▼
             LLM en Groq (openai/gpt-oss-120b) con 2 tools
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
 buscar_reglamentos           web_search
 (índice local del PDF)       (Tavily, hechos actuales)
          │                        │
          └──── resultado ─────────┘
                      │
                      ▼
       el LLM responde citando (Fuente: documento, Pág. X)
```

1. El usuario escribe una pregunta en la interfaz web.
2. `rag_pipeline()` la envía al LLM junto con el System Prompt y la definición de las tools.
3. Si el LLM pide una tool, el programa la ejecuta y le devuelve el resultado. Este ciclo se repite hasta que el LLM responde sin pedir más tools, con un máximo de **5 iteraciones** (`MAX_ITERACIONES`).
4. La respuesta usa solo la información devuelta por las tools y cita documento y página. Si la información no está, responde: *«No encontré información sobre esto en la base de conocimientos.»*

### Tools

| Tool | Tipo | Qué hace |
|---|---|---|
| `buscar_reglamentos(consulta)` | Búsqueda | Devuelve los 10 fragmentos del reglamento más parecidos a la consulta (similitud coseno), cada uno con fuente y página. |
| `web_search(query)` | Web | Busca en internet con Tavily (máximo 5 resultados) para preguntas sobre hechos actuales. |

### Base de conocimientos

- **Documento:** `pdfs/Reglas-de-Juego-2026-27-IFAB.pdf`, la versión oficial en español.
- **Fragmentación:** 500 caracteres con 50 de solapamiento (`paso4.py`). Cada fragmento lleva como prefijo la sección a la que pertenece. En total son 788 fragmentos.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` en ONNX cuantizado (`embeddings_onnx.py`). Corre en CPU sin PyTorch y cabe en los 512 MB de Render.
- **Índice:** `indice/vectores.npy` (vectores) + `indice/fragmentos.json` (texto, fuente y página). La búsqueda se hace con numpy.

---

## Tecnologías

| Componente | Herramienta |
|---|---|
| Interfaz y servidor | Flask + HTML/CSS/JS (`templates/`, `static/`) |
| LLM | Groq, `openai/gpt-oss-120b` (vía `langchain-groq`) |
| Búsqueda web | Tavily (`tavily-python`) |
| Embeddings | ONNX Runtime + SentencePiece |
| Despliegue | Render + gunicorn |

---

## Estructura del repositorio

```
├── app.py                 # Servidor Flask: página principal y endpoint /chat
├── pipelineRag.py         # Agente: tools, System Prompt y bucle de tool calling
├── embeddings_onnx.py     # Modelo de embeddings en ONNX (indexación y consulta)
├── gunicorn.conf.py       # Configuración para Render
├── requirements.txt       # Dependencias de la app
├── pdfs/                  # Reglas de Juego 2026/27 (PDF)
├── indice/                # Índice vectorial usado por la app
├── templates/index.html   # Interfaz del chat
├── static/css/style.css
├── paso0.py … paso7.py    # Construcción paso a paso del RAG (curso)
├── paquetes.py            # Evaluación con Ragas (Avance 2)
└── taller_multimodal/     # Taller de visión y audio (ver más abajo)
```

---

## Ejecutar en local

Requisitos: Python 3.13 y claves de [Groq](https://console.groq.com) y [Tavily](https://tavily.com).

```bash
git clone https://github.com/Jaba1005/Asistente-reglamentario-FIFA.git
cd Asistente-reglamentario-FIFA

python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # Linux / Mac

pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz. Ya está en `.gitignore`, así que nunca se sube:

```
GROQ_API_KEY=tu_clave_de_groq
TAVILY_API_KEY=tu_clave_de_tavily
```

Inicia la app:

```bash
python app.py
```

y abre http://localhost:5000. La primera vez descarga el modelo de embeddings, unos 120 MB.

> Si PowerShell no deja activar el entorno: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Reindexar el PDF (opcional)

Solo es necesario si cambia el documento. Hay que instalar las dependencias adicionales y ejecutar `paso4.py`:

```bash
pip install chromadb langchain-chroma langchain-community langchain-text-splitters pypdf
python paso4.py
```

---

## Despliegue en Render

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app` (lee `gunicorn.conf.py`: 1 worker, 1 hilo y timeout de 180 s)
- **Variables de entorno:** `GROQ_API_KEY` y `TAVILY_API_KEY`
- Los logs del servicio muestran cada pregunta y cada tool que se ejecuta (`[tool] nombre(argumentos)`).

---

## Taller: entrada multimodal y audio

La carpeta `taller_multimodal/` aplica visión y voz a texto al mismo dominio del proyecto.

```
taller_multimodal/
├── parte1_vision.py      # Parte 1: visión
├── parte2_audio.py       # Parte 2: audio → transcripción → respuesta del asistente
├── comun.py              # Funciones compartidas (cliente Groq, imágenes en base64, guardado)
├── valores_reales.json   # Valores leídos a mano para medir la extracción
├── imagenes/             # Páginas del PDF del reglamento usadas como imágenes
├── audios/               # Audios grabados + audios.json (hablante, condición, lo que se dijo)
└── resultados/           # Salida de cada punto en JSON
```

Usa las mismas dependencias y el mismo `.env` del proyecto. Se puede cambiar de modelo con variables de entorno opcionales:

| Variable | Por defecto |
|---|---|
| `VISION_MODEL` | `qwen/qwen3.8-27b` |
| `AUDIO_MODEL` | `whisper-large-v3-turbo` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` |

### Parte 1: visión

```bash
cd taller_multimodal
python parte1_vision.py              # los 4 puntos
python parte1_vision.py pregunta     # 1.1 pregunta sobre el diagrama del terreno de juego
python parte1_vision.py extraccion   # 1.2 extracción en JSON de 3 páginas + % de aciertos
python parte1_vision.py varias       # 1.3 tres imágenes de señales arbitrales en un mensaje
python parte1_vision.py error        # 1.4 error provocado: 4 imágenes (el límite es 3)
```

- Las imágenes son páginas del PDF del proyecto: el diagrama del terreno, las páginas de las Reglas 2, 3 y 7, y las señales del árbitro y del asistente.
- El esquema de extracción tiene 6 campos y está definido con Pydantic (`PaginaRegla`). Si la respuesta no es JSON o no cumple el esquema, el programa lo informa y sigue con la siguiente imagen.
- El punto 1.4 tiene otras variantes: `--tipo sin_vision` (imagen a un modelo solo de texto) y `--tipo mime` (tipo de imagen mal declarado).

### Parte 2: audio

1. Copiar los audios (10 a 30 s; wav, mp3, m4a, ogg o webm; menos de 25 MB) en `audios/`.
2. Completar `audios/audios.json` con el archivo, el hablante, la condición (con o sin ruido) y lo que se dijo.
3. Ejecutar:

```bash
python parte2_audio.py                    # transcribe y responde con el asistente del proyecto
python parte2_audio.py --solo-transcribir
python parte2_audio.py --solo-responder   # reutiliza las transcripciones guardadas
python parte2_audio.py --vocabulario      # pasa a Whisper términos del dominio, para comparar
python parte2_audio.py --modo simple      # responde con el LLM directo, sin la base de conocimientos
```

La transcripción se hace en español (`language="es"`) y la respuesta la da `pipelineRag.rag_pipeline`, el mismo asistente de la app. Por el límite de tokens por minuto de Groq, el script espera 30 s entre preguntas, reintenta solo si recibe el error 429 y guarda después de cada audio.

### Resultados

| Punto | Resultado |
|---|---|
| 1.1 Pregunta sobre el diagrama del terreno | Correcta: 11 m, 9.15 m y 7.32 m (con equivalencia en yardas) |
| 1.2 Extracción estructurada | 18 de 18 campos (100 %), los 3 JSON válidos según el esquema |
| 1.3 Tres imágenes de señales arbitrales | 11 de 11 nombres correctos; 5 gestos mal descritos |
| 1.4 Error provocado (4 imágenes) | `BadRequestError` 400: *Too many images provided. This model supports up to 3 images* |
| 2.1 Transcripción de 4 audios | 4 de 4 con el sentido intacto, incluido el audio con ruido |
| 2.2 Respuesta del asistente | 2 correctas, 1 parcial y 1 sin respuesta (límite de iteraciones del agente) |

---

## Limitaciones conocidas

- **Plan gratuito de Groq:** 8.000 tokens por minuto en `openai/gpt-oss-120b`. Si se hacen varias preguntas seguidas puede aparecer el error 429 (*rate limit*).
- **Preguntas con varias partes:** a veces el agente agota las 5 iteraciones buscando y responde que se alcanzó el límite de llamadas a herramientas.
- **Búsqueda web:** no tiene una lista de sitios permitidos, así que los resultados pueden venir de cualquier página.
- **Historial:** cada pregunta se procesa por separado; el servidor no recibe el historial de la conversación.
