# Consultor de Reglas de Juego FIFA

Asistente conversacional que responde preguntas sobre las **Reglas de Juego 2026/27 de la IFAB** citando la fuente y la página del reglamento. Es un agente con *tool calling*: el modelo decide si busca en la base de conocimientos (el PDF oficial indexado), si verifica una medida reglamentaria o, para información pública o actual, si busca en sitios web oficiales.

Proyecto del curso **Desarrollo de Aplicaciones con IA**.

- **App desplegada:** https://asistente-reglamentario-fifa.onrender.com
- **Integrantes:** Jhoan Alejandro Barrera Aguilar

> En el plan gratuito de Render la app se suspende cuando no se usa: la primera pregunta puede tardar cerca de un minuto mientras arranca.

---

## Cómo funciona

```
Usuario (navegador)
   │  POST /chat  (pregunta + últimos 6 mensajes)
   ▼
app.py (Flask) ──► ejecutar_agente()  [agente.py]
                      │
                      ▼
          LLM en Groq (openai/gpt-oss-120b) con 3 tools
                      │
     ┌────────────────┼──────────────────┐
     ▼                ▼                  ▼
buscar_en_base   verificar_medida    buscar_en_web
(índice local)   (Reglas 1 y 2)      (Tavily, solo sitios oficiales)
     │                │                  │
     └──────── resultado (JSON) ─────────┘
                      │
                      ▼
   respuesta con citas: (Fuente: documento, Pág. X) o [web: URL]
   + tools usadas y fuentes por origen en la interfaz
```

1. El usuario escribe una pregunta. El navegador envía también los últimos 6 mensajes de la conversación.
2. `ejecutar_agente()` manda al LLM el System Prompt (`prompts.py`), el historial, la pregunta y la definición de las tools.
3. Si el LLM pide una tool, el programa la ejecuta y le devuelve el resultado. El ciclo se repite hasta que el LLM responde sin pedir más tools.
4. **Límites en código:** máximo 5 llamadas al LLM y 6 tools por pregunta. En la última vuelta el LLM ya no puede pedir tools y debe responder con lo que recuperó.
5. La respuesta usa solo lo que devolvieron las tools y cita según el origen. Si la información no está, responde: *«No encontré información sobre esto en la base de conocimientos.»*
6. Cada pregunta queda en un **registro** (`logs/uso_tools.jsonl` y el log de Render) con las tools, sus argumentos, las llamadas al LLM, los tokens y el tiempo.

### Tools (`tools.py`)

| Tool | Tipo | Qué hace | Límites (en código) |
|---|---|---|---|
| `buscar_en_base(consulta)` | Búsqueda | Fragmentos del reglamento más parecidos a la consulta, cada uno con documento, página y sección. | Máx. 5 fragmentos de 600 caracteres; consulta de máx. 200 caracteres. |
| `verificar_medida(elemento, valor, unidad, tipo_partido)` | Tarea del dominio | Convierte la unidad (m, yd, ft, in, cm, g, kg, oz, lb, atm, psi, bar, kPa) y verifica si la medida del terreno, la portería o el balón cumple las Reglas 1 y 2. Devuelve la regla y la página. | Solo elementos y unidades de una lista cerrada. |
| `buscar_en_web(consulta)` | Web | Información pública o actual, solo si la base no la tiene. | Máx. 3 resultados con título, URL y extracto de 300 caracteres, marcados como `web`. Solo dominios permitidos. |

**Sitios permitidos para la búsqueda web:** `theifab.com` (IFAB, autor de las Reglas de Juego), `fifa.com` e `inside.fifa.com` (FIFA), `conmebol.com` y `uefa.com` (confederaciones). La lista está fijada en el código (`include_domains` de Tavily) y además se descarta cualquier resultado de otro dominio. A la web solo sale la consulta, nunca el contenido de la base.

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
├── agente.py              # Bucle del agente, límites y registro de uso
├── tools.py               # Las 3 tools y sus límites
├── prompts.py             # System Prompt con ejemplos
├── busqueda.py            # Carga del índice y búsqueda por similitud
├── pipeline_fijo.py       # Pipeline del Avance 2 (sin tools), para comparar
├── pipelineRag.py         # Compatibilidad: rag_pipeline() llama al agente
├── probar_tools.py        # Prueba las tools sin el LLM
├── evaluar_agente.py      # Evaluación: Ragas, costo y casos de comportamiento
├── evaluacion/            # Preguntas, casos y resultados de la evaluación
├── embeddings_onnx.py     # Modelo de embeddings en ONNX (indexación y consulta)
├── gunicorn.conf.py       # Configuración para Render
├── requirements.txt       # Dependencias de la app
├── requirements-eval.txt  # Dependencias extra para la evaluación
├── pdfs/                  # Reglas de Juego 2026/27 (PDF)
├── indice/                # Índice vectorial usado por la app
├── templates/index.html   # Interfaz del chat
├── static/css/style.css
├── paso0.py … paso7.py    # Construcción paso a paso del RAG (curso)
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

## Probar las tools y evaluar

```bash
python probar_tools.py                  # invoca las 3 tools sin el LLM
pip install -r requirements-eval.txt
python evaluar_agente.py                # Ragas + costo + casos de comportamiento
```

`evaluar_agente.py` compara el pipeline fijo del Avance 2 con el agente usando las mismas 15 preguntas (`evaluacion/preguntas_ragas.json`) y el mismo LLM. También ejecuta 9 casos de comportamiento (`evaluacion/casos_comportamiento.json`) cuyo resultado esperado se escribió antes de correrlos. Los resultados quedan en `evaluacion/resultados/`. Si se corta, al volver a ejecutarlo retoma donde iba.

---

## Despliegue en Render

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app` (lee `gunicorn.conf.py`: 1 worker, 1 hilo y timeout de 180 s)
- **Variables de entorno:** `GROQ_API_KEY` y `TAVILY_API_KEY`
- Los logs del servicio muestran cada pregunta, cada tool ejecutada (`[tool] nombre(argumentos)`) y un resumen con llamadas al LLM, tokens y tiempo (`[registro] ...`).

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
- **Costo del agente:** cada pregunta puede usar varias llamadas al LLM, frente a una sola del pipeline fijo (ver `evaluacion/resultados/costo.md`).
- **Registro en Render:** el archivo `logs/uso_tools.jsonl` se borra cuando el servicio se reinicia; el registro permanente es el log del servicio.
- **Historial:** solo se envían los últimos 6 mensajes y se pierde al recargar la página.
