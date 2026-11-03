# CELDA 1 — Parche para el bug de VertexAI en ragas (ejecutar primero)
import sys
import unittest.mock as mock

sys.modules['langchain_community.chat_models.vertexai'] = mock.MagicMock()
sys.modules['langchain_community.llms.vertexai'] = mock.MagicMock()

print("[OK] Parche aplicado")

import importlib.metadata

paquetes = [
    "langchain-core",
    "langchain-community",
    "langchain-huggingface",
    "langchain-groq",
    "langchain-chroma",
    "ragas",
    "sentence-transformers",
    "chromadb",
]

for paquete in paquetes:
    try:
        version = importlib.metadata.version(paquete)
        print(f"  {paquete:<30} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {paquete:<30} ❌ no instalado")


## %pip install --upgrade \
##    langchain-core \
 ##   langchain-community \
   ## langchain-text-splitters \
 ##   langchain-chroma \
 ##   langchain-huggingface \
 ##   langchain-groq \
 ##   ragas \
  ##  sentence-transformers \
  ##  chromadb \
  ##  pypdf
