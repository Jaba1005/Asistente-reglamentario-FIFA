import os
from langchain_community.document_loaders import PyPDFLoader

pdf_dir = "pdfs"
pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

print(f"PDFs disponibles en \"{pdf_dir}/\" ({len(pdf_files)} archivos):")
for i, f in enumerate(pdf_files, 1):
    size_kb = os.path.getsize(os.path.join(pdf_dir, f)) // 1024
    print(f"  [{i}] {f}  ({size_kb} KB)")

documents = []

for pdf_file in pdf_files:
    path = os.path.join(pdf_dir, pdf_file)
    loader = PyPDFLoader(path)
    pages = loader.load()
    documents.extend(pages)
    print(f"  {pdf_file}: {len(pages)} páginas cargadas")

print(f"\nTotal de páginas cargadas: {len(documents)}")

doc_ejemplo = documents[0]

print("Estructura de un objeto Document:")
print(f"  Tipo: {type(doc_ejemplo)}")
print()
print("Metadatos (metadata):")
for k, v in doc_ejemplo.metadata.items():
    print(f"  {k}: {v}")
print()
print(f"Primeros 500 caracteres de page_content:")
print("-" * 50)
print(doc_ejemplo.page_content[:500])

