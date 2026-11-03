import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from langchain_core.embeddings import Embeddings

# Mismo modelo multilingüe, pero en ONNX cuantizado (int8): sin PyTorch,
# cabe en los 512 MB de Render. Indexación y consulta DEBEN usar esta clase.
REPO_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"
MAX_LENGTH = 128

# Tokens especiales del tokenizador XLM-RoBERTa
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class MultilingualOnnxEmbeddings(Embeddings):
    def __init__(self, batch_size=16):
        self.batch_size = batch_size

        # sentencepiece usa ~60 MB de RAM; tokenizer.json (librería tokenizers) ~250 MB
        self.sp = spm.SentencePieceProcessor(
            model_file=hf_hub_download(REPO_ID, "sentencepiece.bpe.model")
        )

        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = 1
        opciones.enable_cpu_mem_arena = False
        # Con el nivel por defecto (ALL) onnxruntime convierte los pesos int8
        # a float32 al cargar y el modelo pasa de ~140 MB a ~390 MB de RAM
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(
            hf_hub_download(REPO_ID, ONNX_FILE),
            sess_options=opciones,
            providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _tokenize(self, text):
        # Igual que XLMRobertaTokenizer: ids de sentencepiece desplazados en 1
        ids = [UNK_ID if i == 0 else i + 1 for i in self.sp.encode(text)]
        return [BOS_ID] + ids[:MAX_LENGTH - 2] + [EOS_ID]

    def _embed(self, texts):
        tokenizados = [self._tokenize(t) for t in texts]
        largo = max(len(t) for t in tokenizados)
        input_ids = np.full((len(texts), largo), PAD_ID, dtype=np.int64)
        attention_mask = np.zeros((len(texts), largo), dtype=np.int64)
        for i, ids in enumerate(tokenizados):
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        token_embeddings = self.session.run(None, inputs)[0]

        # Mean pooling + normalización (igual que sentence-transformers)
        mask = attention_mask[..., None].astype(np.float32)
        vectores = (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        vectores /= np.linalg.norm(vectores, axis=1, keepdims=True)
        return vectores.tolist()

    def embed_documents(self, texts):
        resultado = []
        for i in range(0, len(texts), self.batch_size):
            resultado.extend(self._embed(texts[i:i + self.batch_size]))
        return resultado

    def embed_query(self, text):
        return self._embed([text])[0]
