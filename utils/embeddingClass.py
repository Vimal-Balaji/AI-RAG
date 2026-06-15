import requests
from typing import List
from langchain_core.embeddings import Embeddings

class LMStudioEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-bge-m3"):
        self.url   = "http://127.0.0.1:1234/v1/embeddings"
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.url,
            json={"model": self.model, "input": texts},
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        return [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

embeddings_model = LMStudioEmbeddings(model="text-embedding-bge-m3")