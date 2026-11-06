# import requests
# from typing import List
# from langchain_core.embeddings import Embeddings

# class LMStudioEmbeddings(Embeddings):
#     def __init__(self, model: str = "text-embedding-bge-m3"):
#         self.url   = "http://127.0.0.1:1234/v1/embeddings"
#         self.model = model

#     def _embed(self, texts: List[str]) -> List[List[float]]:
#         response = requests.post(
#             self.url,
#             json={"model": self.model, "input": texts},
#             headers={"Content-Type": "application/json"}
#         )
#         data = response.json()
#         return [
#             item["embedding"]
#             for item in sorted(data["data"], key=lambda x: x["index"])
#         ]

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self._embed(texts)

#     def embed_query(self, text: str) -> List[float]:
#         return self._embed([text])[0]

# embeddings_model = LMStudioEmbeddings(model="text-embedding-bge-m3")
import json
import boto3
from typing import List
from langchain_core.embeddings import Embeddings


class BedrockEmbeddings(Embeddings):

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: str = "ap-south-1"
    ):

        self.model_id = model_id

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:

        embeddings = []

        for text in texts:

            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json"
            )

            result = json.loads(response["body"].read())

            embeddings.append(result["embedding"])

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


# Initialize
embeddings_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1"
)