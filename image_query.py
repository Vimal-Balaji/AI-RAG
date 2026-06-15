from utils.embeddingClass import embeddings_model
from utils.encodeBase64 import encode_image
from utils.constants import DOCUMENTS_PATH, INDEX_PATH,LM_STUDIO_URL

import requests
import numpy as np
import os
import json
from langchain_core.documents import Document
import faiss

def load():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = [
            Document(page_content=json.loads(line)["page_content"],
                     metadata=json.loads(line)["metadata"])
            for line in f
        ]

    index = faiss.read_index(INDEX_PATH)
    return documents, index

def retrieve(query, top_k=3):
    q = embeddings_model.embed_query(query)
    q = np.array([q]).astype("float32")
    scores, idx = index.search(q, top_k)
    print("Scores:", scores)
    print("Indices:", idx)
    return [documents[i] for i in idx[0]]

def generate_description(results, query_text=None):
    """Send retrieved images (base64 encoded) to Gemma and get a synthesized answer."""
    content = []
    
    # Add the user query as text
    user_prompt = query_text if query_text else "Based on these images, provide a comprehensive answer."
    content.append({
        "type": "text",
        "text": f"{user_prompt}.If you find any image relevant to the query explain the answer based on the query and the image content. If you find any image irrelevant to the query ignore that image."
    })
    
    # Add each image as base64
    for item in results:
        if item.metadata["type"] != "image":
            content.append({
                "type": "text",
                "text": item.page_content
            })
        else:
            image_path = item.metadata["path"]
            print(f"Encoding image for description: {image_path}")
            img_b64 = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })
    
    payload = {
        "model": "google/gemma-4-4b",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.2,
        "max_tokens": 3000
    }
    
    response = requests.post(LM_STUDIO_URL, json=payload)
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    documents, index = load()
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            break
        results = retrieve(query, top_k=5)
        answer = generate_description(results, query_text=query)
        print("\nAnswer:")
        print(answer)
        print("\n" + "="*60 + "\n")
