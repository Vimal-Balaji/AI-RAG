from utils.embeddingClass import embeddings_model
from utils.encodeBase64 import encode_image
from utils.constants import DOCUMENTS_PATH, INDEX_PATH, LM_STUDIO_URL

import re
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

def detect_page_query(query):
    """
    Detect if the user is asking about a specific page number.
    Returns the page number (int) if found, otherwise None.

    Handles patterns like:
      - "what is in page number 10"
      - "show me page 3"
      - "contents of page 5"
      - "page no 7"
      - "page #2"
    """
    patterns = [
        r'\bpage\s*(?:number|num|no\.?|#)?\s*(\d+)\b',
        r'\bp(?:g|gs?)\.?\s*(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def get_documents_by_page(page_num):
    """Return all documents whose metadata page_num matches the requested page."""
    return [doc for doc in documents if doc.metadata.get("page_num") == page_num]

def get_available_pages():
    """Return a sorted list of all page numbers present in the document store."""
    return sorted(set(
        doc.metadata["page_num"]
        for doc in documents
        if "page_num" in doc.metadata
    ))



def generate_description(results, query_text=None, page_num=None):
    """Send retrieved images (base64 encoded) to Gemma and get a synthesized answer."""
    content = []

    # Build a context-aware prompt depending on whether this is a page lookup or a semantic search
    user_prompt = query_text if query_text else "Based on these images, provide a comprehensive answer."

    if page_num is not None:
        prompt_text = (
            f"The following content has been extracted directly from Page {page_num} of the document. "
            f"The user asked: \"{user_prompt}\". "
            f"Answer strictly based on the content of Page {page_num} provided below. "
            f"Clearly state that your answer is based on Page {page_num}. "
            f"For any images below, describe what they show and explain how they relate to Page {page_num}'s content."
        )
    else:
        prompt_text = (
            f"{user_prompt}. If you find any image relevant to the query explain the answer "
            "based on the query and the image content. If you find any image irrelevant to "
            "the query ignore that image."
        )

    content.append({"type": "text", "text": prompt_text})

    # Add each retrieved item as text or base64 image
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
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            break

        # ── Page-number query branch ──────────────────────────────────────
        page_num = detect_page_query(query)
        if page_num is not None:
            page_docs = get_documents_by_page(page_num)
            if page_docs:
                answer = generate_description(page_docs, query_text=query, page_num=page_num)
                print("\nAnswer:")
                print(answer)
            else:
                available = get_available_pages()
                print(
                    f"\n⚠️  Page {page_num} is not present in the document store.\n"
                    f"   Available pages: {available}\n"
                )
            print("\n" + "=" * 60 + "\n")
            continue

        # ── Regular RAG branch ────────────────────────────────────────────
        results = retrieve(query, top_k=5)
        answer = generate_description(results, query_text=query)
        print("\nAnswer:")
        print(answer)
        print("\n" + "=" * 60 + "\n")