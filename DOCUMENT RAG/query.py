import boto3
import os
import re
import json
import numpy as np
import faiss

from langchain_core.documents import Document
from utils.embeddingClass import embeddings_model
from utils.constants import DOCUMENTS_PATH, INDEX_PATH

# ── Bedrock client ────────────────────────────────────────────────────────────
REGION   = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "google.gemma-3-4b-it")

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

# ── Document / index loading ──────────────────────────────────────────────────

def load():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        docs = []
        for line in f:
            data = json.loads(line)
            if "metadata" in data and "path" in data["metadata"]:
                data["metadata"]["path"] = data["metadata"]["path"].replace("\\", "/")
            docs.append(
                Document(
                    page_content=data["page_content"],
                    metadata=data["metadata"],
                )
            )
    idx = faiss.read_index(INDEX_PATH)
    return docs, idx


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3):
    q = np.array([embeddings_model.embed_query(query)], dtype="float32")
    scores, idx_arr = index.search(q, top_k)

    print("Scores:", scores)
    print("Indices:", idx_arr + 1)

    retrieved = [documents[i] for i in idx_arr[0]]

    with open("files.txt", "w", encoding="utf-8") as f:
        for rank, chunk in enumerate(retrieved, start=1):
            f.write(f"=== Chunk {rank} ===\n")
            f.write(chunk.page_content)
            f.write("\n\n")

    return retrieved


# ── Page-query helpers ────────────────────────────────────────────────────────

def detect_page_query(query: str):
    """Return the page number if the query targets a specific page, else None."""
    patterns = [
        r"\bpage\s*(?:number|num|no\.?|#)?\s*(\d+)\b",
        r"\bp(?:g|gs?)\.?\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def get_documents_by_page(page_num: int):
    return [d for d in documents if d.metadata.get("page_num") == page_num]


def get_available_pages():
    return sorted(
        {d.metadata["page_num"] for d in documents if "page_num" in d.metadata}
    )


# ── Image helpers ─────────────────────────────────────────────────────────────

def _read_image_bytes(path: str) -> bytes:
    path = path.replace("\\", "/")
    with open(path, "rb") as f:
        return f.read()


def _image_format(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    return {".jpg": "jpeg", ".jpeg": "jpeg",
            ".png": "png", ".gif": "gif", ".webp": "webp"}.get(ext, "jpeg")


# ── Bedrock generation ────────────────────────────────────────────────────────

def generate_description(results, query_text: str = None, page_num: int = None) -> str:
    """
    Build a Bedrock Converse request from retrieved chunks (text + images)
    and return the model's synthesised answer.
    """
    user_prompt = query_text or "Based on these images, provide a comprehensive answer."

    if page_num is not None:
        prompt_text = (
            f"The following content has been extracted directly from Page {page_num} of the document. "
            f'The user asked: "{user_prompt}". '
            f"Answer strictly based on the content of Page {page_num} provided below. "
            f"Clearly state that your answer is based on Page {page_num}. "
            "For any images below, describe what they show and explain how they relate "
            f"to Page {page_num}'s content."
        )
    else:
        prompt_text = (
            f"{user_prompt}. If you find any image relevant to the query explain the answer "
            "based on the query and the image content. If you find any image irrelevant to "
            "the query ignore that image."
        )

    # Build the content list for Bedrock Converse
    content = [{"text": prompt_text}]

    for item in results:
        if item.metadata["type"] != "image":
            content.append({"text": item.page_content})
        else:
            image_path = item.metadata["path"].replace("\\", "/")
            print(f"Encoding image for Bedrock: {image_path}")
            content.append(
                {
                    "image": {
                        "format": _image_format(image_path),
                        "source": {"bytes": _read_image_bytes(image_path)},
                    }
                }
            )

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 3000, "temperature": 0.2},
    )

    return response["output"]["message"]["content"][0]["text"]


# ── Main REPL ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    documents, index = load()

    print("AWS Bedrock RAG ready. Type 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        # ── Page-number branch ────────────────────────────────────────────
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
                    f"\n⚠️  Page {page_num} is not in the document store.\n"
                    f"   Available pages: {available}\n"
                )
            print("\n" + "=" * 60 + "\n")
            continue

        # ── Semantic RAG branch ───────────────────────────────────────────
        results = retrieve(query, top_k=5)
        answer  = generate_description(results, query_text=query)
        print("\nAnswer:")
        print(answer)
        print("\n" + "=" * 60 + "\n")