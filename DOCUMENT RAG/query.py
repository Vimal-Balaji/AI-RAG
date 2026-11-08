import boto3
import os
import re
import json
import numpy as np
import faiss

from langchain_core.documents import Document
from utils.embeddingClass import embeddings_model
from utils.constants import (
    DOCUMENTS_PATH,
    INDEX_PATH,
    S3_BUCKET_NAME,
    S3_DOCUMENTS_KEY,
    S3_INDEX_KEY,
    S3_FILES_TXT_KEY,
    AWS_REGION,
    BEDROCK_MODEL_ID,
)
from utils.s3_storage import (
    read_image_bytes,
    object_exists,
    read_text,
    download_file,
    upload_text,
)

# ── Bedrock client ────────────────────────────────────────────────────────────
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
)

# ── Document / index loading (S3-first with local fallback) ────────────────────

def load(
    s3_docs_key: str = S3_DOCUMENTS_KEY,
    s3_idx_key: str = S3_INDEX_KEY,
    local_docs_path: str = DOCUMENTS_PATH,
    local_index_path: str = INDEX_PATH,
):
    """
    Load documents and FAISS index from S3 (with local fallback if offline).
    """
    docs = []

    # 1. Load documents.jsonl
    loaded_from_s3 = False
    try:
        if object_exists(s3_docs_key, bucket=S3_BUCKET_NAME):
            print(f"Loading documents database from S3: s3://{S3_BUCKET_NAME}/{s3_docs_key}")
            raw_text = read_text(s3_docs_key, bucket=S3_BUCKET_NAME)
            for line in raw_text.strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                if "metadata" in data and "path" in data["metadata"]:
                    data["metadata"]["path"] = data["metadata"]["path"].replace("\\", "/")
                docs.append(
                    Document(
                        page_content=data["page_content"],
                        metadata=data["metadata"],
                    )
                )
            loaded_from_s3 = True
    except Exception as e:
        print(f"Notice: S3 documents check encountered: {e}")

    if not loaded_from_s3:
        if os.path.exists(local_docs_path):
            print(f"Loading documents database from local file: {local_docs_path}")
            with open(local_docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "metadata" in data and "path" in data["metadata"]:
                        data["metadata"]["path"] = data["metadata"]["path"].replace("\\", "/")
                    docs.append(
                        Document(
                            page_content=data["page_content"],
                            metadata=data["metadata"],
                        )
                    )
        else:
            raise FileNotFoundError(
                f"Documents not found in S3 (s3://{S3_BUCKET_NAME}/{s3_docs_key}) "
                f"or locally ({local_docs_path}). Please run embedding.py first."
            )

    # 2. Load FAISS index
    index_loaded_from_s3 = False
    try:
        if object_exists(s3_idx_key, bucket=S3_BUCKET_NAME):
            print(f"Downloading FAISS index from S3: s3://{S3_BUCKET_NAME}/{s3_idx_key}")
            download_file(s3_idx_key, local_index_path, bucket=S3_BUCKET_NAME)
            index_loaded_from_s3 = True
    except Exception as e:
        print(f"Notice: S3 index download encountered: {e}")

    if not index_loaded_from_s3 and not os.path.exists(local_index_path):
        raise FileNotFoundError(
            f"FAISS index not found in S3 (s3://{S3_BUCKET_NAME}/{s3_idx_key}) "
            f"or locally ({local_index_path}). Please run embedding.py first."
        )

    print(f"Reading FAISS index from: {local_index_path}")
    idx = faiss.read_index(local_index_path)
    return docs, idx


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3):
    q = np.array([embeddings_model.embed_query(query)], dtype="float32")
    scores, idx_arr = index.search(q, top_k)

    print("Scores:", scores)
    print("Indices:", idx_arr + 1)

    retrieved = [documents[i] for i in idx_arr[0]]

    # Format retrieval log
    log_content = []
    for rank, chunk in enumerate(retrieved, start=1):
        log_content.append(f"=== Chunk {rank} ===\n{chunk.page_content}\n\n")
    full_log = "".join(log_content)

    # Write locally
    with open("files.txt", "w", encoding="utf-8") as f:
        f.write(full_log)

    # Store in S3 bucket
    try:
        upload_text(full_log, S3_FILES_TXT_KEY, bucket=S3_BUCKET_NAME)
        print(f"Retrieval log uploaded to S3: s3://{S3_BUCKET_NAME}/{S3_FILES_TXT_KEY}")
    except Exception as e:
        print(f"Notice: Could not upload retrieval log to S3: {e}")

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

def _image_format(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    return {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".webp": "webp",
    }.get(ext, "jpeg")


# ── Bedrock generation ────────────────────────────────────────────────────────

def generate_description(results, query_text: str = None, page_num: int = None) -> str:
    """
    Build a Bedrock Converse request from retrieved chunks (text + images)
    and return the model's synthesised answer.
    Image bytes are loaded directly from S3 or local storage.
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
        if item.metadata.get("type") != "image":
            content.append({"text": item.page_content})
        else:
            image_identifier = item.metadata.get("s3_key") or item.metadata.get("path")
            print(f"Reading image for Bedrock from S3/local: {image_identifier}")
            img_bytes = read_image_bytes(image_identifier, bucket=S3_BUCKET_NAME)

            content.append(
                {
                    "image": {
                        "format": _image_format(image_identifier),
                        "source": {"bytes": img_bytes},
                    }
                }
            )

    response = bedrock.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 3000, "temperature": 0.2},
    )

    return response["output"]["message"]["content"][0]["text"]


# ── Main REPL ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    documents, index = load()

    print(f"AWS Bedrock RAG ready (S3 Bucket: {S3_BUCKET_NAME}). Type 'exit' to quit.\n")

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
        answer = generate_description(results, query_text=query)
        print("\nAnswer:")
        print(answer)
        print("\n" + "=" * 60 + "\n")
