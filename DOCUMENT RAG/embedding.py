import os
import fitz
import numpy as np
import json
import faiss

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.constants import (
    PDF_PATH,
    OUTPUT_DIR,
    DESCRIPTION_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCUMENTS_PATH,
    INDEX_PATH,
    S3_BUCKET_NAME,
    S3_IMAGES_PREFIX,
    S3_DOCUMENTS_KEY,
    S3_INDEX_KEY,
)
from utils.image_extraction import image_extraction
from utils.embeddingClass import embeddings_model
from utils.generateDescription import load_or_generate_description
from utils.s3_storage import upload_text, upload_file, object_exists

os.makedirs(DESCRIPTION_DIR, exist_ok=True)

def embed_text(text: str) -> np.ndarray:
    """Embed text using configured Bedrock / embeddings model."""
    return np.array(embeddings_model.embed_query(text))


# ── Storage functions (S3 + local cache) ──────────────────────────────────────

def save_documents(
    docs,
    s3_key: str = S3_DOCUMENTS_KEY,
    local_path: str = DOCUMENTS_PATH,
):
    """
    Save documents as JSONL in the S3 bucket and maintain a local copy.
    """
    lines = []
    for doc in docs:
        if "path" in doc.metadata:
            doc.metadata["path"] = doc.metadata["path"].replace("\\", "/")
        item = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        lines.append(json.dumps(item, ensure_ascii=False))

    jsonl_content = "\n".join(lines) + "\n"

    # 1. Store directly in S3 bucket
    try:
        upload_text(jsonl_content, s3_key, bucket=S3_BUCKET_NAME)
        print(f"Uploaded documents database to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"Warning: Failed to upload documents.jsonl to S3: {e}")

    # 2. Save local backup copy
    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(jsonl_content)
        print(f"Saved local documents copy to {local_path}")


def save_index(
    index,
    s3_key: str = S3_INDEX_KEY,
    local_path: str = INDEX_PATH,
):
    """
    Save FAISS index file to local disk and upload directly to S3 bucket.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    faiss.write_index(index, local_path)
    print(f"FAISS index written to local cache: {local_path}")

    # Store in S3 bucket
    try:
        upload_file(local_path, s3_key, bucket=S3_BUCKET_NAME)
        print(f"Uploaded FAISS index to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"Warning: Failed to upload index to S3: {e}")


# ── Document construction ─────────────────────────────────────────────────────

def build_image_documents(pdf_path: str, output_dir: str = OUTPUT_DIR):
    """
    Extract images from PDF, upload them to S3, generate/load descriptions,
    and create image Documents with S3 references.
    """
    context = image_extraction(pdf_path, output_dir=output_dir, upload_to_s3=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_dir = os.path.join(output_dir, pdf_name).replace("\\", "/")

    if not os.path.exists(image_dir):
        return

    for image_file in os.listdir(image_dir):
        full_path = f"{image_dir}/{image_file}"
        s3_key = f"{S3_IMAGES_PREFIX}/{pdf_name}/{image_file}"
        print("Processing image chunk:", s3_key)

        # Context text lookup handles either S3 key or local path
        context_data = context.get(s3_key) or context.get(full_path, ["", ""])
        context_text = context_data[0]
        page_num = context_data[1]

        desc = load_or_generate_description(
            image_path=s3_key,
            context_text=context_text,
            pdf_name=pdf_name,
        )

        document = Document(
            page_content=desc,
            metadata={
                "path": s3_key,  # Store S3 key so downstream retrieval reads from S3
                "local_path": full_path,
                "s3_bucket": S3_BUCKET_NAME,
                "s3_key": s3_key,
                "page_num": page_num,
                "type": "image",
                "source": os.path.basename(pdf_path),
                "chunk_id": len(documents) + 1,
            },
        )
        documents.append(document)


def build_text_documents(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    Split text from PDF into chunks and append to documents list.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc[page_number]
        page_text = page.get_text("text")
        chunks = splitter.split_text(page_text)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "path": pdf_path.replace("\\", "/"),
                        "page_num": page_number + 1,
                        "type": "text",
                        "source": os.path.basename(pdf_path),
                        "chunk_id": len(documents) + 1,
                    },
                )
            )


def build_index(docs):
    """Embed documents and persist index into S3."""
    embeddings = np.array(
        embeddings_model.embed_documents([doc.page_content for doc in docs])
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    save_index(index)


# ── Main indexing pipeline ───────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_names = ["Docx1.pdf", "Docx2.pdf", "Docx3.pdf", "Docx4.pdf"]
    documents = []

    for pdf_name in pdf_names:
        # Check standard PDF paths
        pdf_path = os.path.join(PDF_PATH, pdf_name)
        if not os.path.exists(pdf_path):
            # Fallback to local 'documents' folder if run from another directory
            pdf_path = os.path.join("documents", pdf_name)

        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found locally: {pdf_path}. Skipping.")
            continue

        print(f"\n--- Indexing PDF: {pdf_path} ---")
        build_image_documents(pdf_path=pdf_path)
        build_text_documents(pdf_path=pdf_path, chunk_size=CHUNK_SIZE)

    if documents:
        print(f"\nTotal documents built: {len(documents)}")
        save_documents(documents)
        build_index(documents)
        print("Embedding and S3 upload finished successfully.")
    else:
        print("No documents were processed.")
