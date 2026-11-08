import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ------------------- BASE DIRECTORIES -------------------
# Root folder of DOCUMENT RAG project
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------- AWS / S3 CONFIGURATION -------------------
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "my-document-rag-bucket")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "ap-south-1"))
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")

def get_s3_key(subpath: str) -> str:
    """Helper to prefix an S3 key with S3_PREFIX if configured."""
    clean_subpath = subpath.lstrip("/")
    return f"{S3_PREFIX}/{clean_subpath}" if S3_PREFIX else clean_subpath

# ------------------- MODEL & RAG CONSTANTS -------------------
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "google.gemma-3-4b-it")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ------------------- LOCAL FILE PATHS -------------------
PDF_PATH = str(BASE_DIR / "documents")
OUTPUT_DIR = str(BASE_DIR / "extracted_images")
DESCRIPTION_DIR = str(BASE_DIR / "image_descriptions")
EMBEDDED_DB_DIR = str(BASE_DIR / "embedded_db" / f"chunk_size_{CHUNK_SIZE}")
DOCUMENTS_PATH = str(BASE_DIR / "embedded_db" / f"chunk_size_{CHUNK_SIZE}" / "documents.jsonl")
INDEX_PATH = str(BASE_DIR / "embedded_db" / f"chunk_size_{CHUNK_SIZE}" / "description_index.faiss")

# ------------------- S3 BUCKET KEYS & PREFIXES -------------------
S3_DOCUMENTS_PREFIX = get_s3_key("documents")
S3_IMAGES_PREFIX = get_s3_key("extracted_images")
S3_DESCRIPTIONS_PREFIX = get_s3_key("image_descriptions")
S3_EMBEDDED_DB_PREFIX = get_s3_key(f"embedded_db/chunk_size_{CHUNK_SIZE}")
S3_DOCUMENTS_KEY = f"{S3_EMBEDDED_DB_PREFIX}/documents.jsonl"
S3_INDEX_KEY = f"{S3_EMBEDDED_DB_PREFIX}/description_index.faiss"
S3_FILES_TXT_KEY = get_s3_key("retrieval_logs/files.txt")
