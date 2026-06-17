import os

#-------------------CONSTANTS-------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
PDF_PATH = "documents"
OUTPUT_DIR = "extracted_images"
DOCUMENTS_PATH = os.path.join("embedded_db", "documents.jsonl")
INDEX_PATH = os.path.join("embedded_db", "description_index.faiss") 