import os

#-------------------CONSTANTS-------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
PDF_PATH = "DOCUMENT RAG/documents"
OUTPUT_DIR = "DOCUMENT RAG/extracted_images"
CHUNK_SIZE = 600
DOCUMENTS_PATH = f"DOCUMENT RAG/embedded_db/chunk_size_{CHUNK_SIZE}/documents.jsonl"
INDEX_PATH = f"DOCUMENT RAG/embedded_db/chunk_size_{CHUNK_SIZE}/description_index.faiss" 
DESCRIPTION_DIR = "DOCUMENT RAG/image_descriptions"


#-----------------VideoRag-----------------------

