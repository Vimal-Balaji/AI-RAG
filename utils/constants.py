import os

#-------------------CONSTANTS-------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
PDF_PATH = "documents"
OUTPUT_DIR = "extracted_images"
CHUNK_SIZE=600
DOCUMENTS_PATH = os.path.join("embedded_db",f"chunk_size_{CHUNK_SIZE}","documents.jsonl")
INDEX_PATH = os.path.join("embedded_db",f"chunk_size_{CHUNK_SIZE}","description_index.faiss") 
DESCRIPTION_DIR = "image_descriptions"


#-----------------VideoRag-----------------------

