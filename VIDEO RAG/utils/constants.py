import os

#-------------------CONSTANTS-------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
VIDEO_PATH = r"VIDEO RAG\data\video"
FRAMES_PATH = r"VIDEO RAG\data\frame"
AUDIO_PATH=r"VIDEO RAG\data\audio"
DOCUMENTS_AUDIO_PATH = os.path.join("VIDEO RAG","embedded_db","audio","documents.jsonl")
DOCUMENTS_FRAME_PATH = os.path.join("VIDEO RAG","embedded_db","frame","documents.jsonl")
INDEX_AUDIO_PATH = os.path.join("VIDEO RAG","embedded_db","audio","description_index.faiss") 
INDEX_FRAME_PATH = os.path.join("VIDEO RAG","embedded_db","frame","description_index.faiss") 
DESCRIPTION_DIR = "VIDEO RAG\data\image_descriptions"


