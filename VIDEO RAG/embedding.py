import os
import fitz
import numpy as np
import json
import faiss

from utils.constants import VIDEO_PATH,AUDIO_PATH,DOCUMENTS_AUDIO_PATH,DOCUMENTS_FRAME_PATH,INDEX_AUDIO_PATH,INDEX_FRAME_PATH
from utils.extract_audio import transcribe_and_chunk
from utils.extract_frames import extract_frames
from utils.embeddingClass import embeddings_model


def load_audio():
    documents= transcribe_and_chunk()
    os.makedirs(os.path.dirname(DOCUMENTS_AUDIO_PATH), exist_ok=True)
    with open(DOCUMENTS_AUDIO_PATH, "w", encoding="utf-8") as f:
        for document in documents:
            item = {
                "video_content": document["video_content"],
                "metadata": document["metadata"],
            }
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Transcript saved: {DOCUMENTS_AUDIO_PATH} ({len(documents)} chunks)\n")
    return documents

def load_frames():
    documents=extract_frames()
    os.makedirs(os.path.dirname(DOCUMENTS_FRAME_PATH), exist_ok=True)
    with open(DOCUMENTS_FRAME_PATH, "w", encoding="utf-8") as f:
        for document in documents:
            item = {
                "video_content": document["video_content"],
                "metadata": document["metadata"],
            }
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Transcript saved: {DOCUMENTS_FRAME_PATH} ({len(documents)} chunks)\n")
    return documents

def build_index(documents, index_path):
    """Build a FAISS index from a list of document dictionaries."""
    if not documents:
        raise ValueError("No documents provided to build_index.")

    texts = [doc["video_content"] for doc in documents]
    embeddings = np.array(embeddings_model.embed_documents(texts), dtype="float32")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved: {index_path} ({index.ntotal} vectors)")
    return index


if __name__=="__main__":
    doc_audio = load_audio()
    doc_frame = load_frames()
    build_index(doc_audio, INDEX_AUDIO_PATH)
    build_index(doc_frame, INDEX_FRAME_PATH)
