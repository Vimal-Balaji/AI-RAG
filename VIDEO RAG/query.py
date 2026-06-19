import json
import faiss
import numpy as np
import requests

from utils.embeddingClass import embeddings_model
from utils.constants import (
    DOCUMENTS_AUDIO_PATH,
    INDEX_AUDIO_PATH,
    DOCUMENTS_FRAME_PATH,
    INDEX_FRAME_PATH,
    LM_STUDIO_URL
)


# =====================================================
# Load Documents
# =====================================================

def load_jsonl(path):
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    return docs


audio_docs = load_jsonl(DOCUMENTS_AUDIO_PATH)
frame_docs = load_jsonl(DOCUMENTS_FRAME_PATH)

audio_index = faiss.read_index(INDEX_AUDIO_PATH)
frame_index = faiss.read_index(INDEX_FRAME_PATH)


# =====================================================
# Retrieve Top K
# =====================================================

def retrieve_audio(query, k=3):

    q_emb = embeddings_model.embed_query(query)

    q_emb = np.array([q_emb]).astype("float32")

    scores, indices = audio_index.search(q_emb, k)

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(audio_docs[idx])

    return results


def retrieve_frames(query, k=3):

    q_emb = embeddings_model.embed_query(query)

    q_emb = np.array([q_emb]).astype("float32")

    scores, indices = frame_index.search(q_emb, k)

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(frame_docs[idx])

    return results


# =====================================================
# Build Context
# =====================================================

def build_context(audio_chunks, frame_chunks):

    context = []

    context.append("=== AUDIO TRANSCRIPTS ===")

    for chunk in audio_chunks:

        text = chunk["video_content"]

        start = chunk["metadata"].get("start")

        end = chunk["metadata"].get("end")

        context.append(
            f"[{start}-{end}] {text}"
        )

    context.append("\n=== FRAME DESCRIPTIONS ===")

    for frame in frame_chunks:

        text = frame["video_content"]

        timestamp = frame["metadata"].get("start")

        context.append(
            f"[{timestamp}] {text}"
        )

    return "\n".join(context)


# =====================================================
# Ask Gemma
# =====================================================

def ask_llm(query, context):

    prompt = f"""
You are a Video RAG assistant.

Use BOTH:
1. Audio transcript evidence
2. Visual frame evidence

Answer ONLY from the provided context.

Context:
{context}

Question:
{query}
"""

    payload = {
        "model": "gemma-3-4b-it",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
        "reasoning_effort": "none"
    }

    response = requests.post(
        f"{LM_STUDIO_URL}",
        json=payload
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# =====================================================
# Main Video RAG Pipeline
# =====================================================

def video_rag(query):

    audio_results = retrieve_audio(query, k=3)

    frame_results = retrieve_frames(query, k=3)

    context = build_context(
        audio_results,
        frame_results
    )

    answer = ask_llm(
        query=query,
        context=context
    )

    return {
        "answer": answer,
        "audio_chunks": audio_results,
        "frame_chunks": frame_results
    }


# =====================================================
# Main Loop
# =====================================================

if __name__ == "__main__":

    print("Video RAG System - Type 'exit' or 'quit' to stop\n")

    while True:
        query = input("Question: ").strip()

        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            result = video_rag(query)
            print("\n===== ANSWER =====\n")
            print(result["answer"])

            print("\n===== AUDIO CHUNKS =====\n")
            for chunk in result["audio_chunks"]:
                print(f"[{chunk['metadata'].get('start', 'N/A')}] {chunk['video_content']}\n")

            print("\n===== FRAME CHUNKS =====\n")
            for chunk in result["frame_chunks"]:
                print(f"[{chunk['metadata'].get('start', 'N/A')}] {chunk['video_content']}\n")

            print("\n" + "="*60 + "\n")

        except Exception as e:
            print(f"Error processing query: {e}\n")