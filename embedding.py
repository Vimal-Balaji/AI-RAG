from utils.image_extraction import image_extraction
from utils.embeddingClass import embeddings_model
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.constants import *
from utils.generateDescription import generate_description

import os
import fitz
import numpy as np
import json
import faiss

os.makedirs("embedded_db", exist_ok=True)

# ── Initialize LM Studio Embeddings ──────────────────────────────────────────────
# embeddings_model = LMStudioEmbeddings(model="text-embedding-bge-m3")

def embed_text(text: str) -> np.ndarray:
    """Embed text using LM Studio text embeddings."""
    return np.array(embeddings_model.embed_query(text))


#--------Writing document-------------

def save_documents(documents, path=DOCUMENTS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        for document in documents:
            item = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def save_index(index, path=INDEX_PATH):
    faiss.write_index(index, path)

def build_image_documents(pdf_path=PDF_PATH, output_dir=OUTPUT_DIR):
    context = image_extraction(pdf_path, output_dir)
    for path in os.listdir(output_dir):
        full_path = os.path.join(output_dir, path)
        print("Processing:", path)
        desc = generate_description(full_path, context.get(full_path, "")[0])
        print(f"Description from image {path}:", desc)

        document = Document(
            page_content=desc,
            metadata={
                "path": full_path,
                "page_num": context.get(full_path, ["", ""])[1],
                "type": "image",
                "source":os.path.basename(pdf_path),
                "chunk_id":len(documents)+1
            },
        )
        documents.append(document)

def build_text_documents(pdf_path=PDF_PATH,chunk_size=500,chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    doc = fitz.open(pdf_path)
    #pdf_path="documents/document.pdf"
    for page_number in range(len(doc)):
        page = doc[page_number]
        page_text = page.get_text("text")
        chunks = splitter.split_text(page_text)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "path": pdf_path,
                        "page_num": page_number + 1,
                        "type": "text",
                        "source":os.path.basename(pdf_path),
                        "chunk_id":len(documents)+1
                    }
                )
            )

def build_index(documents):
    embeddings = np.array(
        embeddings_model.embed_documents([doc.page_content for doc in documents])
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    save_index(index)


if __name__ == "__main__":
    pdf_names=["document.pdf"]
    documents = []
    for pdf_name in pdf_names:
        pdf_path=os.path.join(PDF_PATH,pdf_name)
        build_image_documents(pdf_path=pdf_path)
        build_text_documents(pdf_path=pdf_path)
    save_documents(documents)
    build_index(documents)
