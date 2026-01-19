import os
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import *

def load_pdf_multimodal(folder, image_dir):
    docs = []
    for file in os.listdir(folder):
        if not file.lower().endswith(".pdf"): continue
        pdf_path = os.path.join(folder, file)
        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            page_text = page.get_text().strip()
            
            # 1. Text Document
            if page_text:
                docs.append(Document(
                    page_content=page_text,
                    metadata={"source": file, "page": page_num+1, "type": "text"}
                ))

            # 2. Image Documents
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_name = f"{file}_p{page_num+1}_img{img_index}.{base_image['ext']}"
                image_path = os.path.join(image_dir, image_name)
                
                with open(image_path, "wb") as f:
                    f.write(base_image["image"])

                
                
                docs.append(Document(
                    page_content=f"Visual info/Image from page {page_num+1}. Context: {page_text[:500]}",
                    metadata={
                        "source": file, 
                        "page": page_num+1, 
                        "type": "image", 
                        "image_path": image_path  
                    }
                ))
    return docs

def ingest_multimodal():
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
    VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore", "faiss_index_multimodal")
    
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    print("📄 Loading PDFs and extracting images...")
    all_docs = load_pdf_multimodal(DATA_DIR, IMAGE_DIR)

    # Split only text documents into chunks
    text_docs = [d for d in all_docs if d.metadata["type"] == "text"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(text_docs)

    # Now add image documents without splitting
    image_docs = [d for d in all_docs if d.metadata["type"] == "image"]
    chunks.extend(image_docs)

    print(f"✂️ Total Chunks: {len(chunks)}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)
    print("✅ Vector Store Taiyar Hai!")

if __name__ == "__main__":
    ingest_multimodal()