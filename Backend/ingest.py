import os
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import *
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore", "faiss_index_multimodal")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Load PDFs and extract text + images
def load_pdf_multimodal(folder):
    docs = []
    for file in os.listdir(folder):
        if not file.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder, file)
        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text().strip()
            
            # Add text as a document
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file, "page": page_num+1, "type": "text"}
                ))

            # Extract images from the same page
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"{file}_page{page_num+1}_img{img_index}.{image_ext}"
                image_path = os.path.join(IMAGE_DIR, image_name)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Add image as a document
                docs.append(Document(
                    page_content=image_path,
                    metadata={"source": file, "page": page_num+1, "type": "image"}
                ))

    return docs

# Chunk text only
def ingest_multimodal():
    print(f"📄 Loading PDFs from: {DATA_DIR}")
    docs = load_pdf_multimodal(DATA_DIR)

    # Split text docs into chunks
    text_docs = [d for d in docs if d.metadata["type"]=="text"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(text_docs)

    # Append image docs to chunks (no splitting)
    image_docs = [d for d in docs if d.metadata["type"]=="image"]
    chunks.extend(image_docs)

    print(f"✂️ Total chunks (text + images): {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)

    print(f"✅ Multimodal FAISS index saved to {VECTOR_DIR}")


if __name__ == "__main__":
    ingest_multimodal()
