import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ask_multimodal(query, k=5):
    BASE_DIR = os.getcwd()
    VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore", "faiss_index_multimodal")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

    # Similarity Search
    results = db.similarity_search(query, k=k)

    texts = []
    images = []

    for doc in results:
        if doc.metadata.get("type") == "text":
            texts.append(doc.page_content)
        elif doc.metadata.get("type") == "image":
            img_path = doc.metadata.get("image_path")
            if img_path:
                images.append(img_path)

    return texts, list(set(images))
