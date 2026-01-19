import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ask_multimodal(query, k=5):
    current_file = os.path.abspath(__file__)
    backend_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(backend_dir)
    
   
    VECTOR_DIR = os.path.join(base_dir, "vectorstore", "faiss_index_multimodal")
    if not os.path.exists(VECTOR_DIR):
        print(f"DEBUG: Vector directory not found at {VECTOR_DIR}")
        VECTOR_DIR = os.path.join(base_dir, "Vectorstore", "faiss_index_multimodal")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        db = FAISS.load_local(
            VECTOR_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        return [f"Error loading vector store: {str(e)}"], []

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
                img_path = img_path.replace("\\", "/")
                if "Data/" in img_path:
                    relative_path = img_path.split("Data/")[-1]
                    img_path = os.path.join(base_dir, "Data", relative_path)
                
                images.append(img_path)

    return texts, list(set(images))
