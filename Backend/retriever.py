from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from .config import *
import pickle


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

documents = db.docstore._dict.values()
corpus = [doc.page_content for doc in documents]
bm25 = BM25Okapi([text.split() for text in corpus])

def hybrid_search(query):
    vector_docs = db.similarity_search(query, k=TOP_K)
    bm25_scores = bm25.get_scores(query.split())

    bm25_docs = sorted(
        zip(documents, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_K]

    bm25_docs = [doc for doc, _ in bm25_docs]
    return list(set(vector_docs + bm25_docs))