from fastapi import FastAPI
from Backend.rag_chain import ask

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Medical RAG API is running!"}

@app.post("/query")
def query_rag(question: str):
    answer, sources = ask(question)
    return {
        "answer": answer,
        "sources": sources
    }
