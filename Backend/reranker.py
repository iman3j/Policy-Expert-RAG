import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu"
)

def rerank(query, docs, top_k=5):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker_model.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [d for d, _ in ranked[:top_k]]
