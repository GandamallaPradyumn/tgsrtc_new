import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model, FAISS index, and metadata
model = SentenceTransformer("all-mpnet-base-v2")

index = faiss.read_index("faiss_index.bin")

with open("metadata.json", "r") as f:
    metadata = json.load(f)["docs"]


# -------------------------------------------
# Search Function
# -------------------------------------------
def search_faiss(query, top_k=5, depot=None, date=None):
    # Encode user query
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Perform FAISS search
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = metadata[idx]

        # Optional filtering
        if depot and doc["depot"].lower() != depot.lower():
            continue
        if date and doc["date"] != date:
            continue

        results.append({
            "score": float(score),
            "doc_id": doc["doc_id"],
            "depot": doc["depot"],
            "date": doc["date"],
            "raw_fields": doc["raw_fields"],
            "calculated_fields": doc["calculated_fields"]
        })

    return results
