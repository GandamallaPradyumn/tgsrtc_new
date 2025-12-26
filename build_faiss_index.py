import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------
# Load embedding model
# -------------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")  # very strong sentence embedding model

# -------------------------------------------
# Load RAG documents
# -------------------------------------------
docs = []
with open("rag_documents.jsonl", "r") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Loaded {len(docs)} documents.")


# -------------------------------------------
# Generate embeddings
# -------------------------------------------
texts = [doc["text"] for doc in docs]  # embedding text field
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

print("Embeddings generated:", embeddings.shape)

# -------------------------------------------
# Create FAISS index (cosine similarity)
# -------------------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = inner product (cosine similarity since normalized)

index.add(embeddings)
print("FAISS index size:", index.ntotal)

# -------------------------------------------
# Save index + metadata
# -------------------------------------------
faiss.write_index(index, "/tmp/faiss_index.bin")

# Metadata stored separately for retrieval
metadata = {
    "docs": [
        {
            "doc_id": doc["doc_id"],
            "depot": doc["depot"],
            "date": doc["date"],
            "raw_fields": doc["raw_fields"],
            "calculated_fields": doc["calculated_fields"]
        }
        for doc in docs
    ]
}

with open("/tmp/metadata.json", "w") as f:
    json.dump(metadata, f)

print("FAISS index and metadata saved successfully.")
