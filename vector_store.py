import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer(MODEL_NAME)


def build_vector_store(chunks: list[str]):
    """
    Embed all chunks and build a FAISS index.
    Returns the FAISS index.
    """
    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
    index.add(embeddings)

    return index


def retrieve_relevant_chunks(
    query: str,
    index,
    chunks: list[str],
    k: int = 4
) -> list[str]:
    """
    Find the top-k most relevant chunks for a query.
    Returns list of chunk strings.
    """
    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    relevant = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            relevant.append(chunks[idx])

    return relevant
