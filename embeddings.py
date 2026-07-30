"""OpenAI embedding helpers shared by ingestion and retrieval."""

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
EMBED_BATCH_SIZE = 100

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text in bounded batches while preserving input order."""
    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts[start : start + EMBED_BATCH_SIZE],
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach an embedding to every chunk that contains text."""
    if not chunks:
        return chunks

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks
