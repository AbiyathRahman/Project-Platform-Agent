import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
from openai import OpenAI
from data_loader import embed_texts
from dotenv import load_dotenv

load_dotenv()
ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-4o-mini" 
COLLECTION_NAME = "portfolio_agent"
EMBED_DIM = 3072

SYSTEM_PROMPT = """You are an assistant that answers questions about Abiyath's software projects, \
using only the provided context chunks pulled from his GitHub repos.

Rules:
- Answer only using the given context. If the context doesn't contain enough information to answer, say so explicitly — do not guess or invent details.
- When relevant, mention which file/repo the information came from.
- Be concise and technical; the audience is a software engineer or interviewer.
"""

client = QdrantClient(url="http://localhost:6333", timeout=30)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION_NAME}' with dimension {EMBED_DIM}.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        
def upsert_chunks(chunks: list[dict]):
    ensure_collection()
    if not chunks:
        return

    points = []
    for chunk in chunks:
        payload = {
            "text": chunk["text"],
            **chunk["metadata"],
        }
        point_key = ":".join(
            [
                str(payload.get("repo", "")),
                str(payload.get("file_path", "")),
                str(payload.get("sha", "")),
                payload["text"],
            ]
        )
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key))
        points.append(
            PointStruct(id=point_id, vector=chunk["embedding"], payload=payload)
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} chunks into collection '{COLLECTION_NAME}'.")

def search(query_vector: list[float], top_k: int = 5, repo_filter: str | None = None):
    """Return the most similar chunks, optionally scoped to one repository."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    query_filter = None
    if repo_filter:
        query_filter = Filter(
            must=[FieldCondition(key="repo", match=MatchValue(value=repo_filter))]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
    return response.points

def build_context_block(results) -> str:
    """Formats retrieved Qdrant results into a labeled context string for the prompt."""
    blocks = []
    for r in results:
        payload = r.payload
        header = f"[{payload.get('repo')}/{payload.get('file_path')}]"
        blocks.append(f"{header}\n{payload.get('text')}")
    return "\n\n---\n\n".join(blocks)

def answer_question(query: str, top_k: int = 5, repo_filter: str | None = None) -> dict:
    # 1. embed the query
    query_vec = embed_texts([query])[0]

    # 2. retrieve
    results = search(query_vec, top_k=top_k, repo_filter=repo_filter)

    if not results:
        return {
            "answer": "I couldn't find any relevant repository context to answer that question.",
            "sources": [],
            "usage": None,
        }

    # 3. augment prompt
    context = build_context_block(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    # 4. generate
    response = ai_client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=600,
        temperature=0.2,  # low — we want faithful, grounded answers, not creative ones
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"repo": r.payload.get("repo"), "file_path": r.payload.get("file_path"), "score": r.score}
            for r in results
        ],
        "usage": response.usage.model_dump() if response.usage else None,
    }
