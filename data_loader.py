from openai import OpenAI
from pathlib import Path
import base64
import os

from github import Github
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"  # or "text-embedding-3-small"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size = 1000, chunk_overlap = 200)

EXT_TO_LANG = {
     ".jsx": "javascript",
    ".js": "javascript",
    ".tsx": "typescript",
    ".ts": "typescript",
    ".py": "python",
}

def get_code_splitter(file_path: str) -> CodeSplitter:
    ext = Path(file_path).suffix
    lang = EXT_TO_LANG.get(ext, "text")
    if lang is None:
        return None # type: ignore
    return CodeSplitter(language=lang, chunk_lines=40, chunk_lines_overlap=5, max_chars=1500)

def load_and_chunk_pdf(path: str | Path):
    docs =PDFReader().load_data(file=Path(path))
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        for chunk_text in splitter.split_text(t):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_type": "pdf",
                    "source": str(path),
                    
                }
            })
    return chunks

# Github loading
def fetch_repo_files(
    owner: str,
    repo_name: str,
    file_paths: list[str],
    branch: str = "main",
):
    """Fetch raw text + SHA for selected file paths from a GitHub repo."""
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(f"{owner}/{repo_name}")
    tree = repo.get_git_tree(sha=branch, recursive=True)

    path_set = set(file_paths)
    found = {}

    for item in tree.tree:
        if item.path in path_set:
            blob = repo.get_git_blob(item.sha)
            content = base64.b64decode(blob.content).decode("utf-8")
            found[item.path] = {
                "content": content,
                "sha": item.sha,
            }

    missing = path_set - found.keys()
    if missing:
        print(f"WARNING: these paths were not found in {repo_name}: {missing}")

    return found


def load_and_chunk_doc(text: str, file_path: str, repo: str, sha: str):
    chunks = splitter.split_text(text)
    return [
        {
            "text": chunk,
            "metadata": {
                "repo": repo,
                "file_path": file_path,
                "sha": sha,
                "source_type": "doc",
            },
        }
        for chunk in chunks
    ]

def load_and_chunk_code(text:str, file_path:str, repo:str, sha:str):
    splitter = get_code_splitter(file_path)
    if splitter is None:
        return load_and_chunk_doc(text, file_path, repo, sha)
    chunks = splitter.split_text(text)
    return [
        {
            "text": chunk,
            "metadata": {
                "repo": repo,
                "file_path": file_path,
                "sha": sha,
                "source_type": "code",
            },
        }
        for chunk in chunks
    ]
    
def load_and_chunk_github_file(file_path: str, text: str, repo: str, sha: str):
    if file_path.endswith((".md", ".mdx", ".txt")):
        return load_and_chunk_doc(text, file_path, repo, sha)
    return load_and_chunk_code(text, file_path, repo, sha)

def load_and_chunk_github_repo(owner: str, repo_name:str, file_paths: list[str], branch: str = "main"):
    """Load and chunk selected files from a GitHub repo."""
    files = fetch_repo_files(owner, repo_name, file_paths, branch)
    all_chunks = []
    for path, data in files.items():
        chunks = load_and_chunk_github_file(path, data["text"], repo_name, data["sha"])
        all_chunks.extend(chunks)
    return all_chunks
    
# Embedding
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Takes chunk dicts (with 'text'), returns them with 'embedding' attached."""
    vectors = embed_texts([c["text"] for c in chunks])
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks
    
    