from pathlib import Path
import base64
import os
from qdrant_store import get_existing_sha, delete_file_chunks
from github import Github
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from dotenv import load_dotenv

load_dotenv()

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".html": "html",
    ".css": "css",
}

SUPPORTED_SOURCE_EXTENSIONS = {
    ".css", ".go", ".html", ".java", ".js", ".json", ".jsx", ".md", ".mdx",
    ".py", ".rb", ".rs", ".sql", ".toml", ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
}


def get_code_splitter(file_path: str):
    ext = Path(file_path).suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if not lang:
        return None
    try:
        return CodeSplitter(language=lang, chunk_lines=40, chunk_lines_overlap=5, max_chars=1500)
    except Exception:
        return None


def load_and_chunk_pdf(path: str | Path):
    docs = PDFReader().load_data(file=Path(path))
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


def list_repo_source_files(owner: str, repo_name: str, branch: str = "main") -> list[str]:
    """List text-based source files that can safely be ingested from a repository."""
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(f"{owner}/{repo_name}")
    tree = repo.get_git_tree(sha=branch, recursive=True)
    return sorted(
        item.path
        for item in tree.tree
        if getattr(item, "type", None) == "blob"
        and Path(item.path).suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
    )


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
            content = base64.b64decode(blob.content).decode("utf-8", errors="replace")
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


def load_and_chunk_code(text: str, file_path: str, repo: str, sha: str):
    code_splitter = get_code_splitter(file_path)
    if code_splitter is None:
        return load_and_chunk_doc(text, file_path, repo, sha)
    try:
        chunks = code_splitter.split_text(text)
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
    except Exception:
        # Fall back gracefully to standard sentence splitter if language parser fails
        return load_and_chunk_doc(text, file_path, repo, sha)


def load_and_chunk_github_file(file_path: str, text: str, repo: str, sha: str):
    if file_path.lower().endswith((".md", ".mdx", ".txt", ".json", ".yaml", ".yml", ".toml", ".xml")):
        return load_and_chunk_doc(text, file_path, repo, sha)
    return load_and_chunk_code(text, file_path, repo, sha)


def load_and_chunk_github_repo(owner: str, repo_name: str, file_paths: list[str], branch: str = "main"):
    """Load and chunk selected files from a GitHub repo."""
    files = fetch_repo_files(owner, repo_name, file_paths, branch)
    all_chunks = []
    for path, data in files.items():
        existing_sha = get_existing_sha(repo_name, path)
        if existing_sha == data["sha"]:
            print(f"Skipping {path} (SHA unchanged)")
            continue
        if existing_sha is not None:
            delete_file_chunks(repo_name, path)
        chunks = load_and_chunk_github_file(path, data["content"], repo_name, data["sha"])
        all_chunks.extend(chunks)
    return all_chunks
