"""API and Inngest workflows for the Project Portfolio Agent."""

import logging
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import inngest
import inngest.fast_api
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

from data_loader import list_repo_source_files, load_and_chunk_github_repo
from embeddings import embed_chunks
from qdrant_store import answer_question, upsert_chunks


class IngestRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repo_name: str = Field(min_length=1, max_length=100)
    file_paths: list[str] = Field(min_length=1, max_length=200)
    branch: str = Field(default="main", min_length=1, max_length=255)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=20)
    repo_filter: str | None = Field(default=None, max_length=100)


class RepositoryFilesRequest(BaseModel):
    repo_url: str = Field(min_length=3, max_length=500)
    branch: str = Field(default="main", min_length=1, max_length=255)


def parse_github_repository(repo_url: str) -> tuple[str, str]:
    """Return GitHub owner and repository name from a repository URL or owner/repo."""
    value = repo_url.strip().rstrip("/")
    if value.startswith("git@github.com:"):
        value = value.removeprefix("git@github.com:")
    elif value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise ValueError("The repository URL must point to github.com.")
        value = parsed.path.strip("/")

    parts = value.removesuffix(".git").split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError("Use a GitHub repository URL such as https://github.com/owner/repository.")
    return parts[0], parts[1]


def ingest_repository(request: IngestRequest) -> dict:
    """Load changed files, embed them, and store them in Qdrant."""
    chunks = load_and_chunk_github_repo(
        request.owner, request.repo_name, request.file_paths, request.branch
    )
    upsert_chunks(embed_chunks(chunks))
    return {"chunks_ingested": len(chunks)}


inngest_client = inngest.Inngest(
    app_id="project_portfolio_agent",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="Portfolio: Ingest Repository",
    trigger=inngest.TriggerEvent(event="portfolio/ingest_repository"),
)
async def ingest_repo(ctx: inngest.Context):
    data = ctx.event.data
    request = IngestRequest(
        owner=data["owner"],
        repo_name=data["repo_name"],
        file_paths=data["file_paths"],
        branch=data.get("branch", "main"),
    )
    return await ctx.step.run("ingest-repository", lambda: ingest_repository(request))


@inngest_client.create_function(
    fn_id="Portfolio: Query Repository",
    trigger=inngest.TriggerEvent(event="portfolio/query_repository"),
)
async def query_repository(ctx: inngest.Context):
    data = ctx.event.data
    request = QueryRequest(
        question=data["question"],
        top_k=data.get("top_k", 5),
        repo_filter=data.get("repo_filter"),
    )
    return await ctx.step.run(
        "answer-question",
        lambda: answer_question(request.question, request.top_k, request.repo_filter),
    )


app = FastAPI(title="Project Portfolio Agent", version="0.1.0")


@app.get("/", tags=["portfolio-agent"])
def index() -> dict:
    return {
        "name": "Project Portfolio Agent",
        "docs": "/docs",
        "endpoints": {
            "repository_files": "POST /repository/files",
            "ingest": "POST /ingest",
            "query": "POST /query",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["portfolio-agent"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/repository/files", tags=["portfolio-agent"])
def repository_files(request: RepositoryFilesRequest) -> dict:
    try:
        owner, repo_name = parse_github_repository(request.repo_url)
        files = list_repo_source_files(owner, repo_name, request.branch)
        return {
            "owner": owner,
            "repo_name": repo_name,
            "branch": request.branch,
            "files": files,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load repository files: {exc}") from exc


@app.post("/ingest", tags=["portfolio-agent"])
def ingest(request: IngestRequest) -> dict:
    try:
        return ingest_repository(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Repository ingestion failed: {exc}") from exc


@app.post("/query", tags=["portfolio-agent"])
def query(request: QueryRequest) -> dict:
    try:
        return answer_question(request.question, request.top_k, request.repo_filter)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Repository query failed: {exc}") from exc


inngest.fast_api.serve(app, inngest_client, [ingest_repo, query_repository])
