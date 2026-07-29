from fastapi import FastAPI
import inngest
import inngest.fast_api
import logging
from dotenv import load_dotenv
from data_loader import load_and_chunk_github_repo, embed_chunks
from qdrant_store import upsert_chunks, answer_question

load_dotenv()

# g = Github(os.getenv("GITHUB_TOKEN"))
# repo = g.get_repo("AbiyathRahman/Portfolio")
# tree = repo.get_git_tree(sha="main", recursive=True)
# matched = False
# for item in tree.tree:
#     if item.path in ["src/App.jsx"]:
#         matched = True
#         blob = repo.get_git_blob(item.sha)
#         content = blob.content
#         import base64
#         text = base64.b64decode(content).decode("utf-8")
#         print(f"--- {item.path} ({len(text)} chars) ---")
#         print(text[:300])  # just a preview, not the whole file

# if not matched:
#     print("WARNING: no files matched — check your path list against actual tree paths")

inngest_client = inngest.Inngest(
    app_id = "project_portfolio_agent",
    logger=logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id = "RAG: Ingest Repository",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def ingest_repo(ctx: inngest.Context):
    data = ctx.event.data
    owner = data["owner"]
    repo_name = data["repo_name"]
    file_paths = data["file_paths"]
    branch = data.get("branch", "main")

    chunks = await ctx.step.run(
        "load-and-chunk",
        lambda: load_and_chunk_github_repo(owner, repo_name, file_paths, branch),
    )
    chunks = await ctx.step.run("embed", lambda: embed_chunks(chunks))
    await ctx.step.run("upsert-qdrant", lambda: upsert_chunks(chunks))

    return {"chunks_ingested": len(chunks)}

@inngest_client.create_function(
    fn_id = "RAG: Query Repository",
    trigger=inngest.TriggerEvent(event="rag/query_pdf"),
)
async def rag_query_repo(ctx: inngest.Context):
    data = ctx.event.data
    result = await ctx.step.run(
        "answer-question",
        lambda: answer_question(
            data["question"],
            top_k=data.get("top_k", 5),
            repo_filter=data.get("repo_filter"),
        ),
    )
    return result
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [ingest_repo, rag_query_repo])
