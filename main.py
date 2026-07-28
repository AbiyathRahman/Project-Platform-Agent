from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental import ai
import logging
from github import Github

load_dotenv()

g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo("AbiyathRahman/Portfolio")
tree = repo.get_git_tree(sha="main", recursive=True)
matched = False
for item in tree.tree:
    if item.path in ["src/App.jsx"]:
        matched = True
        blob = repo.get_git_blob(item.sha)
        content = blob.content
        import base64
        text = base64.b64decode(content).decode("utf-8")
        print(f"--- {item.path} ({len(text)} chars) ---")
        print(text[:300])  # just a preview, not the whole file

if not matched:
    print("WARNING: no files matched — check your path list against actual tree paths")

inngest_client = inngest.Inngest(
    app_id = "project_portfolio_agent",
    logger=logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id = "RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def ingest_pdf(ctx: inngest.Context):
    return {"message": "Ingest PDF function is not implemented yet."}
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [ingest_pdf])