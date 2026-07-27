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
    return {"message": "Ingest PDF function triggered."}
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [ingest_pdf])