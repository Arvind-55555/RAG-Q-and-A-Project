# api.py
import os
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from rag_service import query as rag_query
from create_db import generate_data_store

load_dotenv()
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="RAG QA Service")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    device: Optional[str] = os.getenv("EMBED_DEVICE", "cpu")

@app.post("/query")
def run_query(req: QueryRequest):
    if not req.question or len(req.question) < 2:
        raise HTTPException(status_code=400, detail="question too short")
    try:
        res = rag_query(req.question, k=req.k, device=req.device)
        return {
            "answer": res.get("result"),
            "sources": [
                {"metadata": d.metadata, "page_content": d.page_content}
                for d in res.get("source_documents", [])
            ]
        }
    except Exception as e:
        logging.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

class ReindexRequest(BaseModel):
    force: Optional[bool] = False
    device: Optional[str] = os.getenv("EMBED_DEVICE", "cpu")

def _check_admin(token: Optional[str]):
    if token is None or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid ADMIN_TOKEN")

@app.post("/admin/reindex")
def reindex(req: ReindexRequest, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    try:
        generate_data_store(force_reindex=req.force, device=req.device)
        return {"status":"ok", "message":"Reindex triggered"}
    except Exception as e:
        logging.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=str(e))
