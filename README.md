# RAG QA Project 

## Overview
This repo contains a production-ready RAG (Retrieval-Augmented Generation) example:
- Incremental document indexing (`create_db.py`)
- FastAPI backend with `/query` and secured `/admin/reindex` (`api.py`)
- Streamlit demo (`app_streamlit.py`)
- React frontend scaffold (`frontend/`)
- Dockerfiles for GPU backend and frontend; `docker-compose.yml`
- Kubernetes manifests and a minimal Helm chart scaffold

## Quickstart (Local, GPU-enabled)
1. Create a `.env` in the repo root with:
```
HUGGINGFACEHUB_API_TOKEN=hf_xxx
ADMIN_TOKEN=your_admin_token
LLM_REPO=mistralai/Mistral-7B-Instruct-v0.1
EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
EMBED_DEVICE=cuda
```
2. Place your documents under `./data` (pdf, txt, md).
3. Build the index:
```bash
python create_db.py --device cuda
```
4. Run API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
5. Run Streamlit demo:
```bash
streamlit run app_streamlit.py
```
6. Or build & run with Docker Compose:
```bash
docker compose up --build
```

## Notes
- Keep your `.env` secret; do not commit tokens.
- For production, push images to your registry and update k8s manifests / Helm values.
