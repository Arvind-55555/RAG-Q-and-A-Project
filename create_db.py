# create_db.py
import os
import json
import time
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_PATH", "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
INDEX_TRACK_FILE = os.path.join(CHROMA_DIR, "indexed_files.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
SUPPORTED_FILE_EXT = [".pdf", ".txt", ".md"]

def _ensure_dirs():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def _list_data_files():
    p = Path(DATA_DIR)
    files = []
    if not p.exists():
        return files
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_FILE_EXT:
            files.append(f)
    return sorted(files)

def _file_mtime(path: Path):
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0

def _load_file(path: Path):
    if path.suffix.lower() == ".pdf":
        loader = UnstructuredPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        if not d.metadata:
            d.metadata = {}
        d.metadata["source_file"] = str(path)
        d.metadata["source_mtime"] = _file_mtime(path)
    return docs

def _split_documents(docs: List[Document]):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    new_docs = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, chunk in enumerate(chunks):
            metadata = dict(d.metadata) if d.metadata else {}
            metadata["_chunk_id"] = f"{os.path.basename(metadata.get('source_file','unknown'))}::{i}"
            new_docs.append(Document(page_content=chunk, metadata=metadata))
    return new_docs

def _load_indexed_files():
    if not os.path.exists(INDEX_TRACK_FILE):
        return {}
    try:
        with open(INDEX_TRACK_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _save_indexed_files(d):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(INDEX_TRACK_FILE, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)

def _get_embeddings(device: str = EMBED_DEVICE):
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                 model_kwargs={"device": device},
                                 encode_kwargs={"normalize_embeddings": True})

def generate_data_store(force_reindex: bool = False, device: str = EMBED_DEVICE):
    _ensure_dirs()
    embeddings = _get_embeddings(device=device)

    if force_reindex:
        if os.path.exists(CHROMA_DIR):
            print("Force reindex: clearing existing chroma directory...")
            for root, dirs, files in os.walk(CHROMA_DIR, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass

    vectordb = None
    if os.path.exists(CHROMA_DIR):
        try:
            vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception:
            vectordb = None

    files = _list_data_files()
    current_files_meta = { str(f): _file_mtime(f) for f in files }

    indexed_files = _load_indexed_files()

    if force_reindex:
        to_process = files
    else:
        to_process = []
        for f in files:
            path_str = str(f)
            mtime = current_files_meta.get(path_str, 0.0)
            prev = indexed_files.get(path_str)
            if prev is None or mtime > prev + 1e-6:
                to_process.append(f)

    if not to_process:
        if vectordb is None:
            print("No files to index and no existing Chroma. Creating empty store.")
            vectordb = Chroma.from_documents([], embedding=embeddings, persist_directory=CHROMA_DIR)
            return vectordb
        print("No new/modified files to index. Loading existing Chroma.")
        return vectordb

    all_new_chunks = []
    for f in to_process:
        print(f"Loading {f}")
        docs = _load_file(f)
        chunks = _split_documents(docs)
        all_new_chunks.extend(chunks)

    if vectordb is None:
        print(f"Creating new Chroma with {len(all_new_chunks)} chunks...")
        vectordb = Chroma.from_documents(all_new_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    else:
        print(f"Adding {len(all_new_chunks)} new chunks to existing Chroma...")
        vectordb.add_documents(all_new_chunks)

    vectordb.persist()

    for f in to_process:
        indexed_files[str(f)] = current_files_meta.get(str(f), _file_mtime(f))

    _save_indexed_files(indexed_files)
    print(f"Indexing complete ({len(to_process)} files).")
    return vectordb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force full reindex")
    parser.add_argument("--device", type=str, default=EMBED_DEVICE, help="Device for embeddings (cpu|cuda)")
    args = parser.parse_args()
    t0 = time.time()
    generate_data_store(force_reindex=args.force, device=args.device)
    print("Done in %.2f sec" % (time.time() - t0))
