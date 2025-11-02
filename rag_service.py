# rag_service.py
import threading
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from create_db import generate_data_store

load_dotenv()

LLM_REPO = "mistralai/Mistral-7B-Instruct-v0.1"
HUGGINGFACE_TOKEN = None

_lock = threading.Lock()
_llm = None

RQA_PROMPT = PromptTemplate(
    template = """Use the following pieces of context to answer the questions at the end.
Answer only from the context. If you don't know the answer, say you do not know.
{context}
Question: {question}
""",
    input_variables = ["context","question"]
)

def get_llm():
    global _llm
    with _lock:
        if _llm is not None:
            return _llm
        # Delay importing token to runtime via env
        from os import getenv
        token = getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN in env")
        _llm = HuggingFaceHub(repo_id=LLM_REPO,
                             huggingfacehub_api_token=token,
                             model_kwargs={"temperature":0.1, "max_length":512})
        return _llm

def query(question: str, k: int = 5, device: str = "cpu"):
    vectordb = generate_data_store(force_reindex=False, device=device)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff",
                                     retriever=retriever,
                                     chain_type_kwargs={"prompt": RQA_PROMPT},
                                     return_source_documents=True)
    res = qa({"query": question})
    return res
