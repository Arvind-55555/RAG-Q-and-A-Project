# app_streamlit.py
import streamlit as st
import os, requests

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="RAG QA Demo")
st.title("RAG QA — Streamlit Demo")

question = st.text_area("Ask a question about the dataset", height=120)
k = st.slider("Number of retrieved documents (k)", 1, 10, 5)

if st.button("Run Query") and question.strip():
    with st.spinner("Querying..."):
        resp = requests.post(API_URL, json={"question":question, "k":k})
    if resp.status_code == 200:
        data = resp.json()
        st.subheader("Answer")
        st.write(data.get("answer"))
        st.subheader("Sources")
        for i, s in enumerate(data.get("sources", []), 1):
            st.markdown(f"**Source {i}**")
            st.json(s.get("metadata", {}))
            st.write(s.get("page_content")[:1000] + ("..." if len(s.get('page_content',''))>1000 else ""))
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")
