import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# CSV → Retriever 생성 (진행률 표시 포함)
def build_retriever_from_csv(uploaded_file, k: int = 3):
    df = pd.read_csv(uploaded_file)

    # 진행바 생성
    progress_bar = st.progress(0)
    status_text = st.empty()

    docs = []
    total = len(df)

    for i, row in df.iterrows():
        content = row.to_json(force_ascii=False)
        docs.append(Document(page_content=content, metadata={"row": i}))

        # 진행률 업데이트 (5% 단위로 갱신하도록 최적화)
        if i % max(1, total // 1000) == 0 or i == total - 1:
            pct = int((i + 1) / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"인덱싱 진행 중... {pct}% ({i+1}/{total})")

    status_text.text("벡터 스토어 생성 및 retriever 생성 중...")

    # 벡터스토어 & retriever 생성
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embed)

    progress_bar.empty()
    status_text.empty()

    return vs.as_retriever(search_kwargs={"k": k})