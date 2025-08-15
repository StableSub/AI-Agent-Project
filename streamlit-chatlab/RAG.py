import json, tempfile
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def _docs_from_tabular(path: str, ext: str) -> list[Document]:
    """CSV/XLSX/JSON을 텍스트로 변환해 Document 리스트로 만듭니다."""
    docs: list[Document] = []

    if ext == "csv":
        df = pd.read_csv(path)
        text = df.to_markdown(index=False)
        docs.append(Document(page_content=text, metadata={"source": path, "type": "csv"}))

    elif ext in ("xlsx", "xls"):
        # 여러 시트 지원
        sheets = pd.read_excel(path, sheet_name=None)  # dict[sheet_name] = df
        for sheet_name, df in sheets.items():
            text = f"# Sheet: {sheet_name}\n" + df.to_markdown(index=False)
            docs.append(Document(page_content=text, metadata={"source": path, "sheet": sheet_name, "type": "excel"}))

    elif ext in ("json", "jsonl", "ndjson"):
        # jsonl이면 lines=True, json이면 normal load
        if ext in ("jsonl", "ndjson"):
            df = pd.read_json(path, lines=True)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 배열이면 표로, dict면 정규화 후 표로
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
        text = df.to_markdown(index=False)
        docs.append(Document(page_content=text, metadata={"source": path, "type": "json"}))

    else:
        raise ValueError(f"지원하지 않는 탭형식 확장자: {ext}")

    return docs

def rag_test(uploaded_file, llm=None):
    """업로드 파일로 인덱스 생성 → retriever 반환"""
    if not uploaded_file:
        return None

    # 1) 임시파일로 저장 (확장자 유지)
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    # 2) 문서 로드 (확장자별 분기)
    if file_ext == "pdf":
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
    elif file_ext in ("csv", "xlsx", "xls", "json", "jsonl", "ndjson"):
        docs = _docs_from_tabular(temp_path, file_ext)
    else:
        raise ValueError(f"지원하지 않는 확장자: .{file_ext}")

    # 3) 청크
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    # 4) 임베딩 & 벡터스토어
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    # 저장해두면 다음에 재사용 가능
    vectorstore.save_local("vector_db")

    # 5) retriever 반환
    return vectorstore.as_retriever(search_kwargs={"k": 3})