import streamlit as st
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from llm_model import load_model


st.title("테스트")

uploaded_file = st.file_uploader("파일", type=("pdf"))

llm = load_model.load_llm()

if uploaded_file:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # PyMuPDFLoader로 문서 로드
    loader = PyMuPDFLoader(temp_path)
    docs = loader.load()

    # 전체 텍스트 추출
    full_text = "\n\n".join(doc.page_content for doc in docs)

    st.subheader("PDF에서 추출한 텍스트:")
    st.text_area("텍스트", full_text, height=300)  # 너무 길면 자름

    question = st.text_input("파일", placeholder="요약?",)

    if question:
        prompt = PromptTemplate(
            input_variables=["data"],
            template="{question}\n\n 다음은 PDF에서 추출한 내용입니다:\n{data}"
        )
        
        chain = prompt | llm | StrOutputParser()
        st.subheader("응답:")
        st.text(chain.invoke({"data": full_text, "question": question}))
    
        