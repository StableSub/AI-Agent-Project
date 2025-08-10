import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from llm_model import load_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


load_dotenv()

llm = load_model.load_llm()

st.title("테스트")

uploaded_file = st.file_uploader("파일", type=("pdf"))

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
        
    loader = PyMuPDFLoader(temp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local("vector_db")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """다음은 참고할 문서 내용입니다:
    {context}

    사용자의 질문: {question}

    문서를 바탕으로 정확하게 답변해 주세요."""
    prompt = PromptTemplate.from_template(template)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "이 문서에 대해서 어떤게 궁금해?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content" : user_input})
        with st.chat_message("assistant"):
            chain = load_model.load_chain(llm)
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
            )
            response = rag_chain.invoke(user_input)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})