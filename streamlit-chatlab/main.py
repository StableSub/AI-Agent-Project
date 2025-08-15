import os, sys, asyncio, hashlib, tempfile
import streamlit as st

from llm_model import load_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
import pandas as pd

from RAG import rag_test
from EDA import run_eda

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Sub's Agent", page_icon="😀")
st.title("My_Little_LLM")
st.caption("랭체인 및 스트림릿 테스트")

with st.sidebar:
    "[깃허브](https://github.com/StableSub)"

LLM = load_model.load_llm()

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-u", os.path.abspath("MCP/server.py")],
)

# 업로드 된 파일의 바이트 데이터를 해시로 계산.
# 같은 파일이 여러 번 업로드 되어도 중복 인덱싱을 방지.
def file_md5(uploaded_file) -> str | None:
    if not uploaded_file:
        return None
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

# Streamlit의 세션 상태 초기화.
# 사용자와 LLM 간의 대화 내용, 업로드 파일 예시 및 retriever 객체 등을 초기값으로 설정.
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "파일 업로드 후 질문해 보세요!"}]
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "agent_tool_count" not in st.session_state:
        st.session_state.agent_tool_count = 0

init_state()

uploaded_file = st.file_uploader("파일 업로드", type=("pdf", "csv", "xlsx", "json",))
if uploaded_file:
    message = f"{uploaded_file.name}에 대해 질문 해보세요"
    st.session_state.messages = [{"role": "assistant", "content": message}]
new_hash = file_md5(uploaded_file)

# 업로드된 파일이 이전에 없던 파일이라면 조건문 진입.
if uploaded_file and new_hash != st.session_state.file_hash:
    st.info("새 파일 감지 → 인덱싱 중...")
    st.session_state.retriever = rag_test(uploaded_file, LLM) # RAG용 retriever 객체 생성.
    st.session_state.file_hash = new_hash
    suffix = "." + uploaded_file.name.split(".")[-1].lower() # 업로드된 문서의 확장자 추출.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp: # 업로드된 파일을 임시 경로로 저장.
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        st.session_state["uploaded_path"] = tmp_path
        st.success(f"파일 저장 완료: {tmp_path}")
    df = pd.read_csv(tmp_path)
    run_eda(df)

# 여태까지 저장된 메세지 출력.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 사용자 질문을 받아 MCP Agent를 실행하고, 적절한 툴을 사용해 응답 생성.
async def run(user_input: str):
    """매 입력마다 1회 호출: MCP 연결 → 툴 구성 → 에이전트 생성/호출."""
    async with stdio_client(SERVER_PARAMS) as (read, write): # MCP 서버에 연결.
        async with ClientSession(read, write) as session: # 툴 및 르폼프트 로딩을 위해 초기화 실행.
            await session.initialize()
            
            # MCP에 등록된 툴들을 LangChain Agent가 사용할 수 있또록 리스트로 로드
            mcp_tools = await load_mcp_tools(session)
            tools = list(mcp_tools)

            # 업로드된 문서가 있으면, 해당 문서에서 검색할 수 있는 retriever를 LangChain 툴로 등록.
            if st.session_state.retriever:
                retriever_tool = create_retriever_tool(
                    st.session_state.retriever,
                    # toll 이름 및 Agent가 어떤 질문에 이 툴을 사용할지를 결정.
                    name="doc_search",
                    description=(
                        "업로드된 문서에서 질문과 관련된 내용을 찾습니다. "
                        "요약/개요/핵심 정리 등도 이 도구로 필요한 근거를 수집한 후 작성하세요."
                    ),
                )
                tools.append(retriever_tool)
                if st.session_state.retriever:
                    st.success("retriever 생성 완료")
                else:
                    st.warning("retriever 생성 실패")

            # LLM이 자유롭게 툴을 선택.
            agent = create_react_agent(LLM, tools)

            # default_prompt를 기반으로 LLM이 사용할 초기 메세지 생성.
            prompts = await load_mcp_prompt(session, "default_prompt", arguments={"message": user_input})
            
            # LLM Agent를 실행하여 응답 생성.
            response = await agent.ainvoke({"messages": prompts})
            answer = response["messages"][-1].content
            return answer
        
if user_input := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content" : user_input})
    with st.chat_message("assistant"):
        thinking_msg = st.empty()
        thinking_msg.markdown("💭 생각 중입니다...")
        try:
            answer = asyncio.run(run(user_input))
        except Exception as e:
            answer = f"에러가 발생했습니다: {e}"
        thinking_msg.empty()
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})