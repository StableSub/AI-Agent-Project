import os, sys, asyncio, hashlib, tempfile, time
import streamlit as st

from llm_model import load_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
import pandas as pd

from rag_processsing import *
from data_processing import *

target_dir1 = "/Users/anjeongseob/Desktop/Storage/Python/AI-Agent/streamlit-chatlab/data/meta"
target_dir2 = "/Users/anjeongseob/Desktop/Storage/Python/AI-Agent/streamlit-chatlab/data/uploads"

for target_dir in [target_dir1, target_dir2]:
    for file in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Sub's Agent", page_icon="😀")
st.title("My Little AI Agent")
st.caption("AI Agent Prototype")

LLM = load_model.load_llm()

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-u", os.path.abspath("MCP/server.py")],
)

# Streamlit의 세션 상태 초기화.
# 사용자와 LLM 간의 대화 내용, 업로드 파일 예시 및 retriever 객체 등을 초기값으로 설정.
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "질문해 보세요!"}]
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "agent_tool_count" not in st.session_state:
        st.session_state.agent_tool_count = 0
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

init_state()

with st.sidebar:
    st.markdown("### 옵션")
    sample_rows = st.number_input("샘플 로딩 행 수", min_value=10, max_value=2000, value = 100, step=10)

uploaded_file = st.file_uploader("파일을 업로드하세요", type=list({"csv", "tsv", "txt"}))

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        dsid, raw_path, ext = save_upload_to_disk(uploaded_file)
                
        st.session_state["file_hash"] = file_hash
        st.session_state["dsid"] = dsid
        st.session_state["raw_path"] = str(raw_path)
        st.session_state["ext"] = ext
        
        st.success(f"저장 완료 - dataset_id: {dsid} 경로: {raw_path}")
        
        try:
            sniff_info = sniff_file(
                raw_path=raw_path,
                ext=ext
            )
        except Exception as e:
            st.error(f"스니핑 오류: {e}")
            st.stop()
            
        with st.spinner("샘플 로딩 중..."):
            try:
                df, sample_info = sample_load(raw_path=raw_path, sniff_info=sniff_info, sample_rows=int(sample_rows))
            except Exception as e:
                write_meta(dsid, {"sniff": sniff_info, "error": str(e)})
                st.error(f"로드 오류: {e}")
                st.stop()
        meta = {
            "sniff": sniff_info,
            "shape_sample": list(df.shape),
            "shape_total": sample_info.get("shape_total"),
            "columns": list(df.columns),
            "ext": ext,
            "raw_path": str(raw_path),
        }
        write_meta(dsid, meta)
        
        st.session_state["meta"] = meta
        st.session_state["preview_df"] = df.head(20)
        st.session_state["dtype_df"] = pd.DataFrame({
            "column": df.columns,
            "null_count": df.isnull().sum().values,
            "null_ratio": (df.isnull().mean() * 100).round(2).values,
            "dtype": df.dtypes.astype(str).values,
        })
            
    meta = st.session_state["meta"]
    df_preview = st.session_state["preview_df"]
    dtype_df = st.session_state["dtype_df"]
    
    c1, c2, c3, c4 = st.columns(4)
    n_rows, n_cols = meta["shape_total"]
    c2.metric("행 수(추정)", f"{n_rows:,}")
    c3.metric("열 수", f"{n_cols:,}")
    c4.metric("파일 유형", meta["sniff"]["filetype"])
    
    with st.expander("스니핑 결과 보기", expanded=False):
        st.json(meta["sniff"])
    
    st.subheader("미리보기 (head 20)")
    st.dataframe(df_preview, use_container_width=True)
        
    st.subheader("컬럽/타입 요약")
    st.dataframe(dtype_df, use_container_width=True)
    
    with st.expander("기본 통계 (describe, numeric)", expanded=False):
        num_df = dtype_df.select_dtypes(include="number")
        if not num_df.empty:
            st.dataframe(num_df.describe().T, use_container_width=True)
        else:
            st.info("수치형 컬럼이 없습니다.")
            
    dsid = st.session_state["dsid"]
    st.caption(f"메타 저장 위치: '{META_DIR / f'{dsid}.json'}'")
    
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        retriever_info = st.info("Retriever 생성 중...")
        st.session_state.retriever = build_retriever_from_csv(uploaded_file)
        retriever_info.empty()
        
else:
    st.info("좌측 또는 위의 업로더로 CSV/Excel/Parquet 파일을 올리세요. 업로드하면 자동으로 미리보기를 생성합니다.")

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
                if not st.session_state.retriever:
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
        try:
            answer = asyncio.run(run(user_input))
        except Exception as e:
            answer = f"에러가 발생했습니다: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})