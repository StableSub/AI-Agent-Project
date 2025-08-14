import streamlit as st
import asyncio
from llm_model import load_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent


st.set_page_config(page_title='ChatGPT', page_icon='😀')
st.title('My_Little_LLM')
st.caption("랭체인 및 스트림릿 테스트")

server_params = StdioServerParameters(
    command="python",
    args=["MCP/server.py"],
)

with st.sidebar:
    "[깃허브](https://github.com/StableSub)"

llm = load_model.load_llm()
    
async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            
            # 채팅 기록 초기화
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "뭐가 궁금해?"}] # session_state는 서버 RAM에 데이터를 저정하여 딕셔너리 형태로 데이터 저장

            # 렌더링 시 기존 채팅 기록을 보여줌
            for message in st.session_state.messages:
                with st.chat_message(message["role"]): # with은 컨텍스트 관리자 문법으로 리소스를 열고 자동으로 닫게 함
                    st.markdown(message["content"])

            # 채팅을 입력할 수 있는 곳을 만들고, 유저의 채팅을 메모리에 저장
            if user_input := st.chat_input("Say something"):
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "content" : user_input})
                with st.chat_message("assistant"):
                    prompts = await load_mcp_prompt(
                        session, "default_prompt", arguments={"message": user_input}
                    )
                    response = await agent.ainvoke({"messages": prompts})
                    st.markdown(response["messages"][-1].content)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
asyncio.run(run())