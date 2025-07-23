from langchain import hub
from langchain.prompts import ChatPromptTemplate

# 프롬프트 가져오기
prompt = hub.pull("rlm/rag-prompt")

print(prompt)

prompt = ChatPromptTemplate.from_template(
    "주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\nCONTEXT: {context}\n\nSUMMARY:"
)

#프롬프트 올리기
hub.push("teddynote/simple-summary-korean", prompt)