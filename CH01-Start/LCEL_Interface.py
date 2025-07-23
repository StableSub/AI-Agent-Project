import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH01-Basic"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.4, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate.from_template("{topic}에 대하여 3문장으로 설명해줘.")

chain = prompt | llm | StrOutputParser()

#stream : 실시간 출력
print("[stream 실행 결과]")
for response in chain.stream({"topic" : "멀티모달"}):
    print(response, end="", flush=True)

#invoke : 호출
print("\n\n[invoke 호출 결과]")
response = chain.invoke({"topic": "geminai"})
print(response)

#batch : 단위 실행
print("\n[batch 실행 결과]")
response = chain.batch([{"topic": "geminai"}, {"topic": "langchain"}])
print(response)

chain.batch(
    [
        {"topic": "ChatGPT"},
        {"topic": "인스타그랩"},
        {"topic": "랭체인"},
        {"topic": "랭그래프"},
        {"topic": "RAG"},
    ],
        config={"max_concurrency": 3},
)

#async를 통해 비동기로도 가능. 이때는 await를 사용하여 프로세스가 완료될 떄까지 대기 가능.