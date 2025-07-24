import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH01-Basic"

# 랭체인에서 제공하는 LLM Wrapper 클래스
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.4, 
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 템플릿을 만들어서 프롬프트에 적용
template = "{country}의 수도는 어디인가요?"
prompt_template = PromptTemplate.from_template(template)

output_parser = StrOutputParser()

# 체인을 통해 프롬프트를 모델로, 모델 출력 결과를 output_parser로 전달
chain = prompt_template | llm | output_parser

print(chain.invoke({"country": "대한민국"}))