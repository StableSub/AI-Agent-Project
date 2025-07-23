import os
from dotenv import load_dotenv
from enum import Enum
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.enum import EnumOutputParser

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH03-OutputParser"

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class Colors(Enum):
    RED = "빨간색"
    GREEN = "초록색"
    BLUE = "파란색"

parser = EnumOutputParser(enum=Colors)
    
prompt = PromptTemplate.from_template(
    """다음의 물체는 어떤 색깔인가요?
    Object: {object}
    Instructions: {instructions}"""
).partial(instructions=parser.get_format_instructions())

chain = prompt | llm | parser

response = chain.invoke({"object": "하늘"})
print(response)