import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH03-OutputParser"

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(
        name="source",
        description="사용자의 질문에 답하기 위해 사용된 '출처', '웹사이트 주소' 이어야 합니다.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(

    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

chain = prompt | llm | output_parser

print(chain.invoke({"question": "대한민국 수도는 어디인가요?"}))