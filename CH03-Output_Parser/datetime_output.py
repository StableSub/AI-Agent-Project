import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH03-OutputParser"

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

output_parser = DatetimeOutputParser()
output_parser.format = "%Y-%m-%d"

template = """Answer the users question:\n\n#Format Instructions: \n{format_instructions}\n\n#Question: \n{question}\n\n#Answer:"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    },
)

chain = prompt | llm | output_parser

output = chain.invoke({"question": "Google 이 창업한 연도는?"})

print(output.strftime("%Y-%m-%d"))