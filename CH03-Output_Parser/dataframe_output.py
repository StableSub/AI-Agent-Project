import os
import pandas as pd
import pprint
from dotenv import load_dotenv
from typing import Any, Dict
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH03-OutputParser"

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

def format_parser_output(parser_output: Dict[str, Any]) -> None:
    for key in parser_output.keys():
        parser_output[key] = parser_output[key].to_dict()
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)

df = pd.read_csv("./titanic.csv")
print(df.head())

parser = PandasDataFrameOutputParser(dataframe=df)

print(parser.get_format_instructions())

df_query = "Age column 을 조회해 주세요."
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)
chain = prompt | llm | parser
parser_output = chain.invoke({"query": df_query})
format_parser_output(parser_output)

df_query = "Retrieve the first row."
parser_output = chain.invoke({"query": df_query})
format_parser_output(parser_output)

print(df["Age"].head().mean())

df_query = "Retrieve the average of the Ages from row 0 to 4."
parser_output = chain.invoke({"query": df_query})
print(parser_output)

df_query = "Calculate average `Fare` rate."
parser_output = chain.invoke({"query": df_query})
print(parser_output)

print(df["Fare"].mean())