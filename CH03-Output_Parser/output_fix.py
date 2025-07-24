import os
from dotenv import load_dotenv
from enum import Enum
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser
from typing import List

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH03-OutputParser"

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(
        description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

# 잘못된 형식을 일부러 입력
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
print(misformatted)

actor = new_parser.parse(misformatted)
print(actor)