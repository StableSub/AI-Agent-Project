import os
import click
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH04-Models"

@click.group()
def cli():
    pass

llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate.from_template(
    "{country}에 대해서 200자 내외로 요약해줘."
)

chain = prompt | llm

@cli.command()
def sol1():
    set_llm_cache(InMemoryCache())

    response = chain.invoke({"country": "대한민국"})
    print(response)

    response = chain.invoke({"country": "대한민국"})
    print(response)

@cli.command()
def sol2():
    if not os.path.exists("cache"):
        os.makedirs("cache")
        
    set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))
    
    response = chain.invoke({"country": "대한민국"})
    print(response)

    response = chain.invoke({"country": "대한민국"})
    print(response)
    
cli()