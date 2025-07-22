import os
import click
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_teddynote import logging
from langchain_core.prompts import PromptTemplate

load_dotenv()

logging.langsmith("CH01-Basic")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.4, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

@click.group()
def cli():
    pass

@cli.command()
def sol1(): # 직접적으로 데이터를 넣어주기
    prompt = PromptTemplate.from_template("{num}의 10배는?")

    chain = prompt | llm

    print(chain.invoke({"num": 5}))

    # 변수가 1개일 때는 값만 전달하기 가능
    print(chain.invoke(5))

@cli.command()
def sol2(): # Runnable 방식으로 데이터 전달.Runnable은 특정 인자를 어떻게 가공 및 다른 소스로부터 가져올지를 통제하기 위해 사용
    from langchain_core.runnables import RunnablePassthrough

    prompt = PromptTemplate.from_template("{num}의 10배는?")
    RunnablePassthrough().invoke({"num": 10})
    runnable_chain = {"num": RunnablePassthrough()} | prompt | llm
    print(runnable_chain.invoke(10))

    RunnablePassthrough().invoke({"num": 10})
    (RunnablePassthrough.assign(new_num=lambda x:x["num"] * 3)).invoke({"num": 1})

@cli.command()
def sol3(): # RunnableParallel로 여러 프롬프트 동시 실행
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables import RunnableParallel

    runnable = RunnableParallel (
        passed=RunnablePassthrough(),
        extra=RunnablePassthrough.assign(mult=lambda x:x["num"] * 3),
        modified=lambda x:x["num"] + 1,
    )

    runnable.invoke({"num": 1})

    #chain에 RunnableParallel 적용
    chain1 = (
        {"country" : RunnablePassthrough()}
        | PromptTemplate.from_template("{country}의 수도는?")
        | llm
    )
    chain2 = (
        {"country" : RunnablePassthrough()}
        | PromptTemplate.from_template("{country}의 면적은?")
        | llm
    )

    combined_chain = RunnableParallel(capital=chain1, area=chain2)
    print(combined_chain.invoke("대한민국"))

@cli.command()
def sol4(): #RunnableLambda를 통한 사용자 정의 함수 매핑
    from datetime import datetime
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    def get_today(a):
        return datetime.today().strftime("%b-%d")

    prompt = PromptTemplate.from_template(
        "{today}가 생일인 유명인 {n} 명을 나열하세요. 생년월일을 표기해 주세요."
    )

    chain = (
        {"today": RunnableLambda(get_today), "n": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(chain.invoke(3))
    
if __name__ == "__main__":
    cli()