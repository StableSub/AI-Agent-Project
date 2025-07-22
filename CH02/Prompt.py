import os
import click
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_teddynote import logging
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.langsmith("CH02-Prompt")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

@click.group()
def cli():
    pass

@cli.command()
def sol1(): # from_template를 이용하여 데이터 전달
    template = "{country}의 수도는?"
    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    print(chain.invoke("대한민국").content)

    template = "{country1}과 {country2}의 수도는?"
    prompt = PromptTemplate(
        template=template,
        input_variables=["country1"], # 실행 시점에 반드시 성공해야 함 및 동적으로 지정 가능
        partial_variables={ # 템플릿 정의 시점에 미리 고정 됨
            "country2": "미국" 
        },
    )

    prompt_partial = prompt.partial(country2="캐나다") # partial_variables를 따로 수정 가능
    chain = prompt_partial | llm
    print(chain.invoke("대한민국").content)
    print(chain.invoke({"country1": "대한민국", "country2": "호주"}).content) # 실행 시 partial_variables도 지정이 가능

@cli.command()
def sol2(): # 항상 현재 날짜를 표기하기를 원하는 프롬프트를 작성하기 위해 날짜를 반환하는 함수를 사용하여 고정
    from datetime import datetime

    def get_today():
        return datetime.now().strftime("%B %d")

    prompt = PromptTemplate(
        template="오늘의 날짜는 {today}입니다. 오늘이 생일인 유명인 {n}명을 나열해주세요. 생년월일을 표기해주세요",
        input_variables=["n"],
        partial_variables={
            "today": get_today
        },
    )

    chain = prompt | llm

    print(chain.invoke(3).content)
    print(chain.invoke({"today": "Jan 02", "n": 5}).content)

@cli.command()
def sol3(): # 파일로부터 template 읽어오기
    from langchain_core.prompts import load_prompt
    prompt1 = load_prompt("prompts/fruit_color.yaml")
    print(prompt1.format(fruit="사과"))
    
    prompt2 = load_prompt("prompts/capital.yaml")
    print(prompt2.format(country="대한민국"))

@cli.command()
def sol4(): # ChatPromptTemlplate를 이용하여 대화목록을 프롬프트로 주입
    from langchain_core.prompts import ChatPromptTemplate
    
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
            ("human", "반가워요!"),
            ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
            ("human", "{user_input}"),
        ]
    )
    messages = chat_template.format_messages(
        name="정섭", user_input="당신의 이름은?"
    )
    llm.invoke(messages).content
    chain = chat_template | llm
    print(chain.invoke({"name": "StableSub", "user_input": "당신의 이름은?"}).content)

@cli.command()
def sol5(): # MessagePlaceholder를 이용하여 포맷하는 동안 렌더링할 메시지를 제어
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 요약 전문 AI 어시스턴트. 임무는 주요 키워드로 대화를 요약"
            ),
            MessagesPlaceholder(variable_name="conversation"),
            ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
        ]
    )
    formatted_chat_prompt = chat_prompt.format(
        word_count=5,
        conversation=[
            ("human", "저는 오늘 새로 입사한 정섭입니다. 반가워요"),
            ("ai", "잘 부탁 드립니다."),
        ],
    )
    print(formatted_chat_prompt)
    chain = chat_prompt | llm | StrOutputParser()
    print(chain.invoke(
            {
                "word_count": 5,
                "conversation": [
                    ("human", "저는 오늘 새로 입사한 정섭입니다. 반가워요."),
                    ("ai", "잘 부탁 드립니다.")
                ],
            }
        )
    )
    # 굳이 메세지를 넣는 이유는 LLM이 실제 대화 흐름을 이해하게 만들고, 그 흐름을 그래도 전달하여 자연스럽게 맥락 처리를 하기 위함

cli()

