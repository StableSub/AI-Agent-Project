import os
import click
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from itertools import chain
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "CH03-OutputParser"

@click.group()
def cli():
    pass

llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

email_conversation = """From: 김철수 (chulsoo.kim@bikecorporation.me)
To: 이은채 (eunchae@teddyinternational.me)
Subject: "ZENESIS" 자전거 유통 협력 및 미팅 일정 제안

안녕하세요, 이은채 대리님,

저는 바이크코퍼레이션의 김철수 상무입니다. 최근 보도자료를 통해 귀사의 신규 자전거 "ZENESIS"에 대해 알게 되었습니다. 바이크코퍼레이션은 자전거 제조 및 유통 분야에서 혁신과 품질을 선도하는 기업으로, 이 분야에서의 장기적인 경험과 전문성을 가지고 있습니다.

ZENESIS 모델에 대한 상세한 브로슈어를 요청드립니다. 특히 기술 사양, 배터리 성능, 그리고 디자인 측면에 대한 정보가 필요합니다. 이를 통해 저희가 제안할 유통 전략과 마케팅 계획을 보다 구체화할 수 있을 것입니다.

또한, 협력 가능성을 더 깊이 논의하기 위해 다음 주 화요일(1월 15일) 오전 10시에 미팅을 제안합니다. 귀사 사무실에서 만나 이야기를 나눌 수 있을까요?

감사합니다.

김철수
상무이사
바이크코퍼레이션
"""

@cli.command()
def sol1(): # 출력 파서를 사용하지 않는 경우
    prompt = PromptTemplate.from_template(
        "다음의 이메일 내용중 중요한 내용을 추출해 주세요.\n\n{email_conversation}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"email_conversation" : email_conversation})
    
    # pydantic 스타일로 정의된 클래스를 사용하여 정보를 파싱
    class EmailSummary(BaseModel):
        person: str = Field(description="메일을 보낸 사람")
        email: str = Field(description="메일을 보낸 사람의 이메일 주소")
        subject: str = Field(description="메일 제목")
        summary: str = Field(description="메일 본문을 요약한 텍스트")
        date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")
    parser = PydanticOutputParser(pydantic_object=EmailSummary)
    print(parser.get_format_instructions())
    
    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    QUESTION:
    {question}

    EMAIL CONVERSATION:
    {email_conversation}

    FORMAT:
    {format}
    """
    )

    # foramt에 PydanticOutputParser의 부분 포매팅 추가
    prompt = prompt.partial(format=parser.get_format_instructions())

    chain = prompt | llm
    
    response = chain.invoke(
        {
            "email_conversation": email_conversation,
            "question": "이메일 내용 중 주요 내용을 추출해 주세요",
        }
    )
    print(response)
    
    #EmailSummary 객체로 변환
    structed_output = parser.parse(response)
    print(structed_output)
    
    chain = prompt | llm | parser
    
    response = chain.invoke(
        {
            "email_conversation": email_conversation,
            "question": "이메일 내용 중 주요 내용을 추출해 주세요",
        }
    )
    
    print(response)
    
    # 구글은 아직 지원하지 않음
    # llm_with_structed = GoogleGenerativeAI(
    #     model = "gemini-2.5-flash",
    #     temperature=0.4,
    #     google_api_key = os.getenv("GOOGLE_API_KEY"),
    # ).with_structured_output(EmailSummary)
    
    # response = llm_with_structed.invoke(email_conversation)
    # print(response)
    
cli()    