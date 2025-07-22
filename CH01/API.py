import os
from dotenv import load_dotenv
from langchain_teddynote import logging
import google.generativeai as genai

load_dotenv()
logging.langsmith("CH01-Basic")

genai.configure(api_key = os.environ["GOOGLE_API_KEY"])

llm = genai.GenerativeModel (
    model_name="gemini-2.5-flash",
    system_instruction="You are a helpful assistant who answers in Korean.",
    generation_config=genai.types.GenerationConfig(
        temperature=0.4,
        max_output_tokens=2048,
    )
)

# 답변에 대한 출력
response = llm.generate_content("한국의 날씨는 어때?")
print(response.text)

# 실시간 출력
response = llm.generate_content("한국의 날씨는 어때?", stream=True)
for token in response:
    print(token.text, end="", flush=True)

# 멀티 모델
IMAGE_URL = "https://t3.ftcdn.net/jpg/03/77/33/96/360_F_377339633_Rtv9I77sSmSNcev8bEcnVxTHrXB4nRJ5.jpg"

response = llm.generate_content([
    "이 이미지를 분석해줘",
    IMAGE_URL
], stream=True)

for token in response:
    print(token.text, end="", flush=True)