import os, click, pickle, json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.load import dumpd, dumps, load

load_dotenv()
os.environ["LAGNCHAIN_PROJECT"] = "CH04-Models"

llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate.from_template("{fruit}의 색상이 무엇입니까?")

chain = prompt | llm

print(f"Google AI: {chain.is_lc_serializable()}")

dumpd_chain = dumpd(chain)
print(type(dumpd_chain))

dumps_chain = dumps(chain)
print(type(dumps_chain))

# with open("fruit_chain.pkl", "wb") as f:
#     pickle.dump(dumpd_chain, f)
    
# with open("fruit_chain.pkl", "rb") as f:
#     loaded_chain = pickle.load(f)

# chain_from_file = load(loaded_chain)
# print(chain_from_file.invoke({"fruit": "사과"}))

# with open("fruit_chain.json", "w") as fp:
#     json.dump(dumpd_chain, fp)
    
# with open("fruit_chain.json", "r") as fp:
#     loaded_from_json_chain = json.load(fp)
#     loads_chain = load(loaded_from_json_chain)

# print(loads_chain.invoke({"fruit": "사과"}))

# 구글AI는 아직 직렬화 가능 상태가 아님.