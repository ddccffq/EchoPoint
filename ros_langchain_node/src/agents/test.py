from langchain_openai import ChatOpenAI
from agent import Agent
from prompts import IMAGE_TO_TEXT_PROMPT
from pydantic import SecretStr

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=SecretStr("sk-6cdvyOZlgDDbEx35sO7pyX8l2lftAD1wafV9HrxvrQvfkCRw"),
    base_url="https://api.zhangsan.cool/v1",
)

agent = Agent(
    model=model,
    prompt=IMAGE_TO_TEXT_PROMPT,
)


