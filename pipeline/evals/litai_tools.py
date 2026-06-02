import os
import asyncio
from litai import LLM
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

async def multiturn_conversation_with_litai(question: str, history: str) -> list:
    llm = LLM(model=os.getenv("LITAI_MODEL"), enable_async=True)
    response = await llm.chat(question, conversation=history)
    history = llm.get_history(history)
    return response, history

async def inference_with_litai(question: str) -> str:
    llm = LLM(model=os.getenv("LITAI_MODEL"), enable_async=True)
    return await llm.chat(question)

async def main():
    question = "What is the capital of France?"
    response = await inference_with_litai(question)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
