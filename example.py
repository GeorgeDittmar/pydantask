import asyncio
from pydantask.agent import DeepAgent
from pydantic import BaseModel
from pydantic_ai.agent import Agent


agent_registry = {}

da = DeepAgent(
    "Write a report for me on the Deep Research agent architecture and how it is implemented and works.",
    model="gpt-5.1",
)
result = asyncio.run(da.run())

# print(result)
