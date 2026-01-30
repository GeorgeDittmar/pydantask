import asyncio
from pydantask.agent import DeepAgent
from pydantic import BaseModel
from pydantic_ai.agent import Agent


agent_registry = {}

da = DeepAgent(
    "Give me a detailed report on what tax laws changed this year compared to last year. Output a single comprehensive report with citations to markdown.",
)
result = asyncio.run(da.run())

print(result)
