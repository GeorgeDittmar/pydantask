import asyncio
from pydantask.agent import DeepAgent
from pydantic import BaseModel
from pydantic_ai.agent import Agent


agent_registry = {}

da = DeepAgent(
    "Explain to me the IRS tax changes for this year and compare them to last year. Write a single report with your findings and citations",
)
result = asyncio.run(da.run())

# print(result)
