import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()
agent_registry = {}

da = DeepAgent(
    "Explain to me the IRS tax changes for this year and compare them to last year. Write a single report with your findings and citations",
    model="gpt-5.1",
    trace=True,
)
result = asyncio.run(da.run())

# print(result)
