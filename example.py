import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()
agent_registry = {}

da = DeepAgent(
    "Give me a detailed report on the mothman creature. Be sure that when you write the report that it is citing its sources correctly.",
    model="gpt-5.1",
    trace=True,
)
result = asyncio.run(da.run())

# print(result)
