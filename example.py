import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()
agent_registry = {}

da = DeepAgent(
    "Give me a detailed report on strange paranormal facts about Appalacha. Cite all sources in the final document.",
    model="gpt-5.1",
    trace=True,
)
result = asyncio.run(da.run())

# print(result)
