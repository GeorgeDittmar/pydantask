import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()
agent_registry = {}

da = DeepAgent(
    "Give me an analysis of current deep agent frameworks and architectures in production. Each result or assertation must be cited clearly from where that information is from. This is to help validate the results, otherwise we cant trust the information.",
    model="gpt-5.1",
    trace=True,
)
result = asyncio.run(da.run())

# print(result)
