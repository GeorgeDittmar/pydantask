import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()
agent_registry = {}

da = DeepAgent(
    "Give me a detailed report on the intersection with modern US conversatism, nationalism, and conpsiracy belief. I want to know breakdown of misinformation absorbtion as well with these groups vs say more liberal bases.",
    model="gpt-5.1",
    trace=True,
)
result = asyncio.run(da.run())

# print(result)
