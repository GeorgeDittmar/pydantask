import asyncio

from pydantask.agents import DeepAgent

agent_registry = {}

da = DeepAgent(
    "Explain to me the IRS tax changes for this year and compare them to last year. Write a single report with your findings and citations",
)
result = asyncio.run(da.run())

# print(result)
