import asyncio
from pydantask.agent import DeepAgent
from pydantic import BaseModel
from pydantic_ai.agent import Agent


agent_registry = {}

da = DeepAgent(
    "I need help planning a trip to scotland. please provide a compiled trip plan itinerary."
)
result = asyncio.run(da.run())

print(result)
