import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()

deep_agent = DeepAgent(
    prompt="Write a market analysis for LLM tooling in 2026.",
    model="gpt-5.4",
    verbose_logging=True,
    trace=True,
)

final_state = asyncio.run(deep_agent.run())

print(final_state.model_dump_json(indent=2))
