import asyncio

from dotenv import load_dotenv

from pydantask.agents import DeepAgent

load_dotenv()

deep_agent = DeepAgent(
    objective="Write a market analysis for LLM tooling in 2026.",
    model="gpt-5.4",
    verbose_logging=True,
    trace=True,
)

final_state = asyncio.run(deep_agent.run())

print(final_state.model_dump_json(indent=2))
