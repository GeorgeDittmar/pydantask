import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()

deep_agent = DeepAgent(
    prompt="Write a market analysis for LLM tooling in 2026.",
    model="gpt-5.4",
    # sub_agents=[
    #     AgentDescription(
    #         name="custom_research",
    #         description="Domain-specific research for developer tooling.",
    #         tool_func=my_custom_researcher,
    #     ),
    #     AgentDescription(
    #         name="writer",
    #         description="Turns research into long-form reports.",
    #         tool_func=my_custom_writer,
    #     ),
    # ],
    # optional: override planner/supervisor/critic if you want
)

final_state = asyncio.run(deep_agent.run())

print(final_state.model_dump_json(indent=2))
