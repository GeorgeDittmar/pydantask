from pydantask.models import Plan, TaskItem
from pydantask.agents import DeepAgent
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from pathlib import Path
import asyncio

load_dotenv(find_dotenv())

checkpoint_dir = Path("_checkpoint") / "pydantask5"

da = DeepAgent(
    # "I need a report of the news for today. Give me a high level summary and then a detailed version of major pieces of news as it pertains to the US and world. Output the report as markdown withj citations in the report. You must cite all sources at the end of the article.",
    #     """I am testing your Deep Agent ability to plan and execute on an objective. I want to test your ability to create plans.
    # Create 2 Research tasks. Besure to have mentioned in the instructions to not actually search but make something up.
    #   - 1: Pretend to Research about the man on the moon.
    #   - 2: Pretend to Research why stone henge is green.
    # Then have a producer task that requires task 1 and 2 to be completed. Have the producer write a little song about cats in space.
    # Then create 2 more Research tasks depending on the Producer task. Again these are pretend so have the instructions to teh researcher not actually perform the work and just make something up.
    #   - 1. Pretend to research hollow moon.
    #   - 2. Pretend to research quantom physics.
    # Again this is a test to see how well you follow creating a plan.""",
    "I need a compare and contrast report on pydantic ai's new harness ability and how that intersects with pydantask, a deep agent harness I have been working on. Pydantask can be found on github, https://github.com/GeorgeDittmar/pydantask,  as well as read teh docs, https://pydantask.readthedocs.io/en/latest/,  and pypi, https://pypi.org/project/pydantask/. I want a full report in markdown with a breakdown of the two, where they overlap and what layers of the stack they really solve for.",
    model="gpt-5.4",
    trace=True,
    checkpoint=True,
    checkpoint_dir=checkpoint_dir,
)

result = asyncio.run(da.run())

print(f"Checkpoint events saved to: {da.checkpoint_path}")

# pprint(result.model_dump())
# Write JSON data to a file

with open("pydantask_overview.json", "w", encoding="utf-8") as json_file:
    json_file.write(result.model_dump_json(indent=2))

final_output = result.final_result.detailed_output if result.final_result else ""
with open("pydantask_compare_contrast.md", "w", encoding="utf-8") as f:
    f.write(final_output)

pprint(final_output or "<no final result>")
