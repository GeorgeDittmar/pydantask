from pydantask.models import Plan, TaskItem
from pydantask.agents import DeepAgent
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from pathlib import Path
import asyncio

load_dotenv(find_dotenv())

checkpoint_dir = Path("_checkpoint") / "dulce_run"

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
    "Research for me the dulce base incident. Try to find primary sources but be sure to also search for any other sources in case tehre are no strong primary sources. write youre report to a markdown file. Be sure your document has all citations fully listed at the end. Make sure its cited in a format that allows the reader to check sources.",
    model="gpt-5.4",
    trace=True,
    checkpoint=True,
    checkpoint_dir=checkpoint_dir,
)

result = asyncio.run(da.run())

print(f"Checkpoint events saved to: {da.checkpoint_path}")

# pprint(result.model_dump())
# Write JSON data to a file

with open("output_book2.json", "w", encoding="utf-8") as json_file:
    json_file.write(result.model_dump_json(indent=2))

final_output = result.final_result.detailed_output if result.final_result else ""
with open("result_scifi_book2.md", "w", encoding="utf-8") as f:
    f.write(final_output)

pprint(final_output or "<no final result>")
