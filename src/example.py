import asyncio
import json
from pydantask.agents import DeepAgent
from pydantask.models import Plan, TaskItem
from dotenv import load_dotenv, find_dotenv
from pprint import pprint

load_dotenv(find_dotenv())
agent_registry = {}

da = DeepAgent(
    # "I need a report of the news for today. Give me a high level summary and then a detailed version of major pieces of news as it pertains to the US and world. Output the report as markdown withj citations in the report. You must cite all sources at the end of the article.",
    """I am testing your Deep Agent ability to plan and execute on an objective. I want to test your ability to create plans. Create this plan for solving some unknown task.

Create 2 Research tasks.
  - 1: Pretend to Research about the man on the moon.
  - 2: Pretend to Research why stone henge is green.

Then have a producer task that requires task 1 and 2 to be completed. Have the producer write a little song about cats in space.

Then create 2 more Research tasks depending on the Producer task
  - 1. Pretend to research hollow moon.
  - 2. Pretend to research quantom physics.
Again this is a test to see how well you follow creating a plan.""",
    model="gpt-5.4",
    trace=True,
)

result = asyncio.run(da.run())

# pprint(result.model_dump())
# Write JSON data to a file

with open("output_book2.json", "w") as json_file:
    json.dump(result.model_dump_json(), json_file, indent=4)

with open("result_scifi_book2.md", "w") as f:
    f.writelines(result.final_result.detailed_output)

pprint(result.final_result.detailed_output)
