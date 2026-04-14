import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
agent_registry = {}

da = DeepAgent(
    "I need a report of the news for today. Give me a high level summary and then a detailed version of major pieces of news as it pertains to the US and world. Output the report as markdown withj citations in the report. You must cite all sources at the end of the article.",
    model="gpt-5.4",
    trace=True,
)
result = asyncio.run(da.run())

from pprint import pprint

# pprint(result.model_dump())
# Write JSON data to a file
import json

with open("output_news.json", "w") as json_file:
    json.dump(result.model_dump_json(), json_file, indent=4)

with open("result_news.md", "w") as f:
    f.writelines(result.final_result.detailed_output)

pprint(result.final_result.detailed_output)
