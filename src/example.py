from pydantask.models import Plan, TaskItem
from pydantask.agents import DeepAgent
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from pathlib import Path
import asyncio

load_dotenv(find_dotenv())

checkpoint_dir = Path("_checkpoint") / "kg_taxonomy_research_v3"


task = """
Your task is to research any architectures that would be useful for automated knowledge graph / taxonomy generation from raw PDF text. Be sure to give a literature review as well on any research. 
The use case is to create a central source of information
for an api system where the API's return structured data and supporting text from a central  source of truth. this is to replace a current stack where we have a seperate rag system over pdf booklets and
a seperate api for similar data. The idea is when a system queries the API for a particular thing, for example a medical benefit, it then also is able to return associated text from teh callers pdf data
so we cover not only, structured information, but also cover unstructured context that only lives in these documents. We think that we will need a taxonomy of some sort so we can map, api call types to areas in 
these pdfs. I want to know any active reasearch in this space, any architectures that may exist, or help propose a possible archiectures. Make sure the report you write is detailed, has citation links that a reader
can reference, and any synthesis of recommendations you may see. write this report in markdown format with ALL citations at the end of the report. If you need to search the internet, be sure to split the research tasks out into small chunks instead of trying to 
solve things in large steps.

"""


da = DeepAgent(
    task,
    model="gpt-5.4",
    trace=True,
    checkpoint=True,
    checkpoint_dir=checkpoint_dir,
)

result = asyncio.run(da.run())

print(f"Checkpoint events saved to: {da.checkpoint_path}")

# pprint(result.model_dump())
# Write JSON data to a file

with open("literature_review_kg_2.json", "w", encoding="utf-8") as json_file:
    json_file.write(result.model_dump_json(indent=2))

final_output = result.final_result.detailed_output if result.final_result else ""
with open("literature_review_kg_2.md", "w", encoding="utf-8") as f:
    f.write(final_output)

pprint(final_output or "<no final result>")
