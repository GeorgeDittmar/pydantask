import asyncio

from pydantask.agents import DeepAgent
from dotenv import load_dotenv

load_dotenv()
agent_registry = {}

da = DeepAgent(
    "Generate a detailed report on paranormal phenomena known as the ghost lights. Include historical accounts, scientific investigations, and cultural interpretations. The report should be well-structured with sections for each aspect of the topic, and include citations for all sources used in text and at the end.",
    # "Give me an analysis of current deep agent frameworks and architectures in production. Each result or assertation must be cited clearly from where that information is from. This is to help validate the results, otherwise we cant trust the information.",
    # "I am traveling to London next month and want to know the best places to visit, eat, and stay. I also want to know about any events happening during that time. I would like a detailed itinerary for a 5 day trip that includes a mix of tourist attractions and local favorites. Please provide recommendations for each day, including activities, restaurants, and accommodations. Additionally, I would like to know about any cultural norms or tips for visiting London that I should be aware of.",
    model="gpt-5.1",
    trace=True,
)
result = asyncio.run(da.run())

# print(result)
