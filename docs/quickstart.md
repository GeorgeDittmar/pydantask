# Quickstart

Install dependencies:

```sh
pip install pydantask
```

Minimal usage (async):

```python docs/quickstart.md
import asyncio

from pydantask.agents import DeepAgent


async def main() -> None:
    agent = DeepAgent(prompt="Research the best open source LLMs of 2024.")
    run_result = await agent.run()  # DeepAgentRunResult

    print("Objective:", run_result.objective)

    if run_result.final_result is not None:
        print("\nFinal summary:\n", run_result.final_result.summary)
        print("\nFinal detailed output (truncated):\n")
        print(run_result.final_result.detailed_output[:1000])


if __name__ == "__main__":
    asyncio.run(main())
```

For more depth, see the [Agent Concepts](agents.md) page.
