# Quickstart

Install dependencies from PyPI:

```sh
pip install pydantask
```

For development installs (from a cloned repo):

```sh
pip install -e .
```

You will also need at least:

- `OPENAI_API_KEY` set in your environment for the language model backend.
- (Optional) `TAVILY_API_KEY` if you want the default research agent to use
  Tavily web search; otherwise it will fall back to a built-in DuckDuckGo-based
  search tool.

A simple `.env` file (used with `python-dotenv`) might look like:

```bash
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."  # optional; omit to use DuckDuckGo-based search
```

Minimal usage (async):

```python 
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
