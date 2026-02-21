# Quickstart

Install dependencies:

```sh
pip install pydantask
```

Minimal usage:

```python
from pydantask.agents.agent import DeepAgent

agent = DeepAgent(prompt="Research the best open source LLMs of 2024.")
result = await agent.run()
print(result)
```

For more depth, see the [Agent Concepts](agents.md) page.
