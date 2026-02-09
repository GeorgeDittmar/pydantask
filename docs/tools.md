# Tools and Sub-agents

A **tool** is any callable an agent may use. Tools may include:
- Simple functions (e.g., time, file IO, reflection)
- Sub-agents (agents with their own prompt, state access, and toolsets)

## Writing a Tool

Simple example:

```python
async def get_current_datetime():
    from datetime import datetime
    return datetime.now().isoformat()
```

## Registering a Sub-agent

Sub-agents (for synthesis, research, etc.) are registered the same way as tools and enable hierarchical, compositional workflows.

You can pass tools to your agent registry on creation:

```python
from pydantask.tools.default_tools import get_current_datetime

tools = [get_current_datetime, ...]
agent = DeepAgent(prompt="...", tools=tools)
```
