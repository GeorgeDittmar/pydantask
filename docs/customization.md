# Customization

You can add custom sub-agents to your DeepAgent setup.

## Adding a Custom Sub-Agent

Your DeepAgent only accepts other **agents** (not simple tools) in its `sub_agents` parameter. Each sub-agent must be described by an `CapabilityDescription` (see API reference).

Example sub-agent registration:

```python
from pydantask.models import CapabilityDescription
from pydantic_ai import Agent

my_sub_agent = Agent(
    ...,
    # include any tools it needs
)
custom_description = CapabilityDescription(
    name="my_special_agent",
    description="Custom agent for specialized reasoning tasks.",
    capabilitiy=my_sub_agent
)

agent = DeepAgent(prompt="...", sub_agents=[custom_description])
```

If your sub-agent needs access to the plan, shared state, or persistent memory, it should set `deps_type=RuntimeState`.

```python
from pydantask.models import RuntimeState
my_context_agent = Agent(
    ...,
    deps_type=RuntimeState,
    ...
)
```

**Important:**
- Sub-agents that want to interact with DeepAgent's orchestration must use `deps_type=RuntimeState` and accept `deps`.
- Agents with different context/state types are possible but integration is not guaranteed. You must handle such cases yourself.

Stateless agents (not using runtime context) do not need to accept the deps parameter.


For more API detail, see [API Reference](api.md).
