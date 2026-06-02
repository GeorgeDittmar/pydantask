# Customization

You can add custom sub‑agents (capabilities) to your `DeepAgent` setup.

## Adding a Custom Capability

`DeepAgent` accepts additional capabilities via the `sub_agents` parameter. Each capability is described by a `CapabilityDescription` (see API reference) and can wrap either:

- a full `pydantic_ai.Agent` instance, or
- a simple async/sync callable (a plain tool function).

These are merged with the built‑in capabilities (`producer_agent`, `research_agent`,
`worker_agent`) inside `_setup_default_sub_agents`.

### Example: custom sub‑agent (Agent)

```python
from pydantask.agents.agent import DeepAgent
from pydantask.models import CapabilityDescription, RuntimeState
from pydantic_ai import Agent

my_sub_agent = Agent(
    model=...,  # e.g. an OpenAIChatModel
    name="_my_special_agent",
    system_prompt="You are a custom agent for specialized reasoning tasks.",
    deps_type=RuntimeState,  # optional, if you need shared plan/state
    tools=[...],             # any tools this agent should be able to call
)

custom_description = CapabilityDescription(
    name="my_special_agent",  # used in TaskItem.capability
    description="Custom agent for specialized reasoning tasks.",
    tool_func=my_sub_agent,
)

agent = DeepAgent(objective="...", sub_agents=[custom_description])
```

### Example: simple tool as a capability

You can also register a plain function if you want the planner/supervisor to be able to delegate tasks directly to it:

```python
from pydantask.agents.agent import DeepAgent
from pydantask.models import CapabilityDescription, RuntimeState
from pydantic_ai import RunContext

async def my_utility_tool(ctx: RunContext[RuntimeState], payload: str) -> str:
    """Do something simple with the shared RuntimeState and a payload string."""
    # ... use ctx.deps.plan, ctx.deps.document_store, etc. if needed
    return f"processed: {payload}"

my_utility_capability = CapabilityDescription(
    name="my_utility_tool",
    description="Utility capability that performs a simple transformation.",
    tool_func=my_utility_tool,
)

agent = DeepAgent(objective="...", sub_agents=[my_utility_capability])
```

The planner sees the `name` and `description` in `CapabilityDescription` and may choose that capability when constructing `TaskItem.capability` values.

## Accessing RuntimeState in Custom Agents

If your sub‑agent (a `pydantic_ai.Agent`) needs access to the plan, shared state, or persistent memory, it should set `deps_type=RuntimeState`:

```python
from pydantask.models import RuntimeState
from pydantic_ai import Agent

my_context_agent = Agent(
    model=...,
    name="_my_context_agent",
    system_prompt="You can see and update the shared plan via RuntimeState.",
    deps_type=RuntimeState,
    tools=[...],
)
```

In this case, when `DeepAgent` calls `my_context_agent.run(...)`, it passes `deps=runtime_state`, and your tools can access `ctx.deps.plan`, `ctx.deps.document_store`, etc.

**Important:**

- Capabilities that need to interact with DeepAgent's orchestration (plan, task statuses, shared documents) should:
  - use `deps_type=RuntimeState` on their `Agent`, and
  - ensure any tools they call that need state accept a `RunContext[RuntimeState]`.
- Agents with different context/state types are possible but integration with `DeepAgent`'s orchestration is not guaranteed; you must handle such cases yourself.
- Stateless agents or tools (that do not use shared runtime state) do not need to accept `deps` or `RunContext`.

For more API detail, see [API Reference](api.md).
