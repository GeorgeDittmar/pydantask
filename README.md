# PydanTask: Deep Agentic Harness for Pydantic AI

![Pydantask Logo](docs/imgs/pydantask_logo_v3.png)

PydanTask is a Harness for building deep, autonomous agents that don't just respond—they **reason**, **decompose**, and **execute**.

It builds on top of [Pydantic AI](https://ai.pydantic.dev/) and adds:

- Long‑horizon planning and hierarchical task management
- A reusable orchestration loop (`DeepAgent`) with planner → supervisor → worker/researcher/producer → critic
- Shared runtime state across agents (`RuntimeState`)
- Extensible capabilities via `CapabilityDescription` and custom tools/agents

The goal is to give you a solid “agentic backbone” you can adapt, without having to reinvent multi‑step planning and control logic yourself.

For the most up‑to‑date implementation details and API docs, see the hosted documentation: **[PydanTask Documentation](https://pydantask.readthedocs.io/en/latest/)**.

---

## High-Level Architecture

The core orchestrator is `DeepAgent`:

```python
from pydantask.agents.agent import DeepAgent
```

`DeepAgent` coordinates several built‑in agents:

- **Supervisor** – plans and chooses which tasks to run next, based on statuses and dependencies.
- **Researcher** – performs web/external research for tasks that need new information.
- **Producer** – synthesizes intermediate results into a final answer or artifact.
- **Critic** – evaluates task outputs and drives deterministic retry/fail transitions.

They all operate over a shared `RuntimeState`:

```python
from pydantask.models import RuntimeState, TaskItem, TaskResult, Plan
```

Key concepts:

- **Plan** (`Plan`):
  - `reasoning_steps`: planner’s internal notes
  - `tasks`: list of `TaskItem` instances
- **TaskItem**: one sub‑task in the plan, with:
  - `task_id`, `overall_objective`, `sub_task_objective`
  - `capability` (which sub‑agent to use, e.g. `"research_agent"`)
  - `sub_task_dependencies` (other task IDs that must complete first)
  - `status` (`TaskStatus`: `PENDING`, `READY`, `RUNNING`, `NEEDS_REVIEW`, `COMPLETED`, `FAILED`, `ERRORED`, `RERUN`)
  - `result` (`TaskResult`) and `task_feedback` (`TaskQAResult`)
- **RuntimeState**:
  - `plan: Dict[int, TaskItem]`
  - `objective: str`
  - `agent_registry: Dict[str, CapabilityDescription]`
  - `document_store`, `knowledge_store`, `runtime_steps`, etc.

The control loop in `DeepAgent.run()`:

1. Supervisor creates a `Plan` for the objective. This plan is dynammically expanded upon as the DeepAgent runs.
2. `RuntimeState` is initialized with the plan and capabilities.
3. In each cycle:
   - Supervisor decides which tasks to execute next based on plan progress, task dependencies, and self reflection.
   - Ready tasks, so long as dependencies are satisfied, are executed by the appropriate capability (sub‑agent).
   - Critic reviews each result and produces a "QA" report for the supervisor to review if the task failed.
4. Loop stops when the Supervisor sets `all_tasks_completed = True` or `max_steps` is reached.

For more detail, see `docs/agents.md`.

---

## Installation & Setup

PydanTask assumes you already have Pydantic AI and an OpenAI‑compatible model configured. You’ll also need a Tavily API key for the built‑in research agent.

### 1. Install dependencies

From your project root:

```bash
pip install pydantask
```

(or however you manage your environment; if you use Poetry, adjust accordingly.)

### 2. Environment variables

Set the following environment variables (e.g. in your shell or a `.env` file):

- `OPENAI_API_KEY` – for the underlying OpenAIChatModel (or whatever your Pydantic AI provider expects).
- `TAVILY_API_KEY` – used by the `research_agent` (via `tavily_search_tool`). If this key is not set, defaults to DuckDuckGo search.

---

## Quickstart: Running a DeepAgent

Minimal example that creates a `DeepAgent` and runs it on a single objective:

```python
import asyncio

from pydantask.agents.agent import DeepAgent

async def main() -> None:
    agent = DeepAgent(
        prompt="Write an overview of ghost lights folklore and summarize scientific explanations.",
        model="gpt-4.1-mini",  # or any compatible OpenAIChatModel name
        max_steps=10,
    )

    runtime_state = await agent.run()

    # Inspect the final plan and results
    for task_id, task in sorted(runtime_state.plan.items()):
        print(f"Task {task_id} [{task.status}]: {task.sub_task_objective}")
        if task.result is not None:
            print("  Summary:", task.result.summary)
            print("  Outputs:", task.result.output_paths)
            print()

if __name__ == "__main__":
    asyncio.run(main())
```

What this does:

1. Constructs a `DeepAgent` with default Planner, Supervisor, Researcher, Producer, and Critic.
2. Planner breaks down the objective into `TaskItem`s (using built‑in capabilities `research_agent` and `producer_agent`).
3. Supervisor picks tasks to run in each loop iteration.
4. Researcher and Producer execute those tasks, writing notes/reports to `pydantask/tools/tmp_files/` when appropriate.
5. Critic evaluates each task result and marks tasks as `COMPLETED`, retryable (`READY`/`RERUN`), or `FAILED` based on configured retry limits.
6. When done, you get a `RuntimeState` with the full plan and results.

---

## Customizing Capabilities

You can add custom sub‑agents or tools via `CapabilityDescription` and the `sub_agents` argument.

### Example: custom agent capability

```python
from pydantic_ai import Agent
from pydantask.agents.agent import DeepAgent
from pydantask.models import CapabilityDescription, RuntimeState, TaskResult

my_special_agent = Agent(
    model=...,  # e.g. the same OpenAIChatModel
    name="_my_special_agent",
    system_prompt="You are a specialized agent for security analysis.",
    deps_type=RuntimeState,
    output_type=TaskResult,
    tools=[...],  # any tools it needs
)

custom_capability = CapabilityDescription(
    name="security_agent",  # used in TaskItem.capability
    description="Performs security-focused analysis and risk assessment.",
    tool_func=my_special_agent,
)

agent = DeepAgent(
    prompt="Assess the security posture of this web application.",
    sub_agents=[custom_capability],
)

# Now the Planner can choose `security_agent` as a capability in the plan.
```

### Example: simple function capability

```python
from pydantic_ai import RunContext
from pydantask.agents.agent import DeepAgent
from pydantask.models import CapabilityDescription, RuntimeState

async def my_utility_tool(ctx: RunContext[RuntimeState], payload: str) -> str:
    # do something simple with ctx.deps and payload
    return f"processed: {payload}"

utility_capability = CapabilityDescription(
    name="my_utility_tool",
    description="Utility capability that performs a simple transformation.",
    tool_func=my_utility_tool,
)

agent = DeepAgent(prompt="Some goal...", sub_agents=[utility_capability])
```

For more customization details, see:

- `docs/customization.md`
- `docs/tools.md`
- `docs/agents.md`

---

## Running Unit Tests

Tests live under the `test/` directory and are written to be compatible with both `pytest` and the standard library `unittest`.

### Recommended: pytest

From the repository root:

```bash
pip install pytest
pytest
```

### Using unittest directly

If you prefer `unittest`, you can still run the suite with:

```bash
python -m unittest discover -s test -p "test_*.py"
```

Make sure required environment variables (e.g. `TAVILY_API_KEY`, `OPENAI_API_KEY`) are set, or that tests patch them appropriately (as in `test/test_agent.py`).

---

