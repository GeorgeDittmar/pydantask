# PydanTask — Dynamic DAG Orchestrator for Pydantic AI

![Pydantask Logo](docs/imgs/pydantask_logo_v3.png)

PydanTask is an **alpha** orchestrator for building multi-step, self-correcting workflows on top of [Pydantic AI](https://ai.pydantic.dev/).

An LLM supervisor plans a task DAG at runtime, executes dependency-satisfied tasks in parallel, runs each result through a critic for QA, and iterates until the objective is met — or the budget is exhausted.

What you get:

- **Dynamic task DAGs**: a supervisor creates and patches a task graph at runtime
- **Parallel execution**: dependency-satisfied tasks run concurrently via `asyncio.TaskGroup`
- **Critic QA + retries**: failed tasks `RERUN` (with feedback appended) until `max_attempts`, then `FAILED`
- **Observability**: optional tracing (Langfuse, Logfire, LangSmith)
- **Recovery / auditability**: optional event-sourced checkpointing (`events.jsonl` + summaries + large-result sidecars)
- **Extensibility**: register custom capabilities via `CapabilityDescription` — `pydantic_ai.Agent` instances or plain async/sync callables

### Try it in ~3 minutes

1) Install:

```bash
pip install pydantask
```

2) Set env vars:

```bash
export OPENAI_API_KEY="..."
# Optional: enables Tavily web search; otherwise DuckDuckGo-based search is used
export TAVILY_API_KEY="..."
```

3) Run a minimal orchestrator:

```python
import asyncio
from pydantask.agents import PydanTask


async def main() -> None:
    orchestrator = PydanTask(
        objective="Compare 3 open-source LLMs for local inference and recommend one.",
        model="openai:gpt-4.1-mini",  # or "anthropic:..." or pass a Model instance
        trace=False,
        checkpoint=False,
        max_steps=10,
    )

    result = await orchestrator.run()
    print(result.final_result.detailed_output if result.final_result else result.errors)


if __name__ == "__main__":
    asyncio.run(main())
```

> Alpha note: the core loop is working and tested, but the API and prompts are still evolving. If you hit rough edges, please open an issue with a minimal repro.

For deeper docs and API reference, see: **[pydantask.readthedocs.io](https://pydantask.readthedocs.io/en/latest/)**

---

## High-Level Architecture

The core orchestrator is `PydanTask`:

```python
from pydantask.agents import PydanTask
```

`PydanTask` coordinates four built-in capabilities:

- **Supervisor** — plans the DAG, picks which tasks to run next, and decides when the run is complete.
- **Researcher** — performs web/external research for tasks that need new information.
- **Producer** — synthesizes intermediate results into a final answer or artifact.
- **Critic** — evaluates every task output and drives deterministic retry/fail transitions.

They all operate over a shared `RuntimeState`:

```python
from pydantask.models import RuntimeState, TaskItem, TaskResult, Plan
```

Key concepts:

- **Plan** (`Plan`):
  - `reasoning_steps`: the supervisor's internal planning notes
  - `tasks`: list of `TaskItem` instances
- **TaskItem**: one sub-task in the DAG, with:
  - `task_id`, `overall_objective`, `sub_task_objective`
  - `capability` (which node to use, e.g. `"research_agent"`)
  - `sub_task_dependencies` (other task IDs that must complete first)
  - `status` (`TaskStatus`: `PENDING`, `READY`, `RUNNING`, `NEEDS_REVIEW`, `COMPLETED`, `FAILED`, `ERRORED`, `RERUN`)
  - `result` (`TaskResult`) and `task_feedback` (`TaskQAResult`)
- **RuntimeState**:
  - `plan: Dict[int, TaskItem]`
  - `objective: str`
  - `capability_registry: Dict[str, CapabilityDescription]` *(excluded from serialization)*
  - `document_store`, `knowledge_store`, `runtime_steps`, etc.

The control loop in `PydanTask.run()`:

1. The supervisor incrementally builds a task DAG for the objective (via tools like `add_task`).
2. `RuntimeState` is initialized with the capability registry.
3. In each cycle:
   - The supervisor decides which tasks to execute next based on plan progress, dependencies, and self-reflection.
   - Ready tasks (with satisfied dependencies) are executed in parallel by their associated capability.
   - The critic reviews each result and produces a QA report; the supervisor uses it to retry or advance.
4. The loop stops when:
   - the supervisor sets `all_tasks_completed = True` **and** the run's completion invariants are met (exactly one `is_final=True` task, `COMPLETED` with a result), or
   - `max_steps` is reached, or
   - a no-progress guardrail fires (safety exit).

For more detail, see `docs/agents.md`.

---

## Installation & Setup

PydanTask assumes you already have Pydantic AI and an OpenAI-compatible model configured. A Tavily API key is **optional** for the built-in research agent (it falls back to DuckDuckGo search if omitted).

### 1. Install dependencies

From your project root:

```bash
pip install pydantask
```

(or however you manage your environment; if you use Poetry, adjust accordingly.)

### 2. Environment variables

Set the following environment variables (e.g. in your shell or a `.env` file):

- `OPENAI_API_KEY` — for the underlying model provider (or whatever your Pydantic AI provider expects).
- `TAVILY_API_KEY` — *(optional)* used by the `research_agent` (via `tavily_search_tool`). If this key is not set, it defaults to DuckDuckGo search.

---

## Quickstart: Running a PydanTask

Minimal example that creates a `PydanTask` and runs it on a single objective:

```python
import asyncio

from pydantask.agents import PydanTask

async def main() -> None:
    orchestrator = PydanTask(
        objective="Write an overview of ghost lights folklore and summarize scientific explanations.",
        model="gpt-4.1-mini",  # or any compatible OpenAIChatModel name
        max_steps=10,
    )

    run_result = await orchestrator.run()
    runtime_state = run_result.runtime_state

    # Inspect the final plan and results
    for task_id, task in sorted(runtime_state.plan.items()):
        print(f"Task {task_id} [{task.status}]: {task.sub_task_objective}")
        if task.result is not None:
            print("  Summary:", task.result.summary)
            # The main long-form output for the task is stored in-memory:
            print("  Detailed output:", (task.result.detailed_output or "<empty>"))
            print()

if __name__ == "__main__":
    asyncio.run(main())
```

What this does:

1. Constructs a `PydanTask` with default Supervisor, Researcher, Producer, and Critic.
2. Supervisor (dynamic DAG architect) breaks down the objective into `TaskItem`s using built-in capabilities.
3. Supervisor picks tasks to run in each loop iteration.
4. Researcher and Producer execute those tasks and return structured `TaskResult`s.
5. Critic evaluates each task result and marks tasks as:
   - `COMPLETED` when QA passes
   - `RERUN` when QA fails but retries remain (critic feedback is appended to the task objective)
   - `FAILED` when QA fails and `max_attempts` is exceeded
6. When done, you get a `PydanTaskRunResult` with the full plan and results.

> By default, PydanTask treats task outputs as **in-memory** (stored in `TaskResult.detailed_output`).
>
> It *does* support optional **event-sourced checkpointing** (`checkpoint=True`), which persists an append-only `events.jsonl` log (plus summaries and sidecar JSON files for large results) under `_checkpoint/`.
>
> Filesystem tools exist in `pydantask.tools.default_tools` but are **not enabled by default** in the built-in nodes.

---

## Customizing Capabilities

You can add custom task nodes (sub-agents or plain functions) via `CapabilityDescription` and the `capabilities` argument.

### Example: custom agent capability

```python
from pydantic_ai import Agent
from pydantask.agents import PydanTask
from pydantask.models import CapabilityDescription, TaskResult, TaskRunDeps

my_special_agent = Agent(
    model=...,  # e.g. the same OpenAIChatModel
    name="_my_special_agent",
    system_prompt="You are a specialized agent for security analysis.",
    deps_type=TaskRunDeps,   # gives tools access to deps.runtime_state + deps.task
    output_type=TaskResult,
    tools=[...],  # tools should typically accept RunContext[TaskRunDeps]
)

custom_capability = CapabilityDescription(
    name="security_agent",  # used in TaskItem.capability
    description="Performs security-focused analysis and risk assessment.",
    tool_func=my_special_agent,
)

orchestrator = PydanTask(
    objective="Assess the security posture of this web application.",
    capabilities=[custom_capability],
)

# Now the Supervisor can choose `security_agent` as a task node in the DAG.
```

### Example: simple function capability (callable node)

`PydanTask` expects a *capability* to be runnable (i.e. something with a `.run(prompt, deps, usage_limits=...)` method). For plain functions, wrap them with `as_runner(...)`.

```python
from pydantask.agents import PydanTask
from pydantask.capabilities.runner_v2 import as_runner
from pydantask.models import CapabilityDescription, TaskResult, TaskRunDeps


async def my_utility_capability(prompt: str, deps: TaskRunDeps) -> TaskResult:
    # prompt is the task prompt; deps.runtime_state + deps.task give you context
    return TaskResult(task_id=deps.task.task_id, summary="processed", detailed_output=prompt)


utility_capability = CapabilityDescription(
    name="my_utility_tool",
    description="Utility capability that processes a prompt and returns a TaskResult.",
    tool_func=as_runner(my_utility_capability),
)

orchestrator = PydanTask(objective="Some goal...", capabilities=[utility_capability])
```

For more customization details, see:

- `docs/customization.md`
- `docs/tools.md`
- `docs/agents.md`

---

## Running Unit Tests

Tests live under the `test/` directory and are compatible with both `pytest` and the standard library `unittest`.

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
