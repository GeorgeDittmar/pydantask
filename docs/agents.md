# Agent Concepts

Agents are the core entities in Pydantask.

At a high level, agents:

- Accept an objective (a prompt)
- Can call tools via function calling
- Can share mutable state via `RuntimeState` (`deps_type=RuntimeState`)

This page documents how `DeepAgent` works **as implemented today**.

## DeepAgent: The orchestrator

`DeepAgent` coordinates a multi-step run over a shared `RuntimeState`.

### Constructor

The public constructor accepts several optional overrides; the parameters that materially affect the current orchestration behavior are:

- `prompt`: overall objective
- `model`: model identifier or `pydantic_ai.models.Model` instance. Strings may be
  bare model names (defaulting to the OpenAI provider) or provider-prefixed
  values such as `"openai:gpt-4.1-mini"` or `"anthropic:claude-sonnet-4-5"`.
- `max_steps`: outer-loop limit
- `sub_agents`: additional capabilities to register
- `trace`: whether to enable tracing auto-detection
- `output_type`: the producer’s output type (defaults to `TaskResult`)

```python docs/agents.md
from pydantask.agents import DeepAgent
from pydantask.models import TaskResult

agent = DeepAgent(
    prompt="...",
    model="gpt-4.1-mini",
    max_steps=20,
    trace=True,
    output_type=TaskResult,
)
```

The default `research_agent` capability uses Tavily web search when `TAVILY_API_KEY`
 is set; if it is missing, it falls back to a built-in DuckDuckGo search tool
 instead of raising an error.

### Return type

`DeepAgent.run()` returns a `DeepAgentRunResult`:

- `final_result: TaskResult | None`
- `plan: Dict[int, TaskItem]`
- `runtime_state: RuntimeState`

## What happens in `run()`

`DeepAgent.run()` is an outer loop that repeats until either:

- the supervisor says the work is complete (`all_tasks_completed=True`), or
- `max_steps` is reached.

Each cycle:

1. **Supervisor decision**
   - `DeepAgent` calls the supervisor agent with a formatted “mission control board” view of:
     - the current plan (`RuntimeState.plan`)
     - task statuses and dependency edges
     - available capability names/descriptions
   - The supervisor returns a `SupervisorDecision` with:
     - `tasks_to_execute`: task IDs it wants to run next
     - `feedback_to_subagents`: optional per-task guidance
     - `all_tasks_completed`: whether to stop
   - The supervisor can also update the plan at runtime using DeepAgent tools:
     - `add_task`, `cancel_task`, `patch_task`, `update_task_status`.

2. **Execute ready tasks (parallel)**
   - `DeepAgent._execute_ready_tasks(...)` filters the supervisor’s requested tasks to those whose dependencies are satisfied.
   - Dependency rule: a task can run only if every `sub_task_dependency` task has status `COMPLETED`.
   - Eligible tasks execute concurrently via `asyncio.TaskGroup`.
   - When a task runs:
     - it is set to `RUNNING`
     - once the sub-agent returns, it is set to `NEEDS_REVIEW` and its `result` is stored.

3. **Critic / QA**
   - For each executed task, the critic agent evaluates whether the produced `TaskResult` satisfies the task objective.
   - The critic returns `TaskQAResult(passed=..., reasoning=...)`.
   - `handle_critic_result(...)` currently records the latest critic review and
     increments the task’s `attempt_count`, but does **not** automatically
     change `TaskItem.status`. Any transition to `COMPLETED`, `FAILED`, or
     other states must be driven by higher-level logic (e.g., via
     `update_task_status(...)`).

Between cycles, the `RuntimeState` is mutated in-place and, as implemented
 today, a JSON checkpoint of the state is written under `_checkpoint/` for the
 current run. The supervisor therefore sees the updated status board on the
 next iteration.

## RuntimeState, TaskItem, and status

The shared state is the `RuntimeState` model (see `docs/models.md`). The key fields used by the orchestration loop are:

- `plan: Dict[int, TaskItem]`
- `objective: str`
- `next_task_id: int` (used by `add_task(...)`)

`TaskItem.status` uses `TaskStatus` values such as:

- `READY`, `RUNNING`, `NEEDS_REVIEW`, `COMPLETED`, `FAILED`, `CANCELLED`

## Capabilities (sub-agents)

A task is executed by a capability named in `TaskItem.capability`.

Capabilities are stored in `DeepAgent.agent_registry` as `CapabilityDescription` entries:

- `name`: the string used in `TaskItem.capability`
- `description`: human-readable summary (shown to the supervisor)
- `tool_func`: an `Agent` instance (or callable) used to execute that task

### Default capabilities

By default, DeepAgent registers:

- `research_agent` — uses Tavily web search (when `TAVILY_API_KEY` is present)
  or a DuckDuckGo-based search tool to gather information and return a cited
  `TaskResult`.
- `producer_agent` — reads completed tasks and synthesizes a final `TaskResult`.
- `worker_agent` — a general-purpose worker for analysis, summarization,
  document/code/log interpretation, and other tasks that operate on existing
  context.

You can add additional capabilities by passing `sub_agents=[CapabilityDescription(...)]` into `DeepAgent.__init__`.