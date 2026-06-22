# Agent Concepts

Agents are the core entities in Pydantask.

At a high level, agents:

- Accept an objective (a task description)
- Can call tools via function calling
- Can share mutable state via `RuntimeState` (either as `deps_type=RuntimeState` or via `deps_type=TaskRunDeps` where `TaskRunDeps.runtime_state` is the shared state)

This page documents how `DeepAgent` works **as implemented today**.

## DeepAgent: The orchestrator

`DeepAgent` coordinates a multi-step run over a shared `RuntimeState`.

### Constructor

The public constructor accepts several optional overrides; the parameters that materially affect orchestration are:

- `objective`: overall objective
- `model`: model identifier or `pydantic_ai.models.Model` instance. Strings may be
  bare model names (defaulting to the OpenAI provider) or provider-prefixed
  values such as `"openai:gpt-4.1-mini"` or `"anthropic:claude-sonnet-4-5"`.
- `seed_plan`: optional predefined `Plan` (useful for fixed/hybrid DAGs)
- `planning_mode`: `"llm" | "fixed" | "hybrid"`
- `max_steps`: outer-loop limit
- `set_token_budget`: optional best-effort global token budget (the run stops when exceeded)
- `sub_agents`: additional capabilities to register
- `trace`: whether to enable tracing auto-detection
- `checkpoint`: enable event-sourced checkpointing
- `checkpoint_dir`: optionally specify a checkpoint directory
- `run_from_checkpoint`: replay checkpoint events from `checkpoint_dir`
- `verbose_logging`: log richer debugging information during execution

```python docs/agents.md
from pydantask.agents import DeepAgent

agent = DeepAgent(
    objective="...",
    model="gpt-4.1-mini",
    max_steps=20,
    trace=True,
    checkpoint=True,
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
- `errors: list[str]` (top-level warnings/errors, e.g. safety-stop reasons)

## What happens in `run()`

`DeepAgent.run()` is an outer loop that repeats until one of these happens:

- **Completion (guarded):** the supervisor returns `all_tasks_completed=True` *and* a single task is marked `is_final=True` and is `COMPLETED` with a `TaskResult`.
- **Safety stop:** `max_steps` is reached.
- **No-progress stop:** the harness detects repeated cycles where no tasks can run and stops to avoid an infinite loop (the final cycle includes a deterministic deadlock report in the supervisor prompt).

Each cycle:

1. **Deterministic scheduler pass (no LLM)**
   - Before calling the supervisor, DeepAgent normalizes task readiness:
     - `PENDING → READY` when dependencies are satisfied.
     - `READY → PENDING` when dependencies are *not* satisfied (keeps the status board honest).
     - Non-terminal tasks with an unknown capability are marked `ERRORED`.

2. **Supervisor decision (LLM)**
   - `DeepAgent` calls the supervisor agent with a formatted “mission control board” view of:
     - the current plan (`RuntimeState.plan`)
     - task statuses and dependency edges
     - available capability names/descriptions
     - deterministic scheduler notes (if any)
   - The supervisor returns a `SupervisorDecision` with:
     - `tasks_to_execute`: task IDs it wants to run next
     - `feedback_to_subagents`: optional per-task guidance
     - `all_tasks_completed`: whether to stop
   - The supervisor can also update the plan at runtime using DeepAgent tools (depending on `planning_mode`):
     - always: `cancel_task`, `update_task_status`, `view_qa_report`
     - in `llm`/`hybrid`: `add_task`, `patch_task`, `mark_final_task`

3. **Execute ready tasks (parallel)**
   - `DeepAgent._execute_ready_tasks(...)` filters the supervisor’s requested tasks to those whose dependencies are satisfied.
   - Dependency rule: a task can run only if every `sub_task_dependency` task has status `COMPLETED`.
   - Eligible tasks execute concurrently via `asyncio.TaskGroup`.
   - When a task runs:
     - it is atomically claimed (`READY`/`RERUN` → `RUNNING`) under a lock (prevents double-scheduling)
     - once the sub-agent returns, it is set to `NEEDS_REVIEW` and its `result` is stored

4. **Critic / QA**
   - For each executed task, the critic agent evaluates whether the produced `TaskResult` satisfies the task objective.
   - The critic returns `TaskQAResult(passed=..., reasoning=...)`.
   - `handle_critic_result(...)` applies a deterministic transition to the `TaskItem`:
     - if `passed=True` → `TaskItem.status = COMPLETED`
     - else if `attempt_count >= max_attempts` → `TaskItem.status = FAILED`
     - otherwise → `TaskItem.status = RERUN` and the critic feedback is appended to the task objective

Between cycles, the `RuntimeState` is mutated in-place.

If `checkpoint=True`, DeepAgent uses an **event-sourced checkpoint log** under `_checkpoint/<run-id>/` (or `checkpoint_dir` if provided):

- `events.jsonl`: append-only event log (task added/patched/status updates/results/etc.)
- `summaries.jsonl`: lightweight runtime summaries per cycle
- `task_results/`: optional sidecar JSON files when a `TaskResult.detailed_output` is too large for the event log

The supervisor therefore sees the updated status board on the next iteration.

## RuntimeState, TaskItem, and status

The shared state is the `RuntimeState` model (see `docs/models.md`). The key fields used by the orchestration loop are:

- `plan: Dict[int, TaskItem]`
- `objective: str`
- `next_task_id: int` (used by `add_task(...)`)

`TaskItem.status` uses `TaskStatus` values such as:

- `PENDING`, `READY`, `RUNNING`, `NEEDS_REVIEW`, `COMPLETED`
- `RERUN` (retry requested after QA failure)
- `FAILED` (QA rejected too many times)
- `ERRORED` (exception during execution; intentionally *not* treated as terminal so the supervisor can patch + rerun)
- `CANCELLED`

## Capabilities (sub-agents)

A task is executed by a capability named in `TaskItem.capability`.

Capabilities are stored in the agent's capability registry (implementation: `DeepAgent._capability_registry`) as `CapabilityDescription` entries:

- `name`: the string used in `TaskItem.capability`
- `description`: human-readable summary (shown to the supervisor)
- `tool_func`: a runnable capability implementation (typically a `pydantic_ai.Agent`, or a runner wrapper created with `pydantask.capabilities.runner.as_runner(...)`)

At runtime, the capability registry is also passed into `RuntimeState.capability_registry` (excluded from serialization) so tools and agents can reference it.

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