# Supervisor Workflows & DAG Modes

This page explains how the **supervisor agent** in `DeepAgent` (see `src/pydantask/agents/agent.py`) can drive different workflow styles (task DAGs), including:

- **Fully seeded** (predefined DAG)
- **Dynamic / generative** (LLM builds and mutates the DAG at runtime)
- **Hybrid** (seeded backbone + dynamic extensions)

It focuses on *what the supervisor can do*, *which tools it has*, and *how task state + dependencies determine execution*.

---

## Mental model: what the supervisor controls

At runtime, `DeepAgent.run()` repeatedly:

1. Runs a deterministic scheduler pass to normalize readiness (`PENDING ↔ READY`) based on dependencies.
2. Builds a composite “status board” prompt for the supervisor (`_format_supervisor_input_prompt(runtime_state)`).
3. Calls the supervisor agent, which returns a `SupervisorDecision` (e.g. tasks to execute next, optional feedback, and optionally an “all done” flag).
4. Executes tasks that are **READY** (and dependency-satisfied) concurrently (`_execute_ready_tasks`).
5. Sends each executed task result to the critic for QA and applies a deterministic state transition (`handle_critic_result`).
6. Repeats until the supervisor signals completion, `max_steps` is reached, or a safety-stop triggers.

The plan is represented as a DAG stored in memory:

- `runtime_state.plan: dict[int, TaskItem]`
- Each `TaskItem` includes:
  - `task_id`
  - `sub_task_objective`
  - `capability` (which registered sub-agent executes it)
  - `sub_task_dependencies: list[int]` (edges in the DAG)
  - `status` (`TaskStatus` state machine)
  - `result` (structured `TaskResult` from the worker)
  - `task_feedback` (latest `TaskQAResult` from the critic)
  - `is_final: bool` (exactly one task should be marked final; DeepAgent uses this as a completion guardrail and to select `final_result`)

---

## Dependency gating (how the DAG is enforced)

A task is only eligible to run if:

- its status is runnable (`READY` or `RERUN`), **and**
- **all dependency tasks are `COMPLETED`**.

This is enforced in code by `_dependencies_satisfied()`.

Implications:

- The supervisor can “select” any task IDs, but the harness will skip tasks whose dependencies aren’t satisfied.
- To parallelize work, the supervisor should create multiple tasks with no dependencies (or with the same satisfied dependency).

---

## Task lifecycle (QA-driven state machine)

The harness uses these statuses during execution:

- `READY`: eligible to run if dependencies are satisfied.
- `RUNNING`: set immediately before dispatch.
- `NEEDS_REVIEW`: worker produced a result; critic QA still pending.
- `COMPLETED`: critic passed the result.
- `FAILED`: critic failed it too many times (`attempt_count >= max_attempts`).
- `RERUN`: treated as runnable by `_execute_ready_tasks`.
- `ERRORED`: exception occurred during execution.
- `CANCELLED`: supervisor cancelled the task.

### Deterministic QA transitions

After a task runs, the critic’s decision is applied in `handle_critic_result()`:

- If `review.passed` is `True` → task becomes `COMPLETED`.
- Else:
  - If `attempt_count >= max_attempts` → task becomes `FAILED`.
  - Otherwise → task becomes `RERUN`, and the objective is appended with the critic feedback (so the next attempt is guided).

This means:

- The supervisor does **not** “approve” results directly.
- The critic drives pass/fail.
- The supervisor mainly decides **what to run next**, and (in non-fixed modes) how to **extend or patch** the DAG.

---

## Planning modes (seeded vs dynamic vs hybrid)

The supervisor’s freedom is constrained by `planning_mode` via tool registration in `DeepAgent._default_supervisor_tools()`.

### 1) Fully seeded DAG (`planning_mode="fixed"`)

**Purpose:** Run a predetermined DAG without allowing the LLM supervisor to change its shape.

**Requirements:**

- `seed_plan` is required.
- On `run()`, `_apply_seed_plan(runtime_state)` loads tasks into `runtime_state.plan` and validates:
  - Unique `task_id`s
  - Dependencies refer to existing tasks

#### Loading a seeded DAG from YAML (recommended)

If you want users to supply a **pre-defined workflow plan** (a task DAG) you can store it as YAML and import it into a `Plan` using:

- `pydantask.agents.utils.import_yaml_workflow(path)`

This loader validates the YAML against a strict Pydantic schema (`WorkflowYamlConfig` / `WorkflowTaskConfig`), converts it into the canonical `Plan`/`TaskItem` models, and then enforces DAG constraints:

- no duplicate task IDs
- dependencies must reference existing tasks
- no dependency cycles

**Strict YAML contract** (no aliases):

```yaml docs/dags.md
objective: "Write a competitive analysis of X vs Y"
reasoning_steps: "Optional notes about this workflow"

tasks:
  - task_id: 1
    sub_task_objective: "Research product X (pricing, key features, positioning)"
    capability: "research_agent"
    sub_task_dependencies: []

  - task_id: 2
    sub_task_objective: "Research product Y (pricing, key features, positioning)"
    capability: "research_agent"
    sub_task_dependencies: []

  - task_id: 3
    sub_task_objective: "Synthesize final comparative analysis"
    capability: "producer_agent"
    sub_task_dependencies: [1, 2]
    is_final: true
```

**Python usage**:

```python docs/dags.md
import asyncio

from pydantask.agents import DeepAgent
from pydantask.agents.utils import import_yaml_workflow


async def main() -> None:
    seed_plan = await import_yaml_workflow("workflow.yml")

    agent = DeepAgent(
        objective=seed_plan.tasks[0].overall_objective,
        seed_plan=seed_plan,
        planning_mode="fixed",  # or "hybrid"
    )

    result = await agent.run()
    print(result.final_result)


if __name__ == "__main__":
    asyncio.run(main())
```

Notes:

- If the YAML does not mark a task with `is_final: true`, `import_yaml_workflow(..., auto_mark_final=True)` will deterministically mark the highest `task_id` task as final.
- For reproducibility, prefer explicit `is_final: true`.

**Supervisor tools (fixed mode):**

- `update_task_status(task_id, status)`
- `cancel_task(task_id, reason)`
- `view_qa_report(task_id)`
- `get_current_datetime`
- `think_tool`

**Not available to the supervisor in fixed mode:**

- `add_task(...)`
- `patch_task(...)`
- `mark_final_task(...)` *(you should set `is_final: true` in the seed plan/YAML instead)*

**Workflow style:** Strict DAG executor. Best for reproducibility and controlled pipelines.

---

### 2) Dynamic / generative DAG (`planning_mode="llm"`)

**Purpose:** Let the supervisor create and evolve the plan at runtime.

**Requirements:**

- `seed_plan` is *not* required.
- The runtime plan starts empty (`next_task_id = 1`).

**Supervisor tools (llm mode):**

Everything in fixed mode, plus:

- `add_task(sub_task_objective, capability, dependencies=None, metadata=None) -> int`
- `patch_task(task_id, sub_task_objective=None, dependencies=None)`
- `mark_final_task(task_id, reason=None)`

**Workflow style:** The supervisor can:

- Decompose the objective into tasks.
- Fan-out parallel worker/research tasks.
- Insert a final synthesis task (often `producer_agent`) that depends on upstream tasks.
- Modify or extend the DAG as gaps are discovered or QA fails.

---

### 3) Hybrid (`planning_mode="hybrid"`)

**Purpose:** Start with a seeded DAG backbone and allow the supervisor to extend or patch it when needed.

**Requirements:**

- `seed_plan` is required.
- Supervisor tools include `add_task`, `patch_task`, and `mark_final_task` (same as `llm`).

**Workflow style:** Structured but resilient—useful when you want a known baseline workflow plus dynamic recovery/extension.

---

## Common DAG patterns the supervisor can express

### A) Linear pipeline (simple chain)

Use when steps must happen in strict order.

- Example: Research → Analyze → Produce
- Dependencies: Analyze depends on Research; Produce depends on Analyze

### B) Fan-out / fan-in (parallel then synthesis)

Use when you want parallel sub-tasks (research, comparisons, extraction) followed by a single synthesis.

- Multiple tasks with no dependencies (or the same satisfied dependency)
- One final `producer_agent` task that depends on all upstream tasks

Execution detail: `_execute_ready_tasks()` uses an `asyncio.TaskGroup`, so tasks that are ready run concurrently.

### C) Iterative refinement (QA-driven retries)

Use when tasks may fail QA and need retries.

- Critic failure automatically returns the task to `RERUN` (until `max_attempts`) with appended feedback.
- In `llm`/`hybrid`, the supervisor can also `patch_task(...)` to clarify objectives or fix dependencies.

### D) Soft conditional branches

There is no dedicated “if/else” primitive, but the supervisor can emulate branching:

- Add multiple exploratory tasks in parallel.
- Cancel irrelevant branches with `cancel_task(task_id, reason="...")`.

---

## Supervisor feedback to sub-agents (per-task guidance)

`SupervisorDecision` may include feedback for specific tasks. The harness injects it into the task parameters:

- `_execute_ready_tasks()` stores it in `step.parameters["supervisor_feedback"]`
- `execute()` appends it to the worker prompt

Practical uses:

- Tightening output format (“return a bulleted summary + sources”)
- Addressing critic feedback (“fix missing criteria X”) without changing the DAG

---

## Practical guidance

- Prefer **fan-out/fan-in** for research-heavy objectives.
- Keep tasks small and QA-testable to converge quickly on retries.
- Use **hybrid** when you want a reliable backbone plus flexibility.
- Use **fixed** when you need strict control and reproducibility.

---

## Related code

- Supervisor tool gating: `DeepAgent._default_supervisor_tools()`
- Seed plan validation/loading: `DeepAgent._apply_seed_plan()`
- Dependency checks: `DeepAgent._dependencies_satisfied()`
- Concurrent execution: `DeepAgent._execute_ready_tasks()`
- QA transitions: `DeepAgent.handle_critic_result()`