# Supervisor Workflows & DAG Modes

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
- The supervisor mainly decides **what to run next**, and how to **extend or patch** the DAG.

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

NOTE: instructions given to the agent have to make it clear that a conditional is needed

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

---

## Related code

- Supervisor tool gating: `DeepAgent._default_supervisor_tools()`
- Dependency checks: `DeepAgent._dependencies_satisfied()`
- Concurrent execution: `DeepAgent._execute_ready_tasks()`
- QA transitions: `DeepAgent.handle_critic_result()`