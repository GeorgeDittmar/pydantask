# Agent Concepts

Agents are the core entities in Pydantask.

At a high level, agents:

- Accept a prompt or task (an objective or sub‑objective)
- Access a set of tools (simple functions or other agents)
- Can operate with access to persistent shared runtime state
- Coordinate via a shared **plan** and **runtime state** to solve complex goals

This document explains the main agent concepts, how they work together in `DeepAgent`, and how to extend them.

## Built-In Agents

- **Planner**: Breaks down the main objective into discrete tasks
- **Critic**: Evaluates outputs of tasks/subtasks for quality
- **Supervisor**: Oversees execution, advancing task states
- **Producer**: Synthesizes intermediate results into final answers or artifacts
- **Researcher**: Searches for new information to fulfill plan steps

Agents may invoke tools or sub-agents recursively to accomplish their objectives.


## DeepAgent: The Orchestrator

`DeepAgent` is the high‑level orchestrator that uses several sub‑agents to turn a single objective into a multi‑step, tool‑using workflow.

```python
class DeepAgent:
    def __init__(
        self,
        prompt: str,
        model: str | Model = "gpt-4.1-mini",
        critic_agent: Optional[Agent] = None,
        planner_agent: Optional[Agent] = None,
        supervisor_agent: Optional[Agent] = None,
        researcher_agent: Optional[Agent] = None,
        max_steps: int = 20,
        set_token_budget: Union[int, None] = None,
        sub_agents: Union[None, list[CapabilityDescription]] = None,
        human_feedback: bool = False,
        trace: bool = False,
        output_type: Type = TaskResult,
    ):
        ...
```

Internally, `DeepAgent` creates a retrying OpenAI chat model (`OpenAIChatModel`) with an `AsyncTenacityTransport` so that all sub‑agents share the same robust HTTP client and retry behavior.

DeepAgent’s responsibilities:

- Accept a **single objective** (`prompt`)
- Ask the **Planner** to create a structured **Plan** (list of `TaskItem`s)
- Initialize a shared **RuntimeState** containing:
  - The plan (all tasks)
  - The overall objective
  - The available capabilities (sub‑agents)
- Repeatedly:
  - Ask the **Supervisor** what to execute next
  - Execute ready tasks via **sub‑agents** (capabilities)
  - Ask the **Critic** to quality‑check each result
  - Apply deterministic transition logic (`handle_critic_result`) to update task status and retry/failed state
  - Update the plan and runtime state accordingly
- Stop when the Supervisor determines all tasks are completed or `max_steps` is reached

The main entry point is:

```python
@observe
async def run(self) -> RuntimeState:
    ...
```

`run()` returns the final `RuntimeState`, which includes:

- The executed plan
- Task results and QA feedback
- Any file‑system or research artifacts referenced in `RuntimeState`

---

## Task Lifecycle and Control Loop

At a high level, each `TaskItem` goes through a deterministic lifecycle managed by `DeepAgent.run`:

1. **Planning**
   - Planner produces a `Plan` (list of `TaskItem`s) with initial statuses (typically `PENDING`).

2. **Supervision cycle** (repeated until `max_steps` or completion):
   - Supervisor receives a formatted “status board” view of the plan and current `RuntimeState`.
   - Supervisor returns a `SupervisorDecision` with:
     - `tasks_to_execute`: which `task_id`s should run now.
     - `feedback_to_subagents` (optional per‑task hints/instructions).
     - `all_tasks_completed`: `True` if no more useful work can be done.

3. **Execution**
   - `_execute_ready_tasks` filters `tasks_to_execute` based on dependency satisfaction (only tasks whose `sub_task_dependencies` are all `COMPLETED` are allowed to run).
   - Each ready `TaskItem` is set to `RUNNING`, and `execute(...)` is called with the appropriate sub‑agent (capability) from `agent_registry`.
   - The sub‑agent (`Agent`) runs once and returns an `output` (usually a `TaskResult`).
   - `execute(...)` stores this in `task.result` and sets `task.status = NEEDS_REVIEW`.

4. **Critic / QA**
   - For each executed task, `DeepAgent` calls the Critic with `_format_critic_input_prompt`, including:
     - The overall objective
     - The `TaskItem` definition
     - The worker’s `TaskResult`
     - Any relevant documents from `RuntimeState.document_store`
   - The Critic returns a `TaskQAResult` (`task_id`, `passed`, `reasoning`).
   - `handle_critic_result(task, review)` is then invoked to update the `TaskItem` deterministically:
     - If `review.passed` is `True`:
       - `task.status = COMPLETED`.
     - If `review.passed` is `False` and `task.attempt_count < task.max_attempts`:
       - `task.attempt_count += 1`.
       - `task.status` is set to a retryable state (e.g. `READY`/`RERUN`).
       - Critic reasoning is injected into the task context (e.g. appended to `sub_task_objective` and/or stored in `task.parameters["critic_feedback"]`).
     - If `review.passed` is `False` and max attempts are reached:
       - `task.status = FAILED`.
       - `task.error_msg` is populated with critic feedback.

5. **Next supervision cycle**
   - Because `RuntimeState.plan` holds references to the same `TaskItem` objects, any changes made by `handle_critic_result` are visible to the Supervisor on the next cycle.
   - Supervisor can then:
     - Schedule retryable tasks (`READY`/`RERUN`) whose dependencies are satisfied.
     - Avoid `COMPLETED`/`FAILED` tasks.
     - Decide to end the run by setting `all_tasks_completed = True` when all tasks are either `COMPLETED` or terminal (`FAILED`/`ERRORED`).

This design keeps the **control logic deterministic** in Python (status transitions, retries) while still letting LLM agents make the substantive decisions (planning, task execution, quality assessment).

---

## RuntimeState and Plan

### Plan and TaskItem

The **Planner** produces a `Plan` composed of multiple `TaskItem` objects.

In code, `Plan` is:

- `Plan`
  - `reasoning_steps`: the planner’s internal reasoning before finalizing the plan
  - `tasks`: a list of `TaskItem` instances

Each `TaskItem` (see `pydantask/models/models.py`) includes, among other fields:

- `task_id`: unique integer ID
- `overall_objective`: overall objective this task contributes to
- `sub_task_objective`: description of the specific sub‑task
- `capability`: which sub‑agent/capability should execute the task (e.g. `"research_agent"`, `"producer_agent"`)
- `sub_task_dependencies`: list of other `task_id`s that must complete first
- `status`: lifecycle state, one of `TaskStatus` (e.g. `PENDING`, `READY`, `RUNNING`, `NEEDS_REVIEW`, `COMPLETED`, `FAILED`, `ERRORED`, `RERUN`)
- `result`: a `TaskResult` once executed
- `task_feedback`: a `TaskQAResult` from the Critic after QA (if stored)
- `attempt_count` / `max_attempts`: for retry logic
- `metadata`: optional free‑form metadata

The Planner is given:

- The **overall goal**
- The list of **available capabilities** (see below)
- Time context (current datetime, current year)

and returns a structured `Plan` with task IDs and dependencies.

### RuntimeState

`RuntimeState` is the shared, mutable state passed to most agents:

```python
RuntimeState(
    plan=agent_plan_map,
    objective=self.prompt,
    agent_registry=self.agent_registry,
)
```

It contains (conceptually):

- `objective`: the overall user goal
- `plan`: `Dict[int, TaskItem]` — the entire plan indexed by `task_id`
- `agent_registry`: mapping from capability name to `CapabilityDescription`
- `runtime_steps`: how many DeepAgent cycles have been run
- `document_store`: in‑memory document cache / scratch storage
- `knowledge_store`: optional higher‑level knowledge records

Agents that declare `deps_type=RuntimeState` receive this state via `RunContext[RuntimeState]` and can:

- Inspect the plan
- Read or update task statuses and results
- Know which capabilities are available
- Access shared documents and context

---

## CapabilityDescription and Sub‑Agents

Sub‑agents are represented in `DeepAgent.agent_registry` as `CapabilityDescription` entries:

```python
CapabilityDescription(
    name="producer_agent",
    description="Produces output based on information from various sources and sub agents.",
    tool_func=producer_agent,
)
```

Each capability has:

- `name`: logical capability name, referenced in plans (`task.capability`)
- `description`: human‑readable explanation used by the Planner
- `tool_func`: the underlying `Agent` instance (or callable) that actually does the work

By default, `_setup_default_sub_agents` registers:

- `producer_agent`: for synthesizing and writing reports
- `research_agent`: for web/external research

Additional capabilities can be supplied via the `sub_agents` parameter to `DeepAgent.__init__`.

At runtime, `DeepAgent` looks up `agent_registry[step.capability].tool_func` to execute the task.

---

## Built-In Top-Level Agents

DeepAgent uses several **top‑level agents**, each with a specific role.

### Planner

**Purpose**

Break down the main objective into discrete, ordered tasks (`TaskItem`s) and return a `Plan`.

**Implementation**

```python
self._planner_agent = planner_agent or Agent(
    name="_default_Planner_Agent",
    model=self._retry_model,
    system_prompt=PLANNER_SYS_PROMPT,
    output_type=Plan,
    tools=[think_tool],
    end_strategy="exhaustive",
)
```

Key points:

- `output_type=Plan` — must return a structured `Plan`
- Uses `think_tool` as an internal reasoning tool
- Receives:
  - The overall goal
  - A formatted list of **available capabilities**
  - Time context (current datetime and CURRENT_YEAR)
- Must embed time information into the plan metadata where relevant

The Planner’s output is mapped to `Dict[int, TaskItem]` and stored in `RuntimeState.plan`.

---

### Critic

**Purpose**

Evaluate the quality of each task’s result and decide if it sufficiently completes the specific sub‑task.

**Implementation**

```python
self._critic_agent = critic_agent or Agent(
    model=self._retry_model,
    name="_default_Critic_Agent",
    system_prompt=CRITIC_SYS_PROMPT,
    output_type=TaskQAResult,
    deps_type=RuntimeState,
    tools=[read_from_file_system, get_current_datetime, think_tool],
    end_strategy="exhaustive",
)
```

Key points:

- `output_type=TaskQAResult` — must return:
  - `task_id`
  - `passed` (bool)
  - `reasoning` (detailed explanation)
- Has access to:
  - `RuntimeState` via `deps_type=RuntimeState`
  - File system (read only for QA) and time tools
- Is called with:
  - The overall objective
  - The specific `TaskItem` definition
  - The worker’s `TaskResult` for that task

After each worker execution, `DeepAgent.run` calls the Critic with a formatted prompt (via `_format_critic_input_prompt`) and then passes the resulting `TaskQAResult` into `handle_critic_result`, which updates the corresponding `TaskItem` (status, retry counters, and injected feedback).

---

### Supervisor

**Purpose**

Drive execution of the plan:

- Determine which tasks are READY to run (respecting dependencies)
- Optionally inspect QA feedback and adjust statuses
- Indicate when all tasks are completed or the plan cannot be progressed

**Implementation**

```python
self._supervisor_agent = supervisor_agent or self._create_agent_from_spec(
    agent_spec=SupervisorSpec(),
    name="_default_Supervisor_Agent",
    tools=[
        self.update_task_status,
        get_current_datetime,
        think_tool,
        self.view_qa_report,
    ],
    output_type=SupervisorDecision,
    deps_type=RuntimeState,
    model=self._retry_model,
)
```

Key points:

- Uses a spec (`SupervisorSpec`) to generate the system prompt dynamically based on `RuntimeState`.
- `output_type=SupervisorDecision`, which includes:
  - `tasks_to_execute`: list of `task_id`s the supervisor wants to run next
  - `feedback_to_subagents`: optional per‑task feedback strings
  - `all_tasks_completed`: boolean flag indicating whether the plan is done or cannot progress further
- Tools:
  - `update_task_status` — a tool that can mutate `RuntimeState.plan[task_id].status`
  - `view_qa_report` — to inspect critic feedback for a specific task
  - `get_current_datetime`, `think_tool`

In each cycle, DeepAgent:

1. Builds a “status board” string from `RuntimeState` via `_format_supervisor_input_prompt`.
2. Calls the Supervisor with that string and `deps=runtime_state`.
3. Uses `SupervisorDecision.tasks_to_execute` to figure out which `TaskItem`s to run.
4. Passes any `feedback_to_subagents` down into `TaskItem.parameters["supervisor_feedback"]` before execution.

---

### Producer

**Purpose**

The Producer is the main “synthesis” agent that:

- Produces the final answer
- Or generates intermediate artifacts based on existing task results and files

**Implementation**

```python
self._producer_agent = self._create_agent_from_spec(
    agent_spec=ProducerSpec(),
    name="_default_Producer_Agent",
    tools=[
        # Core FS and context tools
        write_to_file_system,
        read_from_file_system,
        save_task_context,
        read_task_context,
        # Reasoning / time / plan-inspection tools
        think_tool,
        get_current_datetime,
        list_documents,
        list_completed_tasks,
        get_task_result,
    ],
    output_type=output_type,  # default: TaskResult
    deps_type=RuntimeState,
    model=self._retry_model,
)
```

Key points:

- Uses `ProducerSpec` for its system prompt.
- Tools:
  - File‑system and context tools (`write_to_file_system`, `read_from_file_system`, `save_task_context`, `read_task_context`).
  - Plan/context introspection tools (`list_documents`, `list_completed_tasks`, `get_task_result`).
  - Reasoning and time tools (`think_tool`, `get_current_datetime`).
- `output_type` is configurable (default `TaskResult`), allowing structured final outputs or custom result types.

Tasks that use the Producer capability typically:

- Generate the final user‑visible answer
- Synthesize outputs from completed research/worker tasks
- Write canonical reports to disk and reference them in `TaskResult.detailed_report_paths`

DeepAgent uses `_build_producer_prompt` to summarize all completed sub‑tasks (including summaries, report paths, and sources) and passes that summary into the Producer when executing the final synthesis task.

---

### Researcher

**Purpose**

Gather external information (e.g. via web search) or perform research‑like tasks.

**Implementation**

```python
self._researcher_agent = researcher_agent or Agent(
    model=self._retry_model,
    name="_default_Research_Agent",  # Use a cheap model for simple tasks
    system_prompt=RESEARCH_AGENT_SYS_PROMPT,
    tools=[
        tavily_search_tool(api_key),
        think_tool,
        # File-system and context tools; prefer save_task_context for reports
        write_to_file_system,
        read_from_file_system,
        save_task_context,
        read_task_context,
        get_current_datetime,
        list_documents,
    ],
    deps_type=RuntimeState,
    output_type=TaskResult,
    end_strategy="exhaustive",
)
```

Key points:

- Uses `tavily_search_tool` (requires `TAVILY_API_KEY`) for web research.
- Can read/write from the file system and task context to store research notes or long‑form reports.
- Uses `RuntimeState` for access to the shared plan and document store.
- Returns `TaskResult` with structured research outcomes (summary, notes, report paths, etc.).

Tasks with this capability are often used early in the plan to gather facts or context needed by later steps.

---