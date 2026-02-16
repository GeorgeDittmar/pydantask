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
- **Researcher**: Searches for new information to fulfill plan steps

Agents may invoke tools or sub-agents recursively to accomplish their objectives.


## DeepAgent: The Orchestrator

`DeepAgent` is the high‑level orchestrator that uses several sub‑agents to turn a single objective into a multi‑step, tool‑using workflow.

```python
class DeepAgent:
    def __init__(
        self,
        prompt: str,
        model: str = "openai:gpt-4.1-mini",
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

## RuntimeState and Plan

### Plan and TaskItem

The **Planner** produces a `Plan` composed of multiple `TaskItem` objects. Conceptually:

- `Plan`
  - A collection of `TaskItem`s, each with:
    - `task_id`: unique integer ID
    - `task_objective`: the sub‑task’s description
    - `capability`: which sub‑agent/capability should execute the task
    - `task_dependencies`: list of other `task_id`s that must complete first
    - `status`: lifecycle state (e.g. PENDING, READY, RUNNING, NEEDS_REVIEW, COMPLETED, ERRORED)
    - `result`: a `TaskResult` once executed
    - `task_feedback`: a `TaskQAResult` from the Critic after QA

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
- Other runtime bookkeeping (e.g. `runtime_steps`, possibly logs or caches)

Agents that declare `deps_type=RuntimeState` receive this state via `RunContext[RuntimeState]` and can:

- Inspect the plan
- Read or update task statuses and results
- Know which capabilities are available

---

## CapabilityDescription and Sub‑Agents

Sub‑agents are represented in `DeepAgent.agent_registry` as `CapabilityDescription` entries:

```python
CapabilityDescription(
    name="synthesizer_agent",
    description="Generate answers based on information from various sources and sub agents.",
    tool_func=synthesizer_agent,
)
```

Each capability has:

- `name`: logical capability name, referenced in plans (`task.capability`)
- `description`: human‑readable explanation used by the Planner
- `tool_func`: the underlying `Agent` instance (or callable) that actually does the work

The Planner chooses capabilities based on the names and descriptions it’s given, and writes them into each `TaskItem.capability`.

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
    model=_model,
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
    model=_model,
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

The Critic writes its QA result back into `TaskItem.task_feedback`, which the Supervisor uses to advance task states.

---

### Supervisor

**Purpose**

Drive execution of the plan:

- Determine which tasks are READY to run
- Decide when tasks should move to REVIEW, COMPLETED, or ERRORED
- Indicate when all tasks are completed

**Implementation**

```python
self._supervisor_agent = supervisor_agent or self._create_agent_from_spec(
    agent_spec=SupervisorSpec(),
    name="_default_Supervisor_Agent",
    tools=[self.update_task_status, get_current_datetime, think_tool],
    output_type=SupervisorDecision,
    deps_type=RuntimeState,
    model=self.model,
)
```

Key points:

- Uses a spec (`SupervisorSpec`) to generate the system prompt dynamically based on `RuntimeState`
- `output_type=SupervisorDecision`, which includes:
  - `tasks_to_execute`: list of `task_id`s the supervisor wants to run next
  - `all_tasks_completed`: boolean
- Tools:
  - `update_task_status` — a tool that can mutate the `RuntimeState.plan[task_id].status`
  - `get_current_datetime`, `think_tool`

In each cycle, DeepAgent:

1. Calls Supervisor with current `RuntimeState`
2. Uses `SupervisorDecision.tasks_to_execute` to figure out which `TaskItem`s to run
3. Later, after QA, Supervisor can be called again to re‑evaluate statuses (e.g. mark tasks as COMPLETED, schedule dependents as READY)

---

### Producer

**Purpose**

The Producer is the main “worker” agent that creates the final answer or produces intermediate artifacts, usually interacting with the file system.

**Implementation**

```python
self._producer_agent = self._create_agent_from_spec(
    agent_spec=ProducerSpec(),
    name="_default_Producer_Agent",
    tools=[write_to_file_system, read_from_file_system, think_tool],
    output_type=output_type,      # default: TaskResult
    deps_type=RuntimeState,
    model=self.model,
)
```

Key points:

- Uses `ProducerSpec` for its system prompt
- Tools:
  - `write_to_file_system`
  - `read_from_file_system`
  - `think_tool`
- `output_type` is configurable (default `TaskResult`), allowing structured final outputs

Tasks that use the Producer capability typically:

- Generate content (code, documents, summaries)
- Store or read files
- Update the shared context for later tasks

---

### Researcher

**Purpose**

Gather external information (e.g. via web search) or perform research‑like tasks.

**Implementation**

```python
self._researcher_agent = researcher_agent or Agent(
    self.model,
    name="_default_Research_Agent",
    system_prompt=RESEARCH_AGENT_SYS_PROMPT,
    tools=[
        tavily_search_tool(api_key),
        think_tool,
        write_to_file_system,
        read_from_file_system,
        get_current_datetime,
    ],
    deps_type=RuntimeState,
    output_type=TaskResult,
    end_strategy="exhaustive",
)
```

Key points:

- Uses `tavily_search_tool` (requires `TAVILY_API_KEY`)
- Can read/write from the file system to store research notes or data
- Uses `RuntimeState` for context
- Returns `TaskResult` with structured research outcomes

Tasks with this capability are often used early in the plan to gather facts or context needed by later steps.

---