# Tools and Sub-agents

A **tool** is any callable an agent may use. Tools may include:

- Simple functions (e.g., time, file IO, reflection)
- Sub-agents (agents with their own prompt, state access, and toolsets)

Tools are ordinary async Python functions (or `Agent` instances) that Pydantic AI can call via function calling. They usually accept a `RunContext[RuntimeState]` when they need access to the shared plan and document store.

---

## Core Built-in Tools (`pydantask.tools.default_tools`)

### `think_tool`

```python
from pydantask.tools.default_tools import think_tool
```

A lightweight reflection tool:

- **Signature**: `async def think_tool(reflection: str) -> str`
- **Purpose**: Let agents write down and “externalize” strategic thoughts.
- **When to use**:
  - After receiving search results (summarize findings and gaps)
  - Before deciding next steps (do I have enough to answer?)

It returns a simple confirmation string and does not modify state.

---

### File System and Document Tools

These tools work within a workspace directory (`pydantask/tools/tmp_files`) and also update `RuntimeState.document_store` so agents can find files by logical name.

#### `write_to_file_system`

```python
from pydantask.tools.default_tools import write_to_file_system
```

- **Signature**:
  - `async def write_to_file_system(ctx: RunContext[RuntimeState], file_name: str, content: str, overwrite: bool = False) -> str`
- **Behavior**:
  - Writes `content` to `tmp_files/<file_name>`.
  - Appends by default; set `overwrite=True` to replace the file.
  - Registers `file_name` in `ctx.deps.document_store` so it can be listed and read later.

#### `read_from_file_system`

```python
from pydantask.tools.default_tools import read_from_file_system
```

- **Signature**:
  - `async def read_from_file_system(ctx: RunContext[RuntimeState], file_name: str) -> str`
- **Behavior**:
  - Resolves `file_name` via `ctx.deps.document_store` (logical name → path) and reads the file from `tmp_files`.
  - Returns an informative message if the file does not exist.

#### `delete_from_file_system`

```python
from pydantask.tools.default_tools import delete_from_file_system
```

- **Signature**:
  - `async def delete_from_file_system(path: str) -> str`
- **Behavior**:
  - Attempts to delete `tmp_files/<path>`; ignores missing files.

#### `list_documents`

```python
from pydantask.tools.default_tools import list_documents
```

- **Signature**:
  - `async def list_documents(ctx: RunContext[RuntimeState]) -> str`
- **Behavior**:
  - Lists logical document names and their resolved filesystem paths from `document_store`.

---

### Task and Context Tools

These tools operate on `RuntimeState.plan` and task‑scoped context files.

#### `get_current_datetime`

```python
from pydantask.tools.default_tools import get_current_datetime
```

- **Signature**: `async def get_current_datetime() -> str`
- **Behavior**:
  - Returns the current timestamp as an ISO‑8601 string.
  - Used by Planner/Supervisor/Researcher for time‑aware reasoning.

#### `list_completed_tasks`

```python
from pydantask.tools.default_tools import list_completed_tasks
```

- **Signature**: `async def list_completed_tasks(ctx: RunContext[RuntimeState]) -> str`
- **Behavior**:
  - Walks `ctx.deps.plan` and returns a human‑readable list of tasks whose `status == TaskStatus.COMPLETED` with brief summaries.

#### `save_task_context`

```python
from pydantask.tools.default_tools import save_task_context
```

- **Signature**:
  - `async def save_task_context(ctx: RunContext[RuntimeState], task_id: int, content: str, kind: str = "notes", overwrite: bool = False) -> str`
- **Behavior**:
  - Writes `content` to a canonical file `task-{task_id}-{kind}.md` in `tmp_files` via `write_to_file_system`.
  - Useful for per‑task notes (`kind="notes"`), research (`"research"`), or final reports (`"final"`).

#### `read_task_context`

```python
from pydantask.tools.default_tools import read_task_context
```

- **Signature**:
  - `async def read_task_context(ctx: RunContext[RuntimeState], task_id: int, kind: str = "notes") -> str`
- **Behavior**:
  - Reads the canonical file `task-{task_id}-{kind}.md` via `read_from_file_system`.

#### `get_task_result`

```python
from pydantask.tools.default_tools import get_task_result
```

- **Signature**:
  - `async def get_task_result(ctx: RunContext[RuntimeState], task_id: int) -> str`
- **Behavior**:
  - Looks up `task_id` in `ctx.deps.plan` and returns the `TaskResult` as pretty‑printed JSON, or a diagnostic message if none exists.

#### `append_scratch_note`

```python
from pydantask.tools.default_tools import append_scratch_note
```

- **Signature**:
  - `async def append_scratch_note(ctx: RunContext[RuntimeState], task_id: int, note: str) -> str`
- **Behavior**:
  - Appends `note` to an in‑memory scratchpad entry `scratch_task_{task_id}` in `ctx.deps.document_store`.
  - Intended for lightweight, transient “running memory” during a task, not final reports.

---

### `ask_user` (optional interactive tool)

```python
from pydantask.tools.default_tools import ask_user
```

- **Signature**:
  - `async def ask_user(ctx: RunContext[RuntimeState], question_for_user: str) -> str`
- **Behavior**:
  - Prompts on stdin and returns the user’s answer.
  - This is synchronous/interactive and most useful in CLI environments.

---

## Registering Tools and Sub‑agents

Built‑in agents (Planner, Critic, Supervisor, Producer, Researcher, Worker) are constructed inside `DeepAgent` with appropriate tool lists (see `pydantask/agents/agent.py`). For example, the default Researcher has access to:

- `tavily_search_tool`
- `think_tool`
- `write_to_file_system`, `read_from_file_system`
- `save_task_context`, `read_task_context`
- `get_current_datetime`, `list_documents`

Additional tools or sub‑agents can be registered by creating `CapabilityDescription` instances and passing them via the `sub_agents` parameter to `DeepAgent.__init__`. Those capabilities then become available to the Planner and Supervisor via `RuntimeState.agent_registry`.
