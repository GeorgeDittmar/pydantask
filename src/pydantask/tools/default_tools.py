import os
import json

import asyncio
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pathlib import Path

from pydantask.models import RuntimeState, TaskRunDeps
from pydantask.models.models import TaskItem

BASE_DIR = Path(__file__).parent.resolve()  # Directory where this script is
DEFAULT_DIR = BASE_DIR / "tmp_files"  # TODO: make this configurable
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)


def _truncate_text(text: str, max_chars: int | None) -> str:
    """Best-effort truncation helper to reduce tool output size.

    This is primarily used to avoid blowing up model context windows (common with
    smaller local models).

    If truncation occurs, we keep both the head and tail of the text.
    """
    if max_chars is None:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + f"\n\n...[TRUNCATED {len(text) - max_chars} chars; original_len={len(text)}]...\n\n"
        + text[-tail_chars:]
    )


async def ask_user(ctx: RunContext[RuntimeState], question_for_user: str) -> str:
    """Prompt the user for input. This is a synchronous blocking call."""
    return input(f"{question_for_user}: ")


async def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on progress and decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: str Your detailed reflection on research progress, findings, gaps, and next steps
    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


async def write_to_file_system(
    ctx: RunContext[RuntimeState],
    file_name: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """
    Write content to a file in the agent's workspace file system.
    write_to_file_system: Tool for saving content to a file in the agent's workspace.

    IMPORTANT:
        - `file_name` must be the logical name used when writing
        (e.g. 'agent_frameworks_survey_task2.md'), *not* the full filesystem path.
        - This function appends to the file by default. If you want to overwrite, set overwrite=True.
    Args:
        file_name: Logical file name key used in write_to_file_system.
        content: The string content to write to the file.
        overwrite: If True, overwrite the file instead of appending.
    Returns:
        Confirmation message indicating where the content was written and how to read it back.
    """
    # Ensure the base directory exists
    # At top:

    path = DEFAULT_DIR / file_name
    mode = "w" if overwrite else "a"
    logger.info(f"Writing to file system at {path} with content: {content}")
    with open(path, mode, encoding="utf-8") as f:
        f.write(content + "\n")

    # Store by the logical file_name key so agents can read it back with that name
    ctx.deps.document_store[file_name] = file_name

    return (
        f"Content written to {path}.\n"
        f"To read this file later, call read_from_file_system with file_name='{file_name}'. "
        f"Use exactly this file_name, not the full path."
    )


async def delete_from_file_system(path: str) -> str:
    """Delete a file or directory on the file system.

    Args: ""
        path: str = The path to the file or directory to delete.
    Returns:
        Confirmation message indicating deletion attempt.
    """
    # Attempt to delete the file, ignore error if it doesn't exist (Python 3.8+)
    try:
        DEFAULT_DIR.joinpath(path).unlink(missing_ok=True)
        return f"File '{path}' deletion attempted (if existed)."
    except PermissionError:
        return f"Permission denied to delete the file '{path}'."
    except Exception as e:
        return f"An error occurred: {e}"


async def read_from_file_system(
    ctx: RunContext[RuntimeState],
    file_name: str,
) -> str:
    """
    Read from a file on the file system. If the file dos not exist, returns a message indicating so.

    Args:
        file_name: = str The path to the file to read.
    Returns:
        String of file contents
    """
    try:
        # First try lookup in doc store, fallback to the file_name
        logical_file = ctx.deps.document_store.get(file_name, file_name)
        path = DEFAULT_DIR / logical_file
        logger.info(
            f"Attempting to read file with logical name '{file_name}' from document store. Found path: {path}"
        )

        full_path = Path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    except FileNotFoundError as e:
        existing = ", ".join(ctx.deps.document_store.keys()) or "<none>"
        return (
            f"File '{file_name}' not found at path '{path}'.\n"
            f"Known document keys: {existing}\n"
            f"If you expected this to exist, you must first write it using write_to_file_system."
        )


# Tools used by the supervisor or researcher mostly
async def get_current_datetime() -> str:
    """
    Get the current date and time in ISO 8601 format.

    When to use:
        - Needing to look up current date and time for context to complete a task
        - Needing to timestamp a step or task
        - Comparing results between different times

    Args: None
    Returns:
        Current date and time as an ISO 8601 formatted string.
    """
    from datetime import datetime

    return str(datetime.now().isoformat())


async def list_documents(ctx: RunContext[RuntimeState]) -> str:
    """List all documents that have been written in this run.

    Returns a human-readable list of logical names and their filesystem paths.
    """
    if not ctx.deps.document_store:
        return "No documents have been written yet."

    lines: list[str] = []
    for name, logical in ctx.deps.document_store.items():
        path = DEFAULT_DIR / logical  # Convert logical name to full path for display
        lines.append(f"- name: {name}\n  path: {path}")
    return "\n".join(lines)


async def list_completed_tasks(ctx: RunContext[TaskRunDeps]) -> str:
    """List all tasks that have completed, with brief summaries.

    Useful for agents that need to review prior work before deciding next steps.
    """
    if not ctx.deps.runtime_state.plan:
        return "No tasks in plan."

    from pydantask.models import TaskStatus  # local import to avoid cycles

    lines: list[str] = []
    for task_id, task in sorted(
        ctx.deps.runtime_state.plan.items(), key=lambda kv: kv[0]
    ):
        if task.status != TaskStatus.COMPLETED:
            continue
        summary = task.result.summary if task.result is not None else "<no result>"
        lines.append(
            f"- task_id: {task_id}\n"
            f"  capability: {task.capability}\n"
            f"  objective: {task.sub_task_objective}\n"
            f"  summary: {summary}"
        )

    if not lines:
        return "No completed tasks yet."

    return "\n".join(lines)


async def save_task_context(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    content: str,
    kind: str = "notes",
    overwrite: bool = False,
) -> str:
    """Save contextual information for a specific task to a canonical file.

    FILENAME CONVENTION (IMPORTANT):
      - The file name will always be ``task-{task_id}-{kind}.md``.
      - Agents should NOT invent their own names when saving task context.

    When to use:
      - After completing a task, to save detailed notes or results for that task.
      - When you want to offload information from memory but keep it accessible by task_id.
      - To create a record of your thought process and findings for each task.

    Args:
        task_id: The ID of the task whose context you want to save.
        content: The string content to save, such as notes, findings, or detailed results.
        kind: A short label like "notes", "research", or "final"; it becomes part of the file name.
        overwrite: If True, overwrite the file instead of appending. Default is False (append).

    Returns:
    Confirmation message indicating where the content was saved and how to read it back.
    """
    file_name = f"task-{task_id}-{kind}.md"
    return await write_to_file_system(
        ctx, file_name=file_name, content=content, overwrite=overwrite
    )


async def read_task_context(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    kind: str = "notes",
) -> str:
    """Read contextual information for a specific task from its canonical file.

    This uses the same convention as ``save_task_context``:
      - file_name = ``task-{task_id}-{kind}.md``.

    Args:
        task_id: The task id whose context file you want to read.
        kind: The same kind string that was used when saving (e.g. "notes", "research", "final").

    Returns:
        File contents as a string, or an informative error message if it does not exist.
    """
    file_name = f"task-{task_id}-{kind}.md"
    return await read_from_file_system(ctx, file_name=file_name)


async def get_task_result(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    max_chars: int | None = 12_000,
) -> str:
    """Return the full TaskResult for a given task_id as JSON.

    When to use:
        - When you need to inspect the detailed output of a prior task.
        - Before synthesizing or critiquing based on earlier work.
    """

    # check if task_id if valid. if not then return saying incorrect id

    if task_id not in ctx.deps.runtime_state.plan:
        return f"CRITICAL: task id: {task_id} does not exist in the plan. Attempt call again and double check the task id before trying again."

    task = ctx.deps.runtime_state.plan.get(task_id, None)

    if task is None:
        return f"CRITICAL: No task with id {task_id}."

    if task.result is None:
        return f"Task {task_id} has no result yet. Current status: {task.status}."

    result_json = task.result.model_dump_json(indent=2)
    return _truncate_text(result_json, max_chars=max_chars)


async def append_scratch_note(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    note: str,
) -> str:
    """
    Tool: Append Scratch Note
    Description: Append a short note to the tasks notepad / memory

    When to use:
        - You want to store intermediate notes or thoughts about your work and any.
    When not to use:
        - Writing final full reports / analysis or answers.
    """
    key = "scratch_notes"
    existing = ctx.deps.task.metadata.get(key, "")
    ctx.deps.task.metadata[key] = existing + f"\n\n{note}"

    recorder = getattr(ctx.deps.runtime_state, "checkpoint_recorder", None)
    if recorder is not None:
        recorder.record(
            "scratch_note_appended",
            {"task_id": task_id, "note": _truncate_text(note, max_chars=1_000)},
        )

    return f"Appended note to scratchpad {key}"


async def read_scratch_notes(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    max_chars: int | None = 8_000,
):
    """
    Tool: Read Scratch Notes
    Description: Reads any notes in the in-memory scratchpad for this task. Use to see if there are any thoughts you need to reason over.
    Pass in the task_id you are working on to view all the notes.
    """

    key = f"scratch_notes"
    existing = ctx.deps.task.metadata.get(key, "")
    return _truncate_text(existing, max_chars=max_chars)
