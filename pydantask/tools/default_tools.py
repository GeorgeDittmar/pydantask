import os
import json

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pathlib import Path

from pydantask.models import RuntimeState

DEFAULT_DIR = Path("tmp_files/")


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
    ctx: RunContext[RuntimeState], file_name: str, content: str
) -> str:
    """
    Read from a file in the document store.

    IMPORTANT:
      - file_name must be the logical name used when writing
        (e.g. 'agent_frameworks_survey_task2.md'), *not* the full filesystem path.

    Args:
        file_name: Logical file name key used in write_to_file_system.
    """
    # Ensure the base directory exists
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    path = os.path.join(cwd, DEFAULT_DIR, file_name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content + "\n")

    # Store by the logical file_name key so agents can read it back with that name
    ctx.deps.document_store[file_name] = str(path)

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
        stored_path = ctx.deps.document_store.get(file_name)
        if stored_path is None:
            return f"File '{file_name}' not found in document store."
        full_path = Path(stored_path)
        with open(full_path, "r") as f:
            return f.read()

    except FileNotFoundError as e:
        return f"File does not exist. If you were expencting it to be, create the file. \n{e}"


# Tools used by the supervisor or planner agents
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
    for name, path in ctx.deps.document_store.items():
        lines.append(f"- name: {name}\n  path: {path}")
    return "\n".join(lines)


async def list_completed_tasks(ctx: RunContext[RuntimeState]) -> str:
    """List all tasks that have completed, with brief summaries.

    Useful for agents that need to review prior work before deciding next steps.
    """
    if not ctx.deps.plan:
        return "No tasks in plan."

    from pydantask.models import TaskStatus  # local import to avoid cycles

    lines: list[str] = []
    for task_id, task in sorted(ctx.deps.plan.items(), key=lambda kv: kv[0]):
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


async def get_task_result(ctx: RunContext[RuntimeState], task_id: int) -> str:
    """Return the full TaskResult for a given task_id as JSON.

    When to use:
        - When you need to inspect the detailed output of a prior task.
        - Before synthesizing or critiquing based on earlier work.
    """
    task = ctx.deps.plan.get(task_id)
    if task is None:
        return f"No task with id {task_id}."

    if task.result is None:
        return f"Task {task_id} has no result yet. Current status: {task.status}."

    return task.result.model_dump_json(indent=2)
