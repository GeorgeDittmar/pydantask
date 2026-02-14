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
    """Create or write to a file on the file system.

    Used to offload context, or write files that may need to be consumed at a later time for execution.

    Args: file_name
        file_name: The name of the file to create or write to.
        content: The content to write into the file."""
    print(f"DEBUG: Context object: {ctx}")
    print(f"DEBUG: Deps object: {ctx.deps}")
    path = DEFAULT_DIR.joinpath(file_name)
    with open(path, "a") as f:
        f.write(content + "\n")
        ctx.deps.document_store[file_name] = str(
            path
        )  # save file name and path to document
        return f"Content written to {file_name}."


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
        if file_name not in ctx.deps.document_store:
            return f"File '{file_name}' not found in document store."
        path = ctx.deps.document_store[file_name]
        with open(path, "r") as f:
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
