import os
import json

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pathlib import Path

DEFAULT_DIR = Path("tmp_files/")


async def call_worker(agent_registry: dict):
    """Call Agent worker to perform a task"""
    pass


async def think_tool(reflection: str):
    """Tool for strategic reflection on progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

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
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


async def write_to_file_system(file_to_write_to, information):
    "Create or write to a file on the filesystem."
    with open(DEFAULT_DIR.joinpath(file_to_write_to), "a") as f:
        f.write(information + "\n")


async def delete_from_file_system(path_to_delete):
    """Delete a file or directory"""
    # Attempt to delete the file, ignore error if it doesn't exist (Python 3.8+)
    try:
        DEFAULT_DIR.joinpath(path_to_delete).unlink(missing_ok=True)
        print(f"File '{path_to_delete}' deletion attempted (if existed).")
    except PermissionError:
        print(f"Permission denied to delete the file '{path_to_delete}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


async def read_from_file_system(file_to_read):
    """Read from file on file system if it exists."""
    try:
        with open(file_to_read, "r") as f:
            return f.read()
    except FileNotFoundError as e:
        return "File does not exist. If you were expencting it to be, create the file."


# Tools used by the supervisor or planner agents
async def get_current_system_time() -> str:
    """Use this to get the current system time for contex on a task."""
    from datetime import datetime

    return datetime.now().isoformat()
