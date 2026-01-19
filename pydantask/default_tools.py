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
