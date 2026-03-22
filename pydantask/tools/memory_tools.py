import os
import json

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pathlib import Path

from pydantask.models import RuntimeState, KnowledgeRecord


async def list_knowledge(ctx: RunContext[RuntimeState]) -> list[KnowledgeRecord]:
    """Return all known knowledge records (IDs + summaries + metadata)."""
    return list(ctx.deps.knowledge_store.values())


async def get_knowledge(ctx: RunContext[RuntimeState], knowledge_id: str) -> dict:
    """
    Return:
      - the KnowledgeRecord, and
      - file contents if path is present and usable.
    """
    record = ctx.deps.knowledge_store.get(knowledge_id)
    if record is None:
        return {"error": f"no knowledge with id={knowledge_id}"}

    content = None
    if record.path is not None:
        # delegate to read_from_file_system tool or inline file read
        content = await read_from_file_system(ctx, record.path)

    return {
        "record": record.model_dump(),
        "content": content,
    }
