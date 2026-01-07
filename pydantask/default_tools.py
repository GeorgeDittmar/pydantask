import os
import json

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent


async def call_worker(agent_registry: dict):
    """Call Agent worker to perform a task"""
