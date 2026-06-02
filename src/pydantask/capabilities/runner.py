from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Optional, Protocol, TypeVar

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantask.models.models import TaskRunDeps  # adjust import path if different

T = TypeVar("T")

@dataclass
class RunResult(Generic[T]):
    output: T

class CapabilityRunner(Protocol[T]):
    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]: ...

@dataclass
class AgentRunner(Generic[T]):
    agent: Agent

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]:
        r = await self.agent.run(prompt, deps=deps, usage_limits=usage_limits)
        return RunResult(output=r.output)

@dataclass
class AsyncFuncRunner(Generic[T]):
    func: Callable[[str, TaskRunDeps], Awaitable[T]]

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,  # kept for compatibility
    ) -> RunResult[T]:
        out = await self.func(prompt, deps)
        return RunResult(output=out)

@dataclass
class SyncFuncRunner(Generic[T]):
    func: Callable[[str, TaskRunDeps], T]

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]:
        out = await asyncio.to_thread(self.func, prompt, deps)
        return RunResult(output=out)

def as_runner(obj: Any) -> CapabilityRunner[Any]:
    """Centralized normalization (the only place you type-check)."""
    if isinstance(obj, Agent):
        return AgentRunner(obj)
    if callable(obj):
        # assume async callable(prompt, deps) by default
        return AsyncFuncRunner(obj)
    raise TypeError(f"Unsupported capability type: {type(obj)!r}")