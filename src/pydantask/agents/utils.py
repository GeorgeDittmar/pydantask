import yaml
from pathlib import Path
from typing import Any, Dict, List, Set

from pydantic import ValidationError

from pydantask.models import Plan, WorkflowYamlConfig


def get_incremented_path(
    base_name: str, extension: str, directory: Path = Path("checkpoints")
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    counter = 1
    # Initial path attempt: checkpoints/state_1.json
    target_path = directory / f"{base_name}_{counter}.{extension}"

    # Keep incrementing until we find a filename that doesn't exist
    while target_path.exists():
        counter += 1
        target_path = directory / f"{base_name}_{counter}.{extension}"

    return target_path


def _ensure_dag_is_valid(plan: Plan) -> None:
    """Validate DAG properties that Pydantic types alone don't enforce."""
    tasks = list(plan.tasks or [])

    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate task_id values in workflow: {dupes}")

    task_ids: Set[int] = set(ids)
    for t in tasks:
        for dep in t.sub_task_dependencies or []:
            if dep not in task_ids:
                raise ValueError(
                    f"Task {t.task_id} depends on missing task_id {dep}. "
                    "All dependencies must reference an existing task_id in the workflow."
                )
            if dep == t.task_id:
                raise ValueError(f"Task {t.task_id} cannot depend on itself")

    # Cycle detection via DFS.
    graph: Dict[int, List[int]] = {
        t.task_id: list(t.sub_task_dependencies or []) for t in tasks
    }

    visiting: Set[int] = set()
    visited: Set[int] = set()

    def dfs(node: int, stack: List[int]) -> None:
        if node in visited:
            return
        if node in visiting:
            # found a cycle; report a helpful path
            cycle_start = stack.index(node) if node in stack else 0
            cycle_path = stack[cycle_start:] + [node]
            raise ValueError(
                f"Dependency cycle detected: {' -> '.join(map(str, cycle_path))}"
            )

        visiting.add(node)
        stack.append(node)
        for dep in graph.get(node, []):
            dfs(dep, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for tid in graph.keys():
        dfs(tid, [])


async def import_yaml_workflow(
    path: str | Path, *, auto_mark_final: bool = True
) -> Plan:
    """Load a pre-defined workflow (task DAG) from a YAML file and validate it.

    Returns:
        Plan: A validated `Plan` suitable to pass as `seed_plan=...` to `DeepAgent`.

    Notes:
        - This is intended for user-provided *seed plans* (pre-defined DAGs).
        - Tasks default to `status=pending` so the deterministic scheduler can
          promote them to READY when dependencies are satisfied.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Validate against the strict user-facing YAML schema and convert into the
    # canonical Plan/TaskItem models.
    try:
        cfg = WorkflowYamlConfig.model_validate(raw)
        plan = cfg.to_plan()
    except ValidationError as e:
        raise ValueError(
            f"Invalid workflow YAML at {str(path)!r}.\n\nPydantic validation error:\n{str(e)}"
        ) from e

    _ensure_dag_is_valid(plan)

    # Enforce/assist with the 'final task' invariant expected by DeepAgent's completion guardrail.
    final_tasks = [t for t in plan.tasks if getattr(t, "is_final", False)]
    if len(final_tasks) > 1:
        raise ValueError(
            f"Workflow YAML marks multiple tasks as final: {[t.task_id for t in final_tasks]}. "
            "Mark exactly one task with `is_final: true`."
        )

    if len(final_tasks) == 0 and auto_mark_final:
        # Deterministic fallback: mark the max task_id as final.
        last = max(plan.tasks, key=lambda t: t.task_id)
        last.is_final = True

    return plan

# from asyncio import tasks
import json
import os
import threading
import asyncio
import inspect
from httpx import AsyncClient, HTTPStatusError
from tenacity import (
    wait_exponential_jitter,
    wait_exponential,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

import uuid
from collections import Counter

from loguru import logger

from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import List, Optional, Literal, Any, Dict, Callable, Union, Type
from datetime import datetime
from asyncio import TaskGroup
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.usage import UsageLimits

from pydantask.capabilities.runner import as_runner
from pathlib import Path
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from pydantask.prompts.prompts_v2 import (
    CRITIC_SYS_PROMPT,
    PRODUCER_SYS_PROMPT,
    RESEARCH_AGENT_SYS_PROMPT,
    SUPERVISOR_INPUT_PROMPT,
    WORKER_AGENT_SYS_PROMPT,
    DYNAMIC_SUPERVISOR_SYS_PROMPT,
    BOOTSTRAP_INSTURCT,
    ORCHESTRATION_INSTRUCT,
)

from pydantask.models import (
    RuntimeState,
    TaskItem,
    Plan,
    TaskQAResult,
    TaskStatus,
    SupervisorDecision,
    CapabilityDescription,
    TaskResult,
    DeepAgentRunResult,
    TaskRunDeps,
    TracingBackend,
)

# Default tool wiring is intentionally in-memory focused.
# Filesystem tools still exist in `pydantask.tools.default_tools` but are not enabled by default.
from pydantask.tools.default_tools import (
    append_scratch_note,
    fetch_url_content,
    get_current_datetime,
    get_task_result,
    list_completed_tasks,
    read_scratch_notes,
    think_tool,
)

from pydantask.observe.tracing import (
    traced,
    init_tracing_backend,
    autodetect_tracing_backend,
    flush_tracing,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

EVENT_RESULT_DETAIL_TRUNCATION = 4_000
# When a task result is too large to keep inline in the event log, we persist
# the full JSON payload under the checkpoint directory and store only a pointer
# (plus a truncated preview) in events.jsonl.
TASK_RESULT_ARTIFACT_DIRNAME = "task_results"

# Consult runs are intended to be quick and cheap.
CONSULT_TOTAL_TOKENS_LIMIT = 1_200

CheckpointEventType = Literal[
    "task_added",
    "task_patched",
    "task_status_updated",
    "task_result",
    "task_metadata_appended",
    "scratch_note_appended",
    "supervisor_decision",
    "critic_feedback",
    "final_task_set",
]


class CheckpointEvent(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.utcnow())
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: CheckpointEventType
    payload: Dict[str, Any] = Field(default_factory=dict)


class CheckpointRecorder:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory / "events.jsonl"
        self.summary_path = directory / "summaries.jsonl"
        self._lock = threading.Lock()

    def record(self, event_type: CheckpointEventType, payload: Dict[str, Any]) -> None:
        event = CheckpointEvent(type=event_type, payload=payload)
        self._append_json_line(self.log_path, event.model_dump_json())

    def record_summary(self, summary: Dict[str, Any]) -> None:
        self._append_json_line(self.summary_path, json.dumps(summary))

    def load_events(self) -> list[CheckpointEvent]:
        if not self.log_path.exists():
            return []
        events: list[CheckpointEvent] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                events.append(CheckpointEvent.model_validate_json(line))
        return events

    def _append_json_line(self, path: Path, json_line: str) -> None:
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json_line + "\n")