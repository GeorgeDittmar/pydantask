from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pydantask.models import ArtifactRef, TaskItem, TaskResult, TaskStatus


def ok_result(
    task: TaskItem,
    *,
    summary: str,
    detailed_output: str | None = None,
    data: dict[str, Any] | None = None,
    artifacts: list[ArtifactRef] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskResult:
    """Build a successful TaskResult for deterministic/callable capabilities."""
    return TaskResult(
        task_id=task.task_id,
        status=TaskStatus.COMPLETED,
        summary=summary,
        detailed_output=detailed_output or "",
        data=data or {},
        artifacts=artifacts or [],
        error_msg=None,
        metadata=metadata or {},
    )


def error_result(
    task: TaskItem,
    *,
    error_msg: str,
    summary: str | None = None,
    detailed_output: str | None = None,
    data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskResult:
    """Build a failed/errored TaskResult for deterministic/callable capabilities."""
    return TaskResult(
        task_id=task.task_id,
        status=TaskStatus.ERRORED,
        summary=summary or "Capability errored.",
        detailed_output=detailed_output or "",
        data=data or {},
        error_msg=error_msg,
        metadata=metadata or {},
    )


def model_to_data(model: BaseModel) -> dict[str, Any]:
    """Best-effort conversion of a Pydantic model into JSON-safe dict for TaskResult.data."""
    dumped = model.model_dump(mode="json")
    return dumped if isinstance(dumped, dict) else {"value": dumped}
