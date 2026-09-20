"""Dynamic model routing — resource-aware task execution backend.

This package implements the SQLite-backed execution queue, structured spawn
argument validation, model registry config, skill schema loading, DAG
resolution, atomic port allocation, and system resource introspection that
power the dynamic model routing system.

Public API::

    from pydantask.execution import QueueStore, ModelConfig, SpawnArgs, MODEL_REGISTRY
    from pydantask.execution import get_system_resources, validate_dag

    store = QueueStore("run_42.db")
    store.init()
    store.insert_task(dag_id="research", target_model_key="qwen2.5-coder-7b", ...)

    # Validate DAG has no cycles before insertion
    valid, result = validate_dag(tasks)

    # Get system resources for scheduling decisions
    resources = get_system_resources(store)
    resources.can_spawn_model(model)
"""

from pydantask.execution.registry import (
    MODEL_REGISTRY,
    find_fit_models,
    find_models_by_tier,
    get_model,
    get_smallest_model,
)
from pydantask.execution.resolver import (
    get_dag_status,
    is_dag_complete,
    resolve_parent_output,
    resolve_ready_tasks,
    validate_dag,
)
from pydantask.execution.resources import (
    get_memory_headroom,
    get_system_resources,
    should_throttle_pressure,
)
from pydantask.execution.schema import (
    ModelConfig,
    QueueStore,
    SpawnArgs,
    TaskOutput,
    TaskPayload,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelConfig",
    "QueueStore",
    "SpawnArgs",
    "TaskOutput",
    "TaskPayload",
    "find_fit_models",
    "find_models_by_tier",
    "get_dag_status",
    "get_memory_headroom",
    "get_model",
    "get_smallest_model",
    "get_system_resources",
    "is_dag_complete",
    "resolve_parent_output",
    "resolve_ready_tasks",
    "should_throttle_pressure",
    "validate_dag",
]
