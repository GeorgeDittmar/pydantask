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
from pydantask.execution.resources import (
    get_system_resources,
    get_memory_headroom,
    should_throttle_pressure,
)
from pydantask.execution.resolver import (
    validate_dag,
    resolve_ready_tasks,
    resolve_parent_output,
    get_dag_status,
    is_dag_complete,
)
from pydantask.execution.schema import (
    ModelConfig,
    QueueStore,
    SpawnArgs,
    TaskOutput,
    TaskPayload,
)

__all__ = [
    # Schema & models
    "QueueStore",
    "ModelConfig",
    "SpawnArgs",
    "TaskPayload",
    "TaskOutput",
    # Registry
    "MODEL_REGISTRY",
    "get_model",
    "get_smallest_model",
    "find_models_by_tier",
    "find_fit_models",
    # DAG resolution
    "validate_dag",
    "resolve_ready_tasks",
    "resolve_parent_output",
    "get_dag_status",
    "is_dag_complete",
    # Resources
    "get_system_resources",
    "get_memory_headroom",
    "should_throttle_pressure",
]
