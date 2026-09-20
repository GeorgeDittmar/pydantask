"""System resource introspection for the dynamic model routing system.

Uses ``psutil`` to query host metrics (memory, CPU, disk) and the queue store
to track active port allocations and resident model states.

The supervisor calls ``get_system_resources()`` before scheduling to determine:
- Whether enough memory is available to spawn a given model
- How many tasks can run in parallel without OOM risk
- Which ports are currently in use by the lifecycle manager

Usage::

    from pydantask.execution.resources import get_system_resources

    resources = get_system_resources()
    print(resources.available_memory_mb)  # e.g. 32000
    print(resources.memory_pressure_percent)  # e.g. 50

    from pydantask.execution.registry import get_model
    model = get_model("qwen2.5-coder-7b")
    can_spawn = resources.can_fit_model(model)  # False if memory is tight
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from pydantask.execution.schema import QueueStore
from pydantask.execution.registry import MODEL_REGISTRY, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Memory pressure thresholds
# ─────────────────────────────────────────────────────────────────────────────

PRESSURE_LOW_THRESHOLD = 30   # < 30% used = low pressure
PRESSURE_HIGH_THRESHOLD = 85  # > 85% used = high pressure


# Sentinel for "callable not yet bound"
_UNBOUND = object()


@dataclass
class SystemResources:
    """Snapshot of system resources at a point in time.

    Attributes:
        available_memory_mb: Available physical/unified memory in MB.
        total_memory_mb: Total physical/unified memory in MB.
        used_memory_mb: Memory currently in use in MB.
        memory_pressure_percent: Percentage of memory in use (0–100).
        active_ports: Set of ports currently leased by the lifecycle manager.
        resident_models: List of model keys currently resident in memory.
        parallel_capacity: Maximum number of parallel tasks that can fit.
    """

    available_memory_mb: int
    total_memory_mb: int
    used_memory_mb: int
    memory_pressure_percent: int
    active_ports: Set[int] = field(default_factory=set)
    resident_models: List[str] = field(default_factory=list)
    parallel_capacity: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Resource introspection
# ─────────────────────────────────────────────────────────────────────────────


def _get_memory_metrics() -> Dict[str, int]:
    """Get memory metrics from psutil (or return zeros if psutil unavailable)."""
    if not HAS_PSUTIL:
        return {
            "available_mb": 0,
            "total_mb": 0,
            "used_mb": 0,
            "percent": 0,
        }
    mem = psutil.virtual_memory()
    return {
        "available_mb": mem.available // (1024 * 1024),
        "total_mb": mem.total // (1024 * 1024),
        "used_mb": mem.used // (1024 * 1024),
        "percent": mem.percent,
    }


def _get_active_ports_from_store(store: QueueStore) -> Set[int]:
    """Get ports currently leased by the lifecycle manager from the store."""
    rows = store.conn.execute(
        "SELECT port FROM port_leases WHERE task_id IN ("
        "  SELECT id FROM task_queue WHERE status = 'processing')"
    ).fetchall()
    return {row["port"] for row in rows}


def _get_active_ports_system_wide() -> Set[int]:
    """Get ports currently in use by any process on the system.

    Falls back to checking /proc/net/tcp on Linux or using socket if psutil
    is unavailable.
    """
    if not HAS_PSUTIL:
        return set()
    try:
        conns = psutil.net_connections(kind="inet")
        return {c.laddr.port for c in conns if c.status == "LISTEN"}
    except (psutil.AccessDenied, OSError):
        return set()


def _compute_parallel_capacity(
    available_mb: int,
    active_ports: Set[int],
) -> int:
    """Estimate how many more tasks can be run in parallel.

    Based on available memory: each task needs at minimum the smallest
    model's total_required_mb.
    """
    smallest_model = min(MODEL_REGISTRY.values(), key=lambda m: m.total_required_mb)
    min_per_task = smallest_model.total_required_mb
    return max(0, available_mb // min_per_task)


def get_system_resources(
    store: Optional[QueueStore] = None,
) -> SystemResources:
    """Query live system metrics for resource-aware scheduling.

    Args:
        store: Optional QueueStore to query active port allocations.
            If provided, ports are read from the lease table.
            If omitted, system-wide port usage is checked via psutil.

    Returns:
        A ``SystemResources`` snapshot with memory metrics, active ports,
        parallel capacity, and helper methods.

        The returned object has these additional attributes (set dynamically
        for convenience, not declared on the dataclass):
        - ``can_fit_model(model: ModelConfig) -> bool``
        - ``best_fitting_model(max_parallel: int = 1) -> Optional[str]``

    Note:
        If ``psutil`` is not installed, returns zeroed metrics.
    """
    metrics = _get_memory_metrics()
    available_mb = metrics["available_mb"]
    total_mb = metrics["total_mb"]
    used_mb = metrics["used_mb"]
    pressure = metrics["percent"]

    # Active ports
    if store is not None:
        active_ports = _get_active_ports_from_store(store)
    else:
        active_ports = _get_active_ports_system_wide()

    # Parallel capacity
    capacity = _compute_parallel_capacity(available_mb, active_ports)

    result = SystemResources(
        available_memory_mb=available_mb,
        total_memory_mb=total_mb,
        used_memory_mb=used_mb,
        memory_pressure_percent=pressure,
        active_ports=active_ports,
        resident_models=[],  # Population handled by lifecycle manager
        parallel_capacity=capacity,
    )

    # Attach helper methods (not dataclass fields — convenience only)
    def can_fit_model(model: ModelConfig) -> bool:
        """Check if a given model can be spawned with current resources."""
        return model.total_required_mb <= available_mb

    def best_fitting_model(max_parallel: int = 1) -> Optional[str]:
        """Find the largest model that fits in available memory."""
        per_task_budget = available_mb // max(max_parallel, 1)
        fitting = [
            (k, m) for k, m in MODEL_REGISTRY.items()
            if m.total_required_mb <= per_task_budget
        ]
        if not fitting:
            return None
        return max(fitting, key=lambda x: x[1].total_required_mb)[0]

    result.can_fit_model = can_fit_model  # type: ignore[attr-defined]
    result.best_fitting_model = best_fitting_model  # type: ignore[attr-defined]
    return result


def get_memory_headroom(model: ModelConfig, store: Optional[QueueStore] = None) -> Optional[bool]:
    """Check if there's sufficient memory headroom for a model.

    Args:
        model: The model to check.
        store: Optional QueueStore for active port tracking.

    Returns:
        True if enough memory is available, False if not, None if psutil
        is unavailable.
    """
    resources = get_system_resources(store)
    if resources.available_memory_mb == 0:
        return None  # psutil unavailable
    return resources.can_spawn_model(model)


def should_throttle_pressure() -> bool:
    """Check if memory pressure exceeds the high threshold.

    When pressure is high, the supervisor should prefer serial execution
    or drop to smaller models.

    Returns:
        True if memory_pressure_percent > PRESSURE_HIGH_THRESHOLD.
    """
    metrics = _get_memory_metrics()
    return metrics["percent"] > PRESSURE_HIGH_THRESHOLD
