"""DAG dependency resolution engine for the dynamic model routing system.

Provides cycle detection via Kahn's algorithm (BFS-based topological sort) and
helper utilities for resolving DAG readiness.

The supervisor calls ``validate_dag()`` before queue insertion to reject
circular dependencies. The lifecycle manager uses ``resolve_ready_tasks()`` to
determine which pending tasks can be executed next.

Usage::

    from pydantask.execution.resolver import (
        validate_dag,
        resolve_ready_tasks,
    )

    # Supervisor: validate before queue insertion
    tasks = [
        {"task_id": 1, "parent_ids": []},
        {"task_id": 2, "parent_ids": [1]},
        {"task_id": 3, "parent_ids": [1, 2]},
    ]
    is_valid, result = validate_dag(tasks)
    if not is_valid:
        raise ValueError(f"Invalid DAG: {result}")

    # Lifecycle manager: get tasks ready for execution
    ready_ids = resolve_ready_tasks(store)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantask.execution.schema import QueueStore

# ─────────────────────────────────────────────────────────────────────────────
# DAG validation — Kahn's algorithm for cycle detection
# ─────────────────────────────────────────────────────────────────────────────


def validate_dag(
    tasks: list[dict[str, Any]],
) -> tuple[bool, list[int] | str]:
    """Validate that a set of task specs forms a valid DAG (no cycles).

    Uses Kahn's algorithm (BFS-based topological sort). If the sort completes
    for all nodes, the graph is acyclic and the topological order is returned.
    If some nodes remain with non-zero in-degree, a cycle exists.

    Args:
        tasks: List of task dicts, each with at minimum:
            - ``task_id``: Unique integer identifier
            - ``parent_ids``: List of parent task IDs this task depends on

    Returns:
        ``(True, topological_order)`` if valid — order is a valid execution
        sequence where no task appears before its parents.
        ``(False, error_message)`` if invalid — ``error_message`` explains
        which nodes are part of the cycle.
    """
    if not tasks:
        return True, []

    # Build adjacency list and in-degree map
    node_ids: set[int] = {t["task_id"] for t in tasks}
    children: dict[int, list[int]] = {tid: [] for tid in node_ids}
    in_degree: dict[int, int] = {tid: 0 for tid in node_ids}

    for task in tasks:
        tid = task["task_id"]
        for parent_id in task.get("parent_ids", []):
            if parent_id not in node_ids:
                return False, (
                    f"Task {tid} references unknown parent {parent_id}. "
                    "All parent_ids must reference existing task_ids in the DAG."
                )
            children[parent_id].append(tid)
            in_degree[tid] += 1

    # Kahn's algorithm: BFS from nodes with zero in-degree
    queue: deque[int] = deque(tid for tid, deg in in_degree.items() if deg == 0)
    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(node_ids):
        # Nodes remaining in in_degree with non-zero values are in a cycle
        cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
        return False, (
            f"DAG contains a cycle involving task IDs: {cycle_nodes}. "
            "Topological sort could not process all nodes."
        )

    return True, order


def validate_dag_against_store(
    store: QueueStore,
    dag_id: str,
    tasks: list[dict[str, Any]],
) -> tuple[bool, list[int] | str]:
    """Validate a DAG against an existing QueueStore.

    In addition to cycle detection, checks that all parent_ids reference
    tasks that actually exist in the store (for tasks with non-empty parent_ids
    lists).

    Args:
        store: An initialized QueueStore.
        dag_id: The DAG this belongs to (for error messages).
        tasks: Task specs as accepted by ``validate_dag()``.

    Returns:
        ``(True, topological_order)`` or ``(False, error_message)``.
    """
    # First check for cycles
    is_valid, result = validate_dag(tasks)
    if not is_valid:
        return False, f"DAG '{dag_id}': {result}"

    # Check parent references exist in the store
    existing_ids: set[int] = set()
    for row in store.get_pending_by_dag(dag_id) + _get_all_task_ids(store, dag_id):
        existing_ids.add(row["id"])

    for task in tasks:
        for parent_id in task.get("parent_ids", []):
            if parent_id not in existing_ids:
                return False, (
                    f"DAG '{dag_id}': Task {task['task_id']} references "
                    f"parent {parent_id} which does not exist in the store."
                )

    return True, result


def _get_all_task_ids(store: QueueStore, dag_id: str) -> list[Any]:
    """Get all tasks (including completed) for a DAG."""
    return store.conn.execute(
        "SELECT id FROM task_queue WHERE dag_id = ?",
        (dag_id,),
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# DAG resolution helpers — used by the lifecycle manager
# ─────────────────────────────────────────────────────────────────────────────


def resolve_ready_tasks(store: QueueStore, limit: int = 10) -> list[int]:
    """Get IDs of tasks ready for execution.

    Returns task IDs where ``status='pending'`` AND ``in_degree=0``.

    Args:
        store: An initialized QueueStore.
        limit: Maximum number of ready tasks to return.

    Returns:
        List of task IDs ready for execution, ordered by insertion time.
    """
    rows = store.get_ready_tasks(limit)
    return [row["id"] for row in rows]


def resolve_parent_output(
    store: QueueStore,
    dag_id: str,
    task_id: int,
) -> str | None:
    """Resolve a task's parent output for downstream consumption.

    When a downstream task needs to read its parent's result, this method
    follows the ``parent_id`` chain and returns the parent's ``output`` column.

    Args:
        store: An initialized QueueStore.
        dag_id: The DAG this task belongs to.
        task_id: The task whose parent output is needed.

    Returns:
        The parent task's ``output`` JSON string, or ``None`` if no parent
        exists or the parent has no output yet.
    """
    row = store.get_task(task_id)
    if row is None:
        return None

    parent_id = row["parent_id"]
    if parent_id is None:
        return None

    parent_row = store.get_task(parent_id)
    if parent_row is None:
        return None

    return parent_row["output"]


def get_dag_status(store: QueueStore, dag_id: str) -> dict[str, int]:
    """Get the completion status of all tasks in a DAG.

    Args:
        store: An initialized QueueStore.
        dag_id: The DAG to inspect.

    Returns:
        Dict mapping status strings to task counts, e.g.::

            {"pending": 2, "completed": 5, "failed": 0, "processing": 1}
    """
    rows = store.conn.execute(
        "SELECT status, COUNT(*) as cnt FROM task_queue WHERE dag_id = ? GROUP BY status",
        (dag_id,),
    ).fetchall()
    return {row["status"]: row["cnt"] for row in rows}


def is_dag_complete(store: QueueStore, dag_id: str) -> bool:
    """Check if all tasks in a DAG have completed.

    A DAG is complete when every task has ``status='completed'`` or
    ``status='failed'``.

    Args:
        store: An initialized QueueStore.
        dag_id: The DAG to check.

    Returns:
        True if no pending or processing tasks remain.
    """
    pending = store.conn.execute(
        "SELECT COUNT(*) as cnt FROM task_queue "
        "WHERE dag_id = ? AND status IN ('pending', 'processing')",
        (dag_id,),
    ).fetchone()["cnt"]
    return pending == 0
