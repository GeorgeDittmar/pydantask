"""Tests for the DAG dependency resolution engine."""

from __future__ import annotations

import tempfile

import pytest

from pydantask.execution.resolver import (
    get_dag_status,
    is_dag_complete,
    resolve_parent_output,
    resolve_ready_tasks,
    validate_dag,
)
from pydantask.execution.schema import QueueStore

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> QueueStore:
    s = QueueStore(tempfile.mktemp(suffix=".db"))
    s.init()
    yield s
    s.close()


@pytest.fixture
def simple_dag_store(store: QueueStore) -> QueueStore:
    """Create a store with a simple DAG: 1 -> 2 -> 3."""
    store.insert_task(dag_id="chain")  # task 1
    store.insert_task(dag_id="chain", parent_id=1, in_degree=1)  # task 2
    store.insert_task(dag_id="chain", parent_id=2, in_degree=1)  # task 3
    return store


# ─────────────────────────────────────────────────────────────────────────────
# DAG validation — Kahn's algorithm
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateDAG:
    def test_empty_dag(self) -> None:
        valid, result = validate_dag([])
        assert valid is True
        assert result == []

    def test_single_node(self) -> None:
        tasks = [{"task_id": 1, "parent_ids": []}]
        valid, order = validate_dag(tasks)
        assert valid is True
        assert order == [1]

    def test_linear_chain(self) -> None:
        tasks = [
            {"task_id": 1, "parent_ids": []},
            {"task_id": 2, "parent_ids": [1]},
            {"task_id": 3, "parent_ids": [2]},
        ]
        valid, order = validate_dag(tasks)
        assert valid is True
        assert order == [1, 2, 3]

    def test_diamond_dependency(self) -> None:
        """1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4."""
        tasks = [
            {"task_id": 1, "parent_ids": []},
            {"task_id": 2, "parent_ids": [1]},
            {"task_id": 3, "parent_ids": [1]},
            {"task_id": 4, "parent_ids": [2, 3]},
        ]
        valid, order = validate_dag(tasks)
        assert valid is True
        assert order[0] == 1  # Root first
        assert order[-1] == 4  # Diamond tip last

    def test_two_roots(self) -> None:
        tasks = [
            {"task_id": 1, "parent_ids": []},
            {"task_id": 2, "parent_ids": []},
        ]
        valid, order = validate_dag(tasks)
        assert valid is True
        assert set(order) == {1, 2}

    def test_simple_cycle(self) -> None:
        """A -> B -> A."""
        tasks = [
            {"task_id": 1, "parent_ids": [2]},
            {"task_id": 2, "parent_ids": [1]},
        ]
        valid, result = validate_dag(tasks)
        assert valid is False
        assert "cycle" in str(result).lower()
        assert "1" in str(result)
        assert "2" in str(result)

    def test_self_cycle(self) -> None:
        """A -> A."""
        tasks = [{"task_id": 1, "parent_ids": [1]}]
        valid, result = validate_dag(tasks)
        assert valid is False

    def test_longer_cycle(self) -> None:
        """A -> B -> C -> A."""
        tasks = [
            {"task_id": 1, "parent_ids": [3]},
            {"task_id": 2, "parent_ids": [1]},
            {"task_id": 3, "parent_ids": [2]},
        ]
        valid, result = validate_dag(tasks)
        assert valid is False
        assert "cycle" in str(result).lower()

    def test_partial_cycle(self) -> None:
        """Some nodes cycle, some don't — cycle nodes should be identified."""
        tasks = [
            {"task_id": 1, "parent_ids": []},  # isolated, fine
            {"task_id": 2, "parent_ids": [3]},  # cycle
            {"task_id": 3, "parent_ids": [2]},  # cycle
        ]
        valid, result = validate_dag(tasks)
        assert valid is False
        # Node 1 should NOT be in the cycle report
        assert "2" in str(result)
        assert "3" in str(result)

    def test_unknown_parent(self) -> None:
        tasks = [
            {"task_id": 1, "parent_ids": []},
            {"task_id": 2, "parent_ids": [99]},
        ]
        valid, result = validate_dag(tasks)
        assert valid is False
        assert "unknown parent" in str(result).lower()

    def test_topological_order_valid(self) -> None:
        """Ensure returned order is a valid topological sort."""
        tasks = [{"task_id": i, "parent_ids": list(range(i - 1, 0, -1))} for i in range(1, 6)]
        # Task 1: no parents
        # Task 2: parent [1]
        # Task 3: parents [1, 2]
        # Task 4: parents [1, 2, 3]
        # Task 5: parents [1, 2, 3, 4]
        valid, order = validate_dag(tasks)
        assert valid is True

        # Each task must appear after all its parents
        position = {tid: idx for idx, tid in enumerate(order)}
        for task in tasks:
            for parent_id in task["parent_ids"]:
                assert position[parent_id] < position[task["task_id"]], (
                    f"Task {task['task_id']} appears before parent {parent_id}"
                )


class TestResolveReadyTasks:
    def test_all_ready(self, store: QueueStore) -> None:
        for _i in range(3):
            store.insert_task(dag_id="dag")

        ready = resolve_ready_tasks(store)
        assert len(ready) == 3

    def test_only_roots_ready(self, store: QueueStore) -> None:
        root_id = store.insert_task(dag_id="dag")
        child_id = store.insert_task(dag_id="dag", parent_id=root_id, in_degree=1)

        ready = resolve_ready_tasks(store)
        assert ready == [root_id]
        assert child_id not in ready

    def test_limits(self, store: QueueStore) -> None:
        for _i in range(5):
            store.insert_task(dag_id="dag")

        ready = resolve_ready_tasks(store, limit=2)
        assert len(ready) == 2

    def test_blocked_children_after_unblock(self, store: QueueStore) -> None:
        parent_id = store.insert_task(dag_id="dag")
        child_id = store.insert_task(dag_id="dag", parent_id=parent_id, in_degree=1)

        # Unblock
        store.decrement_children_in_degree(parent_id)

        ready = resolve_ready_tasks(store)
        assert child_id in ready


class TestResolveParentOutput:
    def test_no_parent(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        assert resolve_parent_output(store, "dag", task_id) is None

    def test_completed_parent(self, store: QueueStore) -> None:
        parent_id = store.insert_task(dag_id="dag")
        child_id = store.insert_task(dag_id="dag", parent_id=parent_id, in_degree=1)

        parent_output = '{"status": "ok", "content": "done", "model_used": "test"}'
        store.mark_completed(parent_id, parent_output)
        store.decrement_children_in_degree(parent_id)

        result = resolve_parent_output(store, "dag", child_id)
        assert result == parent_output

    def test_incomplete_parent(self, store: QueueStore) -> None:
        parent_id = store.insert_task(dag_id="dag")
        store.insert_task(dag_id="dag", parent_id=parent_id, in_degree=1)
        # Parent still pending — no output
        result = resolve_parent_output(store, "dag", parent_id)
        assert result is None

    def test_nonexistent_task(self, store: QueueStore) -> None:
        assert resolve_parent_output(store, "dag", 9999) is None


class TestGetDAGStatus:
    def test_all_pending(self, store: QueueStore) -> None:
        for _ in range(3):
            store.insert_task(dag_id="dag")
        status = get_dag_status(store, "dag")
        assert status == {"pending": 3}

    def test_mixed_status(self, store: QueueStore) -> None:
        root_id = store.insert_task(dag_id="dag")
        store.insert_task(dag_id="dag", parent_id=root_id, in_degree=1)

        store.mark_completed(root_id, '{"status":"ok","content":"","model_used":"t"}')
        status = get_dag_status(store, "dag")
        assert status["completed"] == 1
        assert status["pending"] == 1

    def test_empty_dag(self, store: QueueStore) -> None:
        assert get_dag_status(store, "nonexistent-dag") == {}


class TestIsDAGComplete:
    def test_all_completed(self, store: QueueStore) -> None:
        for _ in range(3):
            task_id = store.insert_task(dag_id="dag")
            store.mark_completed(task_id, '{"status":"ok","content":"","model_used":"t"}')
        assert is_dag_complete(store, "dag") is True

    def test_has_pending(self, store: QueueStore) -> None:
        store.insert_task(dag_id="dag")
        assert is_dag_complete(store, "dag") is False

    def test_has_processing(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        store.mark_processing(task_id, 60001)
        assert is_dag_complete(store, "dag") is False

    def test_has_failed(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        store.mark_failed(task_id, "error", 0)
        assert is_dag_complete(store, "dag") is True

    def test_empty_dag(self, store: QueueStore) -> None:
        assert is_dag_complete(store, "nonexistent-dag") is True


class TestAtomicStartTask:
    def test_start_task_success(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        result = store.start_task(task_id, 60001)
        assert result is True

        row = store.get_task(task_id)
        assert row["status"] == "processing"
        assert row["assigned_port"] == 60001

    def test_start_task_port_already_leased(self, store: QueueStore) -> None:
        task_id1 = store.insert_task(dag_id="dag1")
        task_id2 = store.insert_task(dag_id="dag2")

        store.start_task(task_id1, 60001)
        result = store.start_task(task_id2, 60001)
        assert result is False  # Port already leased

        # Task 2 should NOT have been modified
        row = store.get_task(task_id2)
        assert row["status"] == "pending"

    def test_next_available_port(self, store: QueueStore) -> None:
        port1 = store.next_available_port()
        assert 60000 <= port1 <= 65535

        # After leasing, next call should skip it
        task_id = store.insert_task(dag_id="dag")
        store.start_task(task_id, port1)
        port2 = store.next_available_port()
        assert port2 != port1

    def test_release_task(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        store.start_task(task_id, 60001)

        row = store.get_task(task_id)
        assert row["status"] == "processing"

        store.release_task(task_id)

        row = store.get_task(task_id)
        assert row["status"] == "pending"
        assert row["assigned_port"] is None
