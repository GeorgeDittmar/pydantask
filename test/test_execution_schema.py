"""Tests for the SQLite-backed execution queue and Pydantic validation models."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import sqlite3

from pydantask.execution.schema import (
    ModelConfig,
    SpawnArgs,
    TaskOutput,
    TaskPayload,
    QueueStore,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelConfig:
    def test_basic(self):
        cfg = ModelConfig(
            model_key="qwen2.5-coder-7b",
            path="/models/qwen2.5-coder-7b.gguf",
            default_port=8082,
            default_ctx=8192,
            estimated_vram_mb=4200,
            capability_tier="coder",
            min_memory_headroom_mb=2000,
        )
        assert cfg.model_key == "qwen2.5-coder-7b"
        assert cfg.total_required_mb == 6200

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelConfig(
                model_key="x",
                path="/x",
                default_port=1,
                default_ctx=1,
                estimated_vram_mb=1,
                capability_tier="x",
                min_memory_headroom_mb=1,
                unknown_field="oops",
            )


class TestSpawnArgs:
    def test_defaults(self):
        args = SpawnArgs()
        assert args.ctx_size == 4096
        assert args.n_gpu_layers == 99
        assert args.mmap is True
        assert args.temp == 0.7
        assert args.top_p == 0.9

    def test_custom_values(self):
        args = SpawnArgs(ctx_size=16384, temp=1.0, mmap=False)
        assert args.ctx_size == 16384
        assert args.temp == 1.0
        assert args.mmap is False

    def test_to_dict(self):
        args = SpawnArgs(ctx_size=8192)
        d = args.to_dict()
        assert "ctx_size" in d
        assert d["ctx_size"] == 8192
        # mmap defaults to True so it's excluded by exclude_none=False...
        # actually exclude_none=True, True is not None so it should be included
        assert "mmap" in d

    def test_from_dict(self):
        d = {"ctx_size": 4096, "n_gpu_layers": 32, "temp": 0.5}
        args = SpawnArgs.from_dict(d)
        assert args.ctx_size == 4096
        assert args.n_gpu_layers == 32
        assert args.temp == 0.5

    def test_from_dict_ignores_unknown_keys(self):
        d = {"ctx_size": 4096, "future_flag": "ignored"}
        args = SpawnArgs.from_dict(d)
        assert args.ctx_size == 4096
        assert not hasattr(args, "future_flag")

    def test_validation_rejects_out_of_range(self):
        with pytest.raises(Exception):
            SpawnArgs(temp=-0.1)

        with pytest.raises(Exception):
            SpawnArgs(temp=2.1)

        with pytest.raises(Exception):
            SpawnArgs(top_p=-0.1)

    def test_serialization_roundtrip(self):
        args = SpawnArgs(ctx_size=16384, temp=1.2)
        d = args.to_dict()
        json_str = json.dumps(d)
        loaded = SpawnArgs.from_dict(json.loads(json_str))
        assert loaded.ctx_size == args.ctx_size
        assert loaded.temp == args.temp


class TestTaskPayload:
    def test_defaults(self):
        p = TaskPayload()
        assert p.system_prompt == ""
        assert p.user_prompt == ""
        assert p.constraints is None
        assert p.parent_output_ref is None

    def test_with_data(self):
        p = TaskPayload(
            system_prompt="You are a researcher.",
            user_prompt="Summarize the paper.",
            constraints={"max_words": 500},
        )
        assert "max_words" in p.constraints

    def test_serialization(self):
        p = TaskPayload(system_prompt="hello", user_prompt="world")
        json_str = p.to_json()
        loaded = TaskPayload.from_json(json_str)
        assert loaded.system_prompt == "hello"
        assert loaded.user_prompt == "world"


class TestTaskOutput:
    def test_ok(self):
        out = TaskOutput(status="ok", content="Results ready.", model_used="qwen2.5-coder-7b")
        assert out.status == "ok"
        assert out.error_msg is None

    def test_error(self):
        out = TaskOutput(
            status="error",
            content="",
            model_used="mistral-7b",
            error_msg="OOM during execution",
            tokens_used=1024,
        )
        assert out.error_msg == "OOM during execution"

    def test_serialization(self):
        out = TaskOutput(status="ok", content="done", model_used="tiny")
        json_str = out.to_json()
        loaded = TaskOutput.from_json(json_str)
        assert loaded.status == "ok"
        assert loaded.model_used == "tiny"


# ─────────────────────────────────────────────────────────────────────────────
# QueueStore tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> QueueStore:
    """Create a temporary SQLite database and initialize it."""
    s = QueueStore(tempfile.mktemp(suffix=".db"))
    s.init()
    yield s
    s.close()  # Clean up connection to avoid resource warnings


class TestQueueInit:
    def test_creates_tables(self, store: QueueStore) -> None:
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "task_queue" in names
        assert "port_leases" in names

    def test_wal_mode(self, store: QueueStore) -> None:
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()["journal_mode"]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, store: QueueStore) -> None:
        fk = store.conn.execute("PRAGMA foreign_keys").fetchone()["foreign_keys"]
        assert fk == 1

    def test_double_init_is_safe(self, store: QueueStore) -> None:
        """Calling init() twice shouldn't raise."""
        store.init()


class TestInsertAndGet:
    def test_insert_task(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="research", target_model_key="qwen2.5-coder-7b")
        assert isinstance(task_id, int)
        assert task_id > 0

    def test_insert_with_all_fields(self, store: QueueStore) -> None:
        spawn = SpawnArgs(ctx_size=16384, temp=0.5)
        payload = TaskPayload(
            system_prompt="Be concise.",
            user_prompt="Research the topic.",
            constraints={"max_tokens": 1000},
        )
        task_id = store.insert_task(
            dag_id="research",
            target_model_key="qwen2.5-coder-7b",
            spawn_args=spawn,
            payload=payload,
        )

        row = store.get_task(task_id)
        assert row is not None
        assert row["dag_id"] == "research"
        assert row["target_model_key"] == "qwen2.5-coder-7b"
        assert row["status"] == "pending"

        # Verify spawn_args were stored as JSON
        args = SpawnArgs.from_dict(json.loads(row["spawn_args"]))
        assert args.ctx_size == 16384

    def test_insert_with_parent(self, store: QueueStore) -> None:
        parent_id = store.insert_task(dag_id="workflow")
        child_id = store.insert_task(
            dag_id="workflow",
            parent_id=parent_id,
            in_degree=1,
        )

        child = store.get_task(child_id)
        assert child["parent_id"] == parent_id
        assert child["in_degree"] == 1

    def test_get_task_missing(self, store: QueueStore) -> None:
        assert store.get_task(9999) is None


class TestDAGResolution:
    def test_parent_completion_decrements_child(self, store: QueueStore) -> None:
        parent_id = store.insert_task(dag_id="dag")
        child_id = store.insert_task(dag_id="dag", parent_id=parent_id, in_degree=1)

        # Initially, child has in_degree=1
        child = store.get_task(child_id)
        assert child["in_degree"] == 1

        # Mark parent as completed
        store.mark_completed(task_id=parent_id, output_json=json.dumps({
            "status": "ok",
            "content": "parent result",
            "model_used": "test",
        }))

        # Decrement children — this is what the lifecycle manager does
        store.decrement_children_in_degree(parent_id)

        child = store.get_task(child_id)
        assert child["in_degree"] == 0

    def test_ready_tasks_excludes_blocked(self, store: QueueStore) -> None:
        root_id = store.insert_task(dag_id="dag")
        blocked_id = store.insert_task(
            dag_id="dag",
            parent_id=root_id,
            in_degree=1,
        )

        ready = store.get_ready_tasks()
        ready_ids = [r["id"] for r in ready]
        assert root_id in ready_ids
        assert blocked_id not in ready_ids

    def test_ready_tasks_after_unblock(self, store: QueueStore) -> None:
        parent_id = store.insert_task(dag_id="dag")
        child_id = store.insert_task(dag_id="dag", parent_id=parent_id, in_degree=1)

        # Unblock parent
        store.decrement_children_in_degree(parent_id)

        ready = store.get_ready_tasks()
        ready_ids = [r["id"] for r in ready]
        assert child_id in ready_ids

    def test_no_cycles_rejected_by_topology(self) -> None:
        """Cycles are detected before queue insertion — this is a supervisor
        responsibility, but we verify the invariant here."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = QueueStore(f.name)
            store.init()

            # Task A depends on B, B depends on C, C depends on A
            a = store.insert_task(dag_id="cycle")
            b = store.insert_task(dag_id="cycle", parent_id=a, in_degree=1)
            c = store.insert_task(dag_id="cycle", parent_id=b, in_degree=1)

            # C still depends on A — its in_degree stays at 1
            c_row = store.get_task(c)
            assert c_row["in_degree"] == 1

            # A is the root of this chain, so it's ready
            ready = store.get_ready_tasks()
            ready_ids = [r["id"] for r in ready]
            assert a in ready_ids

            # After completing A, B becomes ready. But completing B doesn't
            # unblock C because C still depends on A (which is now completed
            # but C's parent_id points to B).
            # In a real cycle (A→B→C→A), A would never have parent_id=A
            # so the topological sort at construction time should catch it.
            # This test demonstrates that in_degree-based resolution handles
            # chains correctly; cycle detection is the supervisor's job.


class TestPortLeases:
    def test_acquire_and_release(self, store: QueueStore) -> None:
        port = 60001
        task_id = store.insert_task(dag_id="dag")

        # Use transaction for atomic port lease + status update
        with store.transaction():
            store.mark_processing(task_id, port)
            acquired = store.acquire_lease(port, task_id)
            assert acquired is True

        # Verify lease exists
        row = store.conn.execute(
            "SELECT * FROM port_leases WHERE port = ?", (port,)
        ).fetchone()
        assert row is not None
        assert row["task_id"] == task_id

        # Release
        store.release_lease(port)

        row = store.conn.execute(
            "SELECT * FROM port_leases WHERE port = ?", (port,)
        ).fetchone()
        assert row is None

    def test_double_lease_fails(self, store: QueueStore) -> None:
        port = 60002
        task_id = store.insert_task(dag_id="dag")

        with store.transaction():
            store.mark_processing(task_id, port)
            acquired1 = store.acquire_lease(port, task_id)
            assert acquired1 is True

            # Try to acquire the same port for a different task
            task_id2 = store.insert_task(dag_id="dag")
            acquired2 = store.acquire_lease(port, task_id2)
            assert acquired2 is False  # Port already leased

    def test_stale_lease_detection(self, store: QueueStore) -> None:
        """Insert a lease manually with an old timestamp and verify it's found."""
        from datetime import datetime, timedelta

        port = 60003
        task_id = store.insert_task(dag_id="dag")

        # Insert a lease with a timestamp 10 minutes ago (real datetime, not SQL expression)
        old_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        with store.transaction():
            store.conn.execute(
                "INSERT INTO port_leases (port, task_id, leased_at) VALUES (?, ?, ?)",
                (port, task_id, old_time),
            )

        # Mark task as completed (so it's NOT in processing)
        store.mark_completed(task_id, json.dumps({"status": "ok", "content": "", "model_used": "test"}))

        stale = store.find_stale_leases(max_age_seconds=300)
        assert len(stale) == 1
        assert stale[0][0] == port


class TestCompletionAndFailure:
    def test_mark_completed(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        store.mark_completed(
            task_id,
            json.dumps({"status": "ok", "content": "done", "model_used": "qwen2.5", "tokens_used": 100}),
            spawn_model_key="qwen2.5",
        )

        row = store.get_task(task_id)
        assert row["status"] == "completed"
        assert row["spawn_model_key"] == "qwen2.5"

        output = TaskOutput.from_json(row["output"])
        assert output.content == "done"
        assert output.tokens_used == 100

    def test_mark_failed(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        store.mark_failed(task_id, error_log="OOM", retry_count=1)

        row = store.get_task(task_id)
        assert row["status"] == "failed"
        assert row["error_log"] == "OOM"
        assert row["retry_count"] == 1

    def test_mark_processing(self, store: QueueStore) -> None:
        task_id = store.insert_task(dag_id="dag")
        store.mark_processing(task_id, 60010)

        row = store.get_task(task_id)
        assert row["status"] == "processing"
        assert row["assigned_port"] == 60010


class TestEscalation:
    def test_update_spawn_args(self, store: QueueStore) -> None:
        task_id = store.insert_task(
            dag_id="dag",
            target_model_key="old-model",
            spawn_args=SpawnArgs(ctx_size=4096),
        )

        # Escalate to a larger model config with explicit target
        new_args = SpawnArgs(ctx_size=16384, n_gpu_layers=99)
        store.update_spawn_args(task_id, new_args, target_model_key="qwen2.5-32b")

        row = store.get_task(task_id)
        assert row["status"] == "pending"  # Reset to pending
        assert row["spawn_model_key"] is None  # Will be recorded at execution
        assert row["target_model_key"] == "qwen2.5-32b"

        args = SpawnArgs.from_dict(json.loads(row["spawn_args"]))
        assert args.ctx_size == 16384
        assert args.n_gpu_layers == 99


class TestCleanup:
    def test_cleanup_completed_dag(self, store: QueueStore) -> None:
        for i in range(3):
            store.insert_task(dag_id="cleanup-test")

        # Complete all of them
        for row in store.get_pending_by_dag("cleanup-test"):
            store.mark_completed(row["id"], json.dumps({"status": "ok", "content": "", "model_used": "test"}))

        deleted = store.cleanup_completed_dag("cleanup-test")
        assert deleted == 3

        remaining = store.get_pending_by_dag("cleanup-test")
        assert len(remaining) == 0


class TestTransactionIsolation:
    def test_rollback_on_error(self, store: QueueStore) -> None:
        """Changes inside a failed transaction should be rolled back."""
        task_id = store.insert_task(dag_id="dag")

        try:
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE task_queue SET status = 'processing' WHERE id = ?",
                    (task_id,),
                )
                raise RuntimeError("intentional failure")
        except RuntimeError:
            pass  # Expected

        # Should be rolled back
        row = store.get_task(task_id)
        assert row["status"] == "pending"
