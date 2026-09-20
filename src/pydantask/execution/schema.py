"""SQLite-backed persistent queue for the dynamic model routing system.

Defines the ``task_queue`` and ``port_leases`` table schemas, provides WAL-mode
initialization, and exposes Pydantic validation models for every piece of data
that flows through the queue.

All schema objects are single-source-of-truth — the CREATE TABLE statements and
the Pydantic models share field names and types so the database always matches
the application contract.

Usage::

    from pydantask.execution.schema import QueueStore

    store = QueueStore("run_42.db")
    store.init()  # creates tables, enables WAL mode
    store.insert_task(...)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# SQL schema definitions
# ─────────────────────────────────────────────────────────────────────────────

_TASK_QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS task_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_id          TEXT    NOT NULL,
    parent_id       INTEGER,
    in_degree       INTEGER NOT NULL DEFAULT 0,
    target_model_key    TEXT,
    spawn_model_key       TEXT,
    spawn_args        TEXT,
    assigned_port     INTEGER,
    payload           TEXT,
    output            TEXT,
    status            TEXT    NOT NULL DEFAULT 'pending',
    retry_count       INTEGER NOT NULL DEFAULT 0,
    error_log         TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now', 'utc')),
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now', 'utc'))
);

CREATE INDEX IF NOT EXISTS idx_task_queue_dag
    ON task_queue (dag_id);

CREATE INDEX IF NOT EXISTS idx_task_queue_status_degree
    ON task_queue (status, in_degree);

CREATE INDEX IF NOT EXISTS idx_task_queue_target
    ON task_queue (target_model_key);

CREATE TABLE IF NOT EXISTS port_leases (
    port        INTEGER PRIMARY KEY,
    task_id     INTEGER NOT NULL REFERENCES task_queue (id),
    leased_at   TEXT    NOT NULL DEFAULT (datetime('now', 'utc'))
);

CREATE INDEX IF NOT EXISTS idx_port_leases_task
    ON port_leases (task_id);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic validation models
# ─────────────────────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Configuration for a single model in the model registry.

    The registry is consulted before task insertion to determine whether
    available memory can accommodate a given model, and what default
    spawn arguments to use.

    Attributes:
        model_key: Unique identifier for this model (e.g. ``"qwen2.5-coder-7b"``).
        path: Filesystem path to the GGUF model file.
        default_port: Preferred local port for this model's server instance.
        default_ctx: Default context window size (tokens).
        estimated_vram_mb: Expected VRAM/unified memory footprint when loaded.
        capability_tier: Human-readable capability label (``"coder"``, ``"reasoner"``, etc.).
        min_memory_headroom_mb: Additional headroom required beyond estimated_vram_mb
            for OS, llama-server overhead, and KV cache growth.
    """

    model_config = ConfigDict(extra="forbid")

    model_key: str = Field(description="Unique identifier for this model.")
    path: str = Field(description="Filesystem path to the GGUF model file.")
    default_port: int = Field(description="Preferred local port for this model's server.")
    default_ctx: int = Field(description="Default context window size (tokens).")
    estimated_vram_mb: int = Field(
        description="Expected VRAM/unified memory footprint when loaded."
    )
    capability_tier: str = Field(
        description="Human-readable capability label (e.g. 'coder', 'reasoner')."
    )
    min_memory_headroom_mb: int = Field(
        description=(
            "Additional headroom required beyond estimated_vram_mb for OS, "
            "server overhead, and KV cache growth."
        )
    )

    @property
    def total_required_mb(self) -> int:
        """Total memory needed = estimated_vram + headroom."""
        return self.estimated_vram_mb + self.min_memory_headroom_mb


class SpawnArgs(BaseModel):
    """Structured spawn arguments for a single task execution.

    The supervisor writes these (not a raw CLI command string).
    The lifecycle manager assembles the actual ``llama-server`` invocation
    at runtime from this data.

    Attributes:
        model_path: Absolute or relative path to the GGUF file. Overrides ``ModelConfig.path``
            if provided.
        ctx_size: Context window length. Defaults to ``ModelConfig.default_ctx``.
        n_gpu_layers: Number of layers to offload to Metal (Apple Silicon). Default 99.
        mmap: Whether to memory-map the model file. Default True.
        temp: Sampling temperature. Default 0.7.
        top_p: Nucleus sampling threshold. Default 0.9.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    model_path: str | None = Field(
        default=None,
        description="Path to the GGUF file. Falls back to registry path if omitted.",
    )
    ctx_size: int = Field(
        default=4096,
        description="Context window length (tokens).",
        ge=1,
    )
    n_gpu_layers: int = Field(
        default=99,
        description="Layers to offload to Metal on Apple Silicon.",
        ge=0,
        le=999,
    )
    mmap: bool = Field(
        default=True,
        description="Memory-map the model file (recommended for M-series chips).",
    )
    temp: float = Field(
        default=0.7,
        description="Sampling temperature.",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling top-p threshold.",
        ge=0.0,
        le=1.0,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (safe for JSON storage)."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpawnArgs:
        """Deserialize from a plain dict (e.g. from a JSON column)."""
        # Filter out keys that aren't valid fields (defensive — the DB may
        # contain extra keys from future schema versions).
        known_keys = set(cls.model_fields.keys())
        filtered = {k: v for k, v in data.items() if k in known_keys or k == "model_path"}
        return cls(**filtered)


class TaskPayload(BaseModel):
    """Input payload for a single queued task.

    Attributes:
        system_prompt: System message sent to the LLM server.
        user_prompt: User message / task input.
        constraints: Optional free-form constraints the supervisor attached.
        parent_output_ref: Optional reference to the parent task's output column
            (used when a downstream task needs to read a parent's result).
    """

    model_config = ConfigDict(extra="allow")

    system_prompt: str = Field(
        default="",
        description="System message sent to the LLM server.",
    )
    user_prompt: str = Field(
        default="",
        description="User message or task input.",
    )
    constraints: dict[str, Any] | None = Field(
        default=None,
        description="Optional free-form constraints from the supervisor.",
    )
    parent_output_ref: str | None = Field(
        default=None,
        description=(
            "Optional reference to the parent task's output row (dag_id:task_id). "
            "Used when a downstream task needs to read a parent's result."
        ),
    )

    def to_json(self) -> str:
        """Serialize to JSON string for the ``payload`` column."""
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_json(cls, raw: str) -> TaskPayload:
        """Deserialize from a JSON string (from the ``payload`` column)."""
        return cls.model_validate_json(raw)


class TaskOutput(BaseModel):
    """Structured output from a completed task execution.

    Stored in the ``output`` JSON column of ``task_queue``. Downstream DAG
    nodes read this to consume parent results.

    Attributes:
        status: ``"ok"`` or ``"error"``.
        content: The actual result content (text, JSON, etc.).
        model_used: The model key that actually executed this task
            (may differ from ``target_model_key`` after escalation).
        tokens_used: Number of prompt + completion tokens consumed.
        error_msg: If status is ``"error"``, a description of what failed.
    """

    model_config = ConfigDict(extra="allow")

    status: str = Field(
        description="Outcome: 'ok' or 'error'.",
    )
    content: str = Field(
        default="",
        description="The actual result content.",
    )
    model_used: str = Field(
        description="Model key that executed this task (may differ after escalation).",
    )
    tokens_used: int = Field(
        default=0,
        description="Total tokens consumed (prompt + completion).",
    )
    error_msg: str | None = Field(
        default=None,
        description="If status is 'error', a description of what failed.",
    )

    def to_json(self) -> str:
        """Serialize to JSON string for the ``output`` column."""
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_json(cls, raw: str) -> TaskOutput:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Queue Store — SQLite initialization + CRUD helpers
# ─────────────────────────────────────────────────────────────────────────────


class QueueStore:
    """Thin wrapper around the task_queue + port_leases SQLite tables.

    Handles WAL-mode setup, table creation, and provides transactional helpers
    for the lifecycle manager and supervisor.

    Uses a shared connection for all operations to avoid SQLite "database is
    locked" errors that arise from rapid connection churn in WAL mode.

    Args:
        path: Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazily-initialized shared connection."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode and sane defaults."""
        conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=30,  # 30s wait for locks
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def init(self) -> None:
        """Create tables and enable WAL mode. Safe to call multiple times."""
        self.conn.executescript(_TASK_QUEUE_DDL)
        # PRAGMAs are already set on connection creation; executescript may
        # have changed journal_mode, so re-assert it.
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for atomic multi-step operations.

        Auto-commits on success, auto-rolls-back on exception.
        Useful when updating both ``task_queue`` and ``port_leases`` in one
        atomic step.
        """
        try:
            yield self.conn
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Close the shared connection and checkpoint the WAL."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Port lease operations ────────────────────────────────────────────

    def _acquire_lease(self, port: int, task_id: int) -> bool:
        """Acquire a port lease (internal — no auto-commit).

        Used within transactions. For external callers, use ``acquire_lease``.
        """
        try:
            self.conn.execute(
                "INSERT INTO port_leases (port, task_id, leased_at) VALUES (?, ?, datetime('now', 'utc'))",
                (port, task_id),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def acquire_lease(self, port: int, task_id: int) -> bool:
        """Acquire a port lease atomically.

        Returns True if the lease was acquired, False if the port is already
        leased to another task.

        Commits the transaction. For use within a transaction, use ``_acquire_lease``.
        """
        result = self._acquire_lease(port, task_id)
        if result:
            self.conn.commit()
        return result

    def release_lease(self, port: int) -> None:
        """Release a port lease.

        Commits the transaction. For use within a transaction, use ``_release_lease``.
        """
        self._release_lease(port)
        self.conn.commit()

    def _release_lease(self, port: int) -> None:
        """Release a port lease (internal — no auto-commit).

        Used within transactions.
        """
        self.conn.execute("DELETE FROM port_leases WHERE port = ?", (port,))

    def find_stale_leases(self, max_age_seconds: int = 300) -> list[tuple[int, int, str]]:
        """Find leases older than *max_age_seconds*.

        Returns list of ``(port, task_id, leased_at)`` tuples.

        Used by the background scanner to detect orphaned ports after
        unclean process deaths.
        """
        rows = self.conn.execute(
            """
            SELECT pl.port, pl.task_id, pl.leased_at
            FROM port_leases pl
            WHERE (julianday('now') - julianday(pl.leased_at)) * 86400 > ?
              AND pl.task_id NOT IN (
                  SELECT id FROM task_queue WHERE status = 'processing'
              )
            """,
            (max_age_seconds,),
        ).fetchall()
        return [(r["port"], r["task_id"], r["leased_at"]) for r in rows]

    # ── Task queue operations ────────────────────────────────────────────

    def insert_task(
        self,
        dag_id: str,
        target_model_key: str | None = None,
        spawn_args: SpawnArgs | None = None,
        payload: TaskPayload | None = None,
        parent_id: int | None = None,
        in_degree: int = 0,
        assigned_port: int | None = None,
    ) -> int:
        """Insert a new task into the queue.

        Args:
            dag_id: Workflow DAG this task belongs to.
            target_model_key: Desired model (may change on retry).
            spawn_args: Structured spawn arguments (serialized to JSON).
            payload: Task input payload.
            parent_id: Optional parent task ID for DAG tracking.
            in_degree: Number of unresolved dependencies (0 if no parent).
            assigned_port: Port to assign (if pre-allocated by supervisor).

        Returns:
            The new task's ``id``.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO task_queue
                (dag_id, parent_id, in_degree, target_model_key, spawn_model_key,
                 spawn_args, assigned_port, payload, status)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?, 'pending')
            """,
            (
                dag_id,
                parent_id,
                in_degree,
                target_model_key,
                json.dumps(spawn_args.to_dict()) if spawn_args else None,
                assigned_port,
                payload.to_json() if payload else json.dumps({}),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_ready_tasks(self, limit: int = 10) -> list[sqlite3.Row]:
        """Get pending tasks with all dependencies met (in_degree == 0).

        Ordered by insertion time (FIFO).
        """
        return self.conn.execute(
            """
            SELECT * FROM task_queue
            WHERE status = 'pending' AND in_degree = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def mark_processing(self, task_id: int, port: int) -> None:
        """Mark a task as processing and assign a port.

        Commits the transaction. For use within a transaction, use ``_mark_processing``.
        """
        self._mark_processing(task_id, port)
        self.conn.commit()

    def _mark_processing(self, task_id: int, port: int) -> None:
        """Mark a task as processing and assign a port (internal — no auto-commit).

        Used within transactions.
        """
        self.conn.execute(
            "UPDATE task_queue SET status = 'processing', assigned_port = ?, updated_at = datetime('now', 'utc') WHERE id = ?",
            (port, task_id),
        )

    def mark_completed(
        self,
        task_id: int,
        output_json: str,
        spawn_model_key: str | None = None,
    ) -> None:
        """Mark a task completed and store its output."""
        self.conn.execute(
            """
            UPDATE task_queue
            SET status = 'completed', output = ?, spawn_model_key = COALESCE(?, spawn_model_key),
                updated_at = datetime('now', 'utc')
            WHERE id = ?
            """,
            (output_json, spawn_model_key, task_id),
        )
        self.conn.commit()

    def mark_failed(
        self,
        task_id: int,
        error_log: str,
        retry_count: int,
    ) -> None:
        """Mark a task as failed and update retry count."""
        self.conn.execute(
            """
            UPDATE task_queue
            SET status = 'failed', error_log = ?, retry_count = ?,
                updated_at = datetime('now', 'utc')
            WHERE id = ?
            """,
            (error_log, retry_count, task_id),
        )
        self.conn.commit()

    def decrement_children_in_degree(self, completed_task_id: int) -> None:
        """Decrement in_degree for all tasks that depend on *completed_task_id*.

        Called when a parent task transitions to 'completed'.
        """
        self.conn.execute(
            """
            UPDATE task_queue
            SET in_degree = MAX(in_degree - 1, 0),
                updated_at = datetime('now', 'utc')
            WHERE parent_id = ?
            """,
            (completed_task_id,),
        )
        self.conn.commit()

    def get_task(self, task_id: int) -> sqlite3.Row | None:
        """Fetch a single task by ID."""
        return self.conn.execute("SELECT * FROM task_queue WHERE id = ?", (task_id,)).fetchone()

    def update_spawn_args(
        self,
        task_id: int,
        spawn_args: SpawnArgs,
        target_model_key: str | None = None,
    ) -> None:
        """Update spawn_args for a task (used during escalation).

        The supervisor calls this when retrying with a different model config.

        Args:
            task_id: Task to update.
            spawn_args: New structured spawn arguments.
            target_model_key: Updated target model key (defaults to the
                ``spawn_args.model_path`` if provided, otherwise leaves it
                unchanged).
        """
        if target_model_key is None and spawn_args.model_path is not None:
            target_model_key = spawn_args.model_path

        if target_model_key is not None:
            self.conn.execute(
                """
                UPDATE task_queue
                SET spawn_args = ?, target_model_key = ?, status = 'pending',
                    spawn_model_key = NULL, updated_at = datetime('now', 'utc')
                WHERE id = ?
                """,
                (json.dumps(spawn_args.to_dict()), target_model_key, task_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE task_queue
                SET spawn_args = ?, status = 'pending',
                    spawn_model_key = NULL, updated_at = datetime('now', 'utc')
                WHERE id = ?
                """,
                (json.dumps(spawn_args.to_dict()), task_id),
            )
        self.conn.commit()

    def get_pending_by_dag(self, dag_id: str) -> list[sqlite3.Row]:
        """Get all pending/processing tasks for a DAG (for lifecycle tracking)."""
        return self.conn.execute(
            "SELECT * FROM task_queue WHERE dag_id = ? AND status IN ('pending', 'processing')",
            (dag_id,),
        ).fetchall()

    def cleanup_completed_dag(self, dag_id: str) -> int:
        """Delete all completed tasks for a DAG.

        Returns the number of rows deleted.

        Call after all tasks in a DAG are completed to reclaim space.
        """
        cursor = self.conn.execute(
            "DELETE FROM task_queue WHERE dag_id = ? AND status = 'completed'",
            (dag_id,),
        )
        self.conn.commit()
        return cursor.rowcount

    # ── Port allocation helpers ──────────────────────────────────────────

    def next_available_port(self, start: int = 60000, end: int = 65535) -> int:
        """Find the next free port in the reserved range.

        Scans from *start* upward, skipping leased ports, and wraps around
        to *start* if the entire range is exhausted.

        Args:
            start: Lower bound of port range (default 60000).
            end: Upper bound of port range (default 65535).

        Returns:
            A port number that is not currently leased.

        Raises:
            RuntimeError: If no ports are available in the range.
        """
        leased = self.conn.execute("SELECT port FROM port_leases").fetchall()
        leased_ports = {r["port"] for r in leased}

        for port in range(start, end + 1):
            if port not in leased_ports:
                return port

        raise RuntimeError(
            f"No available ports in range [{start}, {end}] — "
            f"{len(leased_ports)} ports currently leased."
        )

    # ── Atomic task start ────────────────────────────────────────────────

    def start_task(self, task_id: int, port: int) -> bool:
        """Atomically start a task: mark processing + assign port + acquire lease.

        All operations happen in a single SQLite transaction to prevent
        race conditions between concurrent lifecycle managers. If any step
        fails, the transaction rolls back and the task remains in its original
        state.

        Args:
            task_id: The task to start.
            port: The port to assign.

        Returns:
            True if the task was started successfully, False if the port is
            already leased to another task.
        """
        try:
            with self.transaction():
                # 1. Mark as processing and assign port
                self.conn.execute(
                    "UPDATE task_queue SET status = 'processing', assigned_port = ?, "
                    "updated_at = datetime('now', 'utc') WHERE id = ? AND status = 'pending'",
                    (port, task_id),
                )
                # 2. Acquire the lease (fails if port already taken)
                self.conn.execute(
                    "INSERT INTO port_leases (port, task_id, leased_at) VALUES (?, ?, datetime('now', 'utc'))",
                    (port, task_id),
                )
            # If we get here, the transaction committed successfully
            return True
        except sqlite3.IntegrityError:
            # Lease conflict — port already taken
            return False

    def release_task(self, task_id: int) -> None:
        """Release a task: remove port lease and reset status.

        Called on task failure to undo the start_task() atomically.

        Args:
            task_id: The task to release.
        """
        row = self.get_task(task_id)
        if row is None:
            return
        port = row["assigned_port"]
        if port is not None:
            self._release_lease(port)
        self.conn.execute(
            "UPDATE task_queue SET status = 'pending', assigned_port = NULL, "
            "updated_at = datetime('now', 'utc') WHERE id = ?",
            (task_id,),
        )
        self.conn.commit()
