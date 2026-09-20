# Pydantask Resource-Aware Execution & Dynamic Model Routing (v0.1 Spec - Revised)

**Spec Changelog:** [CHANGELOG.md](CHANGELOG.md) — tracks what changed, why, and how it affects implementation order.

> **Session Instruction:** Before reading any other part of this spec, review [CHANGELOG.md](CHANGELOG.md) first. It contains the reasoning behind structural decisions (phase ordering, schema changes, moved components) that are not self-evident from the spec text alone. Understanding the changelog prevents reverting intentional changes and helps assess whether the spec has evolved past the current implementation state.

---

**Implementation Status:**

| Phase | Description | Status |
|-------|-------------|--------|
| 1.1 | Core State, Registry & Skill Foundation | **✅ Done** — `src/pydantask/execution/schema.py`, `src/pydantask/execution/registry.py`, `src/pydantask/skills/__init__.py`, 74 tests |
| 1.2 | Model Registry & Structured Skill Schema | **✅ Done** — `skills/llama_cpp_skill.json`, `skills/llama_cpp.md`, `src/pydantask/execution/registry.py`, 38 tests |
| 2 | DAG Resolution & Port Allocation | **✅ Done** — `src/pydantask/execution/resolver.py`, `src/pydantask/execution/resources.py`, `QueueStore` atomic ops, 46 tests |
| 3 | Dynamic Lifecycle & Process Management (with Warm Standby) | ⬜ Pending |
| 4 | Sandboxing & Execution Isolation | ⬜ Pending |
| 5 | Supervisor Routing & Self-Healing Loop | ⬜ Pending |
| 6 | Performance Optimization | ⬜ Pending |


## 1. Overview

A local-first execution backend for Pydantask designed to operate under strict hardware constraints (e.g., single-machine 64GB unified memory limits). Decouples task planning from task execution via an SQLite event queue. Instead of a monolithic model resident in memory indefinitely, the supervisor determines resource profiles, queries live system metrics, and dynamically spawns, isolates, and terminates localized execution workloads on demand.

---

## 2. Core Architecture Components

### A. The Model Registry & Skill Reference

The model registry contains baseline paths, default profiles, and structured capability metadata. A local `skills/llama_cpp.md` file provides the supervisor with the necessary context on `llama.cpp` CLI arguments (e.g., `--ctx-size`, `--mmap`, `--n-gpu-layers`, `--port`) so it can intelligently adapt startup commands when needed.

* **Model Registry Entry Structure:**
```json
{
  "model_key": "qwen2.5-coder-7b",
  "path": "/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
  "default_port": 8082,
  "default_ctx": 8192,
  "estimated_vram_mb": 4200,
  "capability_tier": "coder",
  "min_memory_headroom_mb": 2000
}

```

The registry is the single source of truth for "what models exist and what they can do." The supervisor consults `estimated_vram_mb` and `min_memory_headroom_mb` when allocating tasks against available memory.

* **Structured Skill Definition Schema (`skills/llama_cpp_skill.json`):**
To avoid relying on the supervisor to parse prose for valid CLI arguments, skill definitions are stored as structured JSON alongside the prose reference. The supervisor reads this schema to validate every command it assembles before queue insertion.
```json
{
  "engine": "llama-server",
  "required_flags": ["--model", "--port"],
  "optional_flags": {
    "--ctx-size": {"type": "integer", "default": 4096, "note": "context window length"},
    "--mmap": {"type": "boolean", "default": true, "note": "memory-map the model file (recommended)"},
    "--n-gpu-layers": {"type": "integer", "default": 99, "note": "layers to offload to Metal on Apple Silicon"},
    "--temp": {"type": "number", "default": 0.7},
    "--top-p": {"type": "number", "default": 0.9}
  },
  "constraints": {
    "port_range": [60000, 65535],
    "max_concurrent_per_port": 1
  }
}

```

* **`skills/llama_cpp.md` Snippet (Loaded into Supervisor Context — Human-Readable Reference):**
> When spawning local execution instances via `llama-server`, optimize for M-series unified memory using `--mmap true` to ensure instant mapping. Set context size using `--ctx-size <val>` according to task requirements. Always assign an open local port via `--port <port>`. Terminate processes immediately after task batch completion.


### B. SQLite-Backed Persistent Queue

An ACID-compliant, file-based state store utilizing WAL mode to safely manage process handoffs, ensure crash resilience, and track independent task DAG nodes.

* **Table Schema (`task_queue`):**
* `id`: INTEGER PRIMARY KEY AUTOINCREMENT
* `dag_id`: TEXT
* `parent_id`: INTEGER (nullable, for dependency tracking)
* `in_degree`: INTEGER DEFAULT 0 (tracks number of unresolved dependencies for DAG resolution)
* `target_model_key`: TEXT (references registry)
* `spawn_model_key`: TEXT (nullable — the actual model used at execution time; may differ from `target_model_key` after escalation)
* `spawn_args`: JSON (structured spawn arguments, not a raw command string. Keys: model path, ctx_size, n_gpu_layers, mmap, port, temp, top_p, etc. The lifecycle manager assembles the CLI from this.)
* `assigned_port`: INTEGER
* `payload`: JSON (system prompts, input data, constraints)
* `output`: JSON (nullable — stores task result when completed; consumed by downstream DAG nodes)
* `status`: TEXT (`pending`, `processing`, `completed`, `failed`)
* `retry_count`: INTEGER DEFAULT 0
* `error_log`: TEXT (stores critiques or execution errors)

**Why structured `spawn_args` over raw `spawn_command`**: Storing arguments as structured JSON enables inspection, validation, and programmatic modification (e.g., swapping model tiers during escalation) without string parsing. The lifecycle manager assembles the CLI command at runtime.

### C. Port Allocation Strategy

Ports are allocated from a reserved range (60000–65535, matching the skill definition constraint) using a lease mechanism to prevent race conditions:

1. **Atomic Lease**: Before a task can be executed, the lifecycle manager attempts an atomic UPDATE on the task row to set `assigned_port` and `status='processing'`. Only one worker wins this race — SQLite row-level locking handles this in WAL mode.
2. **Lease Table (`port_leases`)**: A companion table tracks active leases: `port INTEGER PRIMARY KEY`, `task_id INTEGER`, `leased_at TEXT`. The lifecycle manager inserts a lease row inside the same transaction as the status update, ensuring port and task status are always consistent.
3. **Lease Release**: On task completion or failure, the transaction deletes the lease row and resets the task status. If the process dies uncleanly, a background health-check scanner periodically scans for stale leases (port bound but no matching task in `processing` state) and purges them.

### D. DAG Dependency Resolution

The lifecycle manager must determine which `pending` task nodes are actually executable (all parents completed). This is handled by a dependency-resolution engine:

1. **In-Degree Tracking**: When the supervisor inserts a task with a `parent_id`, it sets `in_degree` to the number of unresolved parents. When a parent task transitions to `completed`, the manager decrements `in_degree` for all its direct children (via a cascade UPDATE triggered by the parent's status change).
2. **Ready Queue**: Tasks with `status='pending'` AND `in_degree=0` are "ready" and eligible for execution. The lifecycle manager polls for these.
3. **Cycle Detection**: On DAG construction, the supervisor runs a topological sort validation. If a cycle is detected (in_degree never reaches 0 for some nodes), the entire DAG is rejected before queue insertion.

### E. Result Persistence & Output Retrieval

Task outputs are stored in the `output` JSON column and must be retrievable by downstream consumers:

1. **Output Contract**: Every completed task writes its result to the `output` column. The output schema is task-dependent but must include at minimum: `{"status": "ok"|"error", "content": "<result>", "model_used": "<spawn_model_key>", "tokens_used": <int>}`.
2. **Output Resolution by DAG Nodes**: When resolving dependencies, downstream nodes that reference a parent's output read it from the `output` column of the completed parent task row (via `dag_id` and `parent_id` join).
3. **Output TTL**: Completed task outputs are retained until the entire DAG (identified by `dag_id`) is fully resolved. A cleanup job periodically purges old completed DAGs.

### F. System Resource Check Tool (`get_system_resources`)

A runtime introspection tool exposed to the supervisor or handled deterministically by the task runner using `psutil`. It queries active host metrics before provisioning tasks to prevent OOM events and determine whether to spin up new instances or reuse active hot-residents.

* **Metrics Captured:**
* Available physical/unified memory (`psutil.virtual_memory().available`).
* Current memory pressure/percentage.
* Active lifecycle manager port allocations and resident model states.

### G. Execution Sandboxing & Work Containers

To prevent side effects, protect host system integrity, and safely handle generated code execution or untrusted file processing, task execution is wrapped in a lightweight isolation layer:

* **Process-Level Isolation:** Workers execute within strict boundary constraints (e.g., constrained working directories, restricted file system access, and stripped environment variables).
* **Pluggable Sandboxing:** Supports native OS jailing (such as macOS `sandbox-exec` profiles) or containerized runtimes (lightweight Docker/Podman or restricted Python execution interpreters) depending on whether the task involves executing arbitrary code snippets.

### H. Supervisor-Driven Command Assembly & Scheduling

When the supervisor breaks down a workflow plan, it executes a resource check and populates the task queue item with structured spawn arguments:

1. **Intelligent Allocation:** Supervisor checks available headroom via `get_system_resources` against each candidate model's `estimated_vram_mb` and `min_memory_headroom_mb` from the registry. If memory is flush, it queues parallel tasks; if memory is tight, it schedules serial execution or selects a smaller model profile.
2. **Structured Argument Assembly:** Supervisor writes validated `spawn_args` (JSON object with keys like `model_path`, `ctx_size`, `n_gpu_layers`, `mmap`, `temp`, etc.) into the task row. These arguments are validated against the structured skill schema (`skills/llama_cpp_skill.json`) before insertion — not assembled from prose parsing.
3. **Fallback:** If custom flags are omitted, the assembly logic fills in defaults from the registry's `MODEL_REGISTRY` entry for that `target_model_key`.

### I. The Dynamic Lifecycle Manager

An asynchronous Python subprocess supervisor that handles resource provisioning:

1. **Queue Polling:** Inspects SQLite for tasks with `status='pending'` AND `in_degree=0` (all upstream dependencies cleared).
2. **Port Lease & Process Spawn:** Acquires an atomic port lease (same transaction as status update to `processing`), reads the structured `spawn_args` field, assembles the `llama-server` CLI command at runtime, and executes it inside the designated work container/sandbox boundary via `asyncio.create_subprocess_exec`.
3. **Health Check & Execution:** Polls the local server endpoint (`/health`), fires the payload via HTTP, captures the response into the task's `output` column, and records `spawn_model_key` (which may differ from `target_model_key` after escalation).
4. **Teardown & Lease Release:** Issues a clean termination signal (`SIGTERM`) to the process, awaits process exit, deletes the port lease row, and resets the task status to `completed` or `failed`. The warm standby pool (Phase 3.8) bypasses teardown for the smallest model, keeping it resident with a context reset instead.

---

## 3. Self-Healing Critique & Escalation Loop

1. **Failure Capture:** If a worker node errors out or fails a critique check:
* Output, error logs, and critique notes are written back to the task row.
* `retry_count` increments.

2. **Supervisor Re-evaluation:**
* If retries remain, the task is routed back to the supervisor alongside current system resource statistics and critique feedback.
* **Escalation:** The supervisor can modify `target_model_key` and write updated `spawn_args` (e.g., swapping a fast 3B model out for a heavy 32B reasoner or coder model). The `spawn_model_key` field is set to the new model key, providing an audit trail of model swaps across retries.


# FEATURE WORK PHASES

## Phase 1: Core State, Registry & Skill Foundation

1. **SQLite WAL Queue Schema**: Implement an ACID-compliant SQLite state store utilizing Write-Ahead Logging (WAL) mode. Create the `task_queue` table supporting fields for `dag_id`, `parent_id`, `in_degree` (for DAG dependency resolution), `target_model_key`, `spawn_model_key`, `spawn_args` (structured JSON of CLI arguments — not a raw command string), `assigned_port`, `payload` (JSON), `output` (JSON, nullable — stores task result for downstream DAG consumption), `status`, `retry_count`, and `error_log`. Also create a `port_leases` table (`port INTEGER PRIMARY KEY`, `task_id INTEGER`, `leased_at TEXT`) for atomic port allocation.
2. **Model Registry & Structured Skill Schema**: Build a static `MODEL_REGISTRY` dictionary and strict Pydantic schemas (`TaskPayload`, `ModelConfig`) to validate task specifications. Write the structured skill definition (`skills/llama_cpp_skill.json`) containing all valid flags, their types, defaults, and constraints. The prose reference (`skills/llama_cpp.md`) supplements this but is not the source of truth — the JSON schema is. The supervisor validates every `spawn_args` object against this schema before queue insertion.

## Phase 2: DAG Resolution & Port Allocation

3. **Dependency Resolution Engine**: Implement in-degree tracking for DAG nodes. When the supervisor inserts a task with a `parent_id`, it sets `in_degree` to the number of unresolved parents. Implement a cascade mechanism that decrements children's `in_degree` when a parent completes. Tasks with `status='pending'` AND `in_degree=0` are "ready" for execution. On DAG construction, run a topological sort validation — reject any DAG with cycles before queue insertion.
4. **Atomic Port Lease Mechanism**: Implement port allocation from the reserved range (60000–65535). The lifecycle manager performs an atomic UPDATE to set `assigned_port` and `status='processing'` while inserting a `port_leases` row in the same SQLite transaction. On task completion or failure, delete the lease row. Implement a background stale-lease scanner that periodically detects orphaned port bindings (port bound but no matching `processing` task) and purges them.
5. **System Resource Tool (`get_system_resources`)**: Write a Python introspection utility using `psutil` to query available physical/unified memory, active memory pressure, and current lifecycle manager port allocations, making live metrics accessible to the supervisor prior to scheduling. The registry's `estimated_vram_mb` and `min_memory_headroom_mb` fields are used to determine whether a given model can be spawned.

## Phase 3: Dynamic Lifecycle & Process Management (with Warm Standby)

6. **Async Subprocess Lifecycle Manager**: Implement an `asyncio`-driven manager capable of reading structured `spawn_args` from task rows, assembling the `llama-server` CLI command at runtime, launching instances via `asyncio.create_subprocess_exec`, and tracking active processes mapped to ports (via the `port_leases` table).
7. **Health-Check & Teardown Handlers**: Build asynchronous polling loops to verify local server readiness (`/health` endpoints) before dispatching HTTP payloads, alongside clean teardown logic (`SIGTERM`, process exit awaiting, and memory-flush verification) to return resources to macOS.
8. **Warm Standby Pool (moved from Phase 6)**: Given that spawn/teardown cycles can cost 10–30 seconds each, implement a minimal hot-resident model pool. The smallest/fastest model in the registry is kept alive between task executions. A context-reset mechanism (clearing the KV cache) is used between tasks instead of full teardown. This eliminates cold-start latency for sequential micro-tasks. The pool size is capped at one model to minimize baseline memory pressure — additional models are spawned on-demand for parallel tasks.

## Phase 4: Sandboxing & Execution Isolation

9. **Native macOS `sandbox-exec` Integration**: Create a sandboxing wrapper using macOS native `sandbox-exec` profiles to isolate task execution environments, restricting unauthorized file system or network access at the kernel level without VM overhead.
10. **RAM Disk Workspace Manager**: Implement an optional temporary workspace utility using `hdiutil` to mount and clear high-speed RAM disks (`tmpfs`-equivalent) for tasks requiring heavy file read/write operations.

## Phase 5: Supervisor Routing & Self-Healing Loop

11. **Dynamic Command Assembly**: Build supervisor logic that evaluates resource availability via `get_system_resources`, consults the model registry (`estimated_vram_mb`, `capability_tier`) and the structured skill schema, and writes validated `spawn_args` into task queue rows. If memory is tight, the supervisor schedules serial execution or drops to a smaller model profile. Fallback to registry defaults when custom flags are omitted.
12. **Critique-and-Retry Escalation Loop**: Implement a failure-capture mechanism that logs worker errors and critique feedback, increments `retry_count`, and routes tasks back to the supervisor to dynamically escalate model tiers (e.g., swapping a 3B model for a 32B model). On escalation, `target_model_key` is updated and `spawn_model_key` records the actual model used at each retry attempt.

## Phase 6: Performance Optimization

13. **Apple Silicon Optimization Flags**: Configure baseline command templates to enforce full Metal acceleration (`--n-gpu-layers 99`) and instant memory-mapping (`--mmap true`).
14. **Result Output Contract & Cleanup**: Enforce the output schema defined in Section 2.E (status, content, model_used, tokens_used). Implement a DAG lifecycle manager that purges completed DAG outputs once all nodes in a `dag_id` are resolved, and a periodic cleanup job that removes old completed tasks and their outputs.