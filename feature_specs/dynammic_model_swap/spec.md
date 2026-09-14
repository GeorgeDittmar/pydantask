# Pydantask Resource-Aware Execution & Dynamic Model Routing (v0.1 Spec - Revised)

## 1. Overview

A local-first execution backend for Pydantask designed to operate under strict hardware constraints (e.g., single-machine 64GB unified memory limits). Decouples task planning from task execution via an SQLite event queue. Instead of a monolithic model resident in memory indefinitely, the supervisor determines resource profiles, queries live system metrics, and dynamically spawns, isolates, and terminates localized execution workloads on demand.

---

## 2. Core Architecture Components

### A. The Model Registry & Skill Reference (`skills/llama_cpp.md`)

The model registry contains baseline paths and default profiles, while a local `skills/llama_cpp.md` file provides the supervisor with the necessary context on `llama.cpp` CLI arguments (e.g., `--ctx-size`, `--mmap`, `--n-gpu-layers`, `--port`) so it can intelligently adapt startup commands when needed.

* **Model Registry Entry Structure:**
```json
{
  "model_key": "qwen2.5-coder-7b",
  "path": "/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
  "default_port": 8082,
  "default_ctx": 8192
}

```


* **`skills/llama_cpp.md` Snippet (Loaded into Supervisor Context):**
> When spawning local execution instances via `llama-server`, optimize for M-series unified memory using `--mmap true` to ensure instant mapping. Set context size using `--ctx-size <val>` according to task requirements. Always assign an open local port via `--port <port>`. Terminate processes immediately after task batch completion.



### B. SQLite-Backed Persistent Queue

An ACID-compliant, file-based state store utilizing WAL mode to safely manage process handoffs, ensure crash resilience, and track independent task DAG nodes.

* **Table Schema (`task_queue`):**
* `id`: INTEGER PRIMARY KEY AUTOINCREMENT
* `dag_id`: TEXT
* `parent_id`: INTEGER (nullable, for dependency tracking)
* `target_model_key`: TEXT (references registry)
* `spawn_command`: TEXT (The exact CLI command/args constructed dynamically by the supervisor or resolved from defaults)
* `assigned_port`: INTEGER
* `payload`: JSON (system prompts, input data, constraints)
* `status`: TEXT (`pending`, `processing`, `completed`, `failed`)
* `retry_count`: INTEGER DEFAULT 0
* `error_log`: TEXT (stores critiques or execution errors)



### C. System Resource Check Tool (`get_system_resources`)

A runtime introspection tool exposed to the supervisor or handled deterministically by the task runner using `psutil`. It queries active host metrics before provisioning tasks to prevent OOM events and determine whether to spin up new instances or reuse active hot-residents.

* **Metrics Captured:**
* Available physical/unified memory (`psutil.virtual_memory().available`).
* Current memory pressure/percentage.
* Active lifecycle manager port allocations and resident model states.



### D. Execution Sandboxing & Work Containers

To prevent side effects, protect host system integrity, and safely handle generated code execution or untrusted file processing, task execution is wrapped in a lightweight isolation layer:

* **Process-Level Isolation:** Workers execute within strict boundary constraints (e.g., constrained working directories, restricted file system access, and stripped environment variables).
* **Pluggable Sandboxing:** Supports native OS jailing (such as macOS `sandbox-exec` profiles) or containerized runtimes (lightweight Docker/Podman or restricted Python execution interpreters) depending on whether the task involves executing arbitrary code snippets.

### E. Supervisor-Driven Command Assembly & Scheduling

When the supervisor breaks down a workflow plan, it executes a resource check and populates the task queue item with a tailored execution command:

1. **Intelligent Allocation:** Supervisor checks available headroom via `get_system_resources`. If memory is flush, it queues parallel worker processes; if memory is tight, it schedules serial execution or drops to a smaller model profile.
2. **Dynamic Generation:** Supervisor writes custom spawn arguments into the task row based on `skill.md` rules.
3. **Fallbacking:** If custom flags are omitted, the execution engine falls back to the registry's default profile.

### F. The Dynamic Lifecycle Manager

An asynchronous Python subprocess supervisor that handles resource provisioning:

1. **Queue Polling:** Inspects SQLite for `pending` nodes with cleared upstream dependencies.
2. **Process Spawn:** Takes the `spawn_command` field, executes it inside the designated work container/sandbox boundary via `asyncio.create_subprocess_exec`, and tracks active processes by port.
3. **Health Check & Execution:** Polls the local server endpoint, fires the payload via HTTP, and captures results.
4. **Teardown:** Issues a clean termination signal (`SIGTERM`) to the process immediately after execution, completely flushing the model from RAM/VRAM and wiping temporary container workspace state.

---

## 3. Self-Healing Critique & Escalation Loop

1. **Failure Capture:** If a worker node errors out or fails a critique check:
* Output, error logs, and critique notes are written back to the task row.
* `retry_count` increments.


2. **Supervisor Re-evaluation:**
* If retries remain, the task is routed back to the supervisor alongside current system resource statistics and critique feedback.
* **Escalation:** The supervisor can modify the `target_model_key` and write an updated `spawn_command` (e.g., swapping a fast 3B model out for a heavy 32B reasoner or coder model) for the retry attempt.


# FEATURE WORK PHASES

## Phase 1: Core State & Registry Foundation

1. **SQLite WAL Queue Schema**: Implement an ACID-compliant SQLite state store utilizing Write-Ahead Logging (WAL) mode. Create the `task_queue` table supporting fields for `dag_id`, `parent_id` (for dependency tracking), `target_model_key`, `spawn_command`, `assigned_port`, `payload` (JSON), `status`, `retry_count`, and `error_log`.
2. **Model Registry & Schema Definitions**: Build a static `MODEL_REGISTRY` dictionary and strict Pydantic schemas (`TaskPayload`, `ModelConfig`) to validate task specifications and prevent model configuration hallucinations before queue insertion.

## Phase 2: Resource Introspection

3. **System Resource Tool (`get_system_resources`)**: Write a Python introspection utility using `psutil` to query available physical/unified memory, active memory pressure, and current lifecycle manager port allocations, making live metrics accessible to the supervisor prior to scheduling.

## Phase 3: Dynamic Lifecycle & Process Management

4. **Async Subprocess Lifecycle Manager**: Implement an `asyncio`-driven manager capable of parsing task spawn commands, launching `llama-server` instances via `asyncio.create_subprocess_exec`, and tracking active processes mapped to local ports.
5. **Health-Check & Teardown Handlers**: Build asynchronous polling loops to verify local server readiness (`/health` endpoints) before dispatching HTTP payloads, alongside clean teardown logic (`SIGTERM`, process exit awaiting, and memory-flush verification) to return resources to macOS.

## Phase 4: Sandboxing & Execution Isolation

6. **Native macOS `sandbox-exec` Integration**: Create a sandboxing wrapper using macOS native `sandbox-exec` profiles to isolate task execution environments, restricting unauthorized file system or network access at the kernel level without VM overhead.
7. **RAM Disk Workspace Manager**: Implement an optional temporary workspace utility using `hdiutil` to mount and clear high-speed RAM disks (`tmpfs`-equivalent) for tasks requiring heavy file read/write operations.

## Phase 5: Supervisor Routing & Self-Healing Loop

8. **`skills/llama_cpp.md` Context Integration**: Draft the reference guide detailing core `llama.cpp` arguments (`--ctx-size`, `--mmap`, `--n-gpu-layers`, `--port`) and inject it into the supervisor's system prompt context.
9. **Dynamic Command Assembly**: Build supervisor parsing logic that enables it to evaluate resource availability via `get_system_resources`, read `skills/llama_cpp.md`, and write tailored execution commands directly into task queue payloads (with fallback to registry defaults).
10. **Critique-and-Retry Escalation Loop**: Implement a failure-capture mechanism that logs worker errors and critique feedback, increments `retry_count`, and routes tasks back to the supervisor to dynamically escalate model tiers (e.g., swapping a 3B model for a 32B model) on subsequent attempts.

## Phase 6: Performance Optimization

11. **Apple Silicon Optimization Flags**: Configure baseline command templates to enforce full Metal acceleration (`--n-gpu-layers 99`) and instant memory-mapping (`--mmap true`).
12. **Warm Standby Pool**: Implement an idle state manager for frequent micro-models to keep hot-resident processes alive with context resets, bypassing cold-start overhead for repetitive sub-tasks.