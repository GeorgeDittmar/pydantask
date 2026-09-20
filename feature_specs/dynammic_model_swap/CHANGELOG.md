# Spec Changelog — Dynamic Model Routing

All notable changes to this spec.

---

## [Unreleased] — Structuring, DAG Resolution & Port Safety

> **Date:** 2025-09-xx
> **Trigger:** Code review of the initial spec identified underspecified areas (port allocation, DAG dependency resolution, result persistence), architectural risks (cold-start latency), and fragility (prose-only skill definitions). Changes address these before implementation begins.

### Phase 1.1 — SQLite Queue Schema & Pydantic Models (DONE)

**Date:** 2025-09-xx
**Scope:** Phase 1.1 from spec — database schema, Pydantic models, QueueStore

**Files created:**
- `src/pydantask/execution/__init__.py` — package exports
- `src/pydantask/execution/schema.py` (~590 lines) — DDL, Pydantic models, QueueStore
- `test/test_execution_schema.py` — 36 tests across 10 classes

**What's in it:**
- `task_queue` DDL with 14 columns including `dag_id`, `in_degree`, `spawn_args` (JSON), `output`, `spawn_model_key`, indexes
- `port_leases` DDL with port/task/leased_at columns
- Pydantic models: `ModelConfig`, `SpawnArgs`, `TaskPayload`, `TaskOutput`
- `QueueStore`: shared-connection SQLite wrapper with WAL mode, transaction context manager, port lease CRUD, task CRUD, DAG resolution (`in_degree` tracking, ready task polling, child unblocking), escalation support (`update_spawn_args`), cleanup

**Design decisions:**
- Shared connection instead of open/close per operation — avoids SQLite "database is locked" errors from rapid connection churn in WAL mode
- Structured `spawn_args` JSON over raw `spawn_command` TEXT — enables inspection, validation, and programmatic modification
- `SpawnArgs.from_dict()` uses `model_fields.keys()` (not `.alias`) — `.alias` is `None` when no explicit alias is set

### Phase 1.2 — Model Registry & Structured Skill Schema (DONE)

**Date:** 2025-09-xx
**Scope:** Phase 1.2 from spec — model registry, structured skill definition, skill validation, CLI assembly

**Files created:**
- `src/pydantask/skills/llama_cpp_skill.json` — structured skill schema (authoritative source of truth)
- `src/pydantask/skills/llama_cpp.md` — human-readable supervisor reference
- `src/pydantask/skills/__init__.py` (~230 lines) — `SkillRegistry`, `SkillSchema`, validation, CLI assembly
- `src/pydantask/execution/registry.py` (~186 lines) — `MODEL_REGISTRY` dict with 9 models, lookup helpers
- `test/test_execution_registry.py` — 38 tests across 8 test classes

**What's in it:**
- **Skill schema** (`llama_cpp_skill.json`): Engine config, required/optional flags with types, defaults, min/max bounds, port range constraints, health check settings. Each optional flag maps to a `SpawnArgs` field via `spawn_args_field`.
- **Skill registry** (`SkillRegistry`): Auto-discovers `*_skill.json` files from the skills directory. Validates `SpawnArgs` against skill schema (type, bounds, constraints). Assembles `llama-server` CLI commands from structured args.
- **Model registry** (`MODEL_REGISTRY`): 9 pre-configured models across 4 tiers (`fast`, `coder`, `general`, `reasoner`). Each has `estimated_vram_mb`, `min_memory_headroom_mb`, `total_required_mb`, default port (in 60000+ range). Helper functions: `get_model()`, `find_models_by_tier()`, `find_fit_models()`, `get_smallest_model()`.
- **CLI assembly** (`assemble_llama_server_command()`): Converts structured `SpawnArgs` + port into a `list[str]` for `asyncio.create_subprocess_exec`. Omits default-valued flags to keep commands short.

**Validation layers (belt-and-suspenders):**
1. Pydantic `ge`/`le` on `SpawnArgs` fields catches out-of-range values at construction time
2. `SkillRegistry.validate_spawn_args()` re-checks bounds against the skill schema at validation time
3. Both layers must pass before a task can be queued

**Design decisions:**
- Skill schema is JSON (not prose) — machine-parseable, validateable, extensible
- Prose reference (`llama_cpp.md`) supplements but does not replace the schema
- Ports in `MODEL_REGISTRY` use the 60000-65535 range (matches skill constraint)
- Command assembly omits default-valued flags — shorter, faster CLI invocation
- `default_registry` singleton — single shared instance across supervisor and lifecycle manager

### Phase 2 — DAG Resolution & Port Allocation (DONE)

**Date:** 2025-09-xx
**Scope:** Phase 2 from spec — dependency resolution, atomic port allocation, system resource introspection

**Files created:**
- `src/pydantask/execution/resolver.py` (~240 lines) — DAG cycle detection (Kahn's algorithm), ready task resolution, parent output resolution, DAG status/completion checks
- `src/pydantask/execution/resources.py` (~220 lines) — System resource introspection via psutil, memory pressure throttling
- `src/pydantask/execution/schema.py` — enhanced QueueStore with atomic `start_task()`, `release_task()`, `next_available_port()`
- `test/test_execution_resolver.py` — 31 tests across 7 classes
- `test/test_execution_resources.py` — 16 tests across 4 classes

**What's in it:**
- **DAG cycle detection** (`validate_dag()`): Kahn's algorithm (BFS-based topological sort). Returns `(True, topological_order)` for valid DAGs or `(False, error_message)` identifying cycle nodes. Handles self-cycles, partial cycles, and unknown parent references.
- **Ready task resolution** (`resolve_ready_tasks()`): Returns task IDs where `status='pending'` AND `in_degree=0`, ordered by insertion time (FIFO).
- **Parent output resolution** (`resolve_parent_output()`): Follows the `parent_id` chain and returns the parent task's `output` JSON for downstream consumption.
- **DAG status** (`get_dag_status()`, `is_dag_complete()`): Aggregates task counts by status per DAG and checks if all tasks have reached a terminal state.
- **Atomic port allocation** (`next_available_port()`, `start_task()`): `next_available_port()` scans for the next free port in 60000–65535. `start_task()` atomically updates task status to 'processing', assigns the port, and acquires the lease in a single SQLite transaction — on failure, the entire transaction rolls back cleanly.
- **System resources** (`get_system_resources()`): Queries available memory, memory pressure percentage, active port leases, and parallel capacity. Returns `can_fit_model()` and `best_fitting_model()` helpers.
- **Memory throttling** (`should_throttle_pressure()`): Returns True when memory pressure exceeds 85%, signaling the supervisor to prefer serial execution.

**Key design decisions:**
- `start_task()` uses a single `transaction()` context manager — if the lease acquisition fails, `_mark_processing`'s changes are rolled back too
- `should_throttle_pressure()` uses `_get_memory_metrics()` for consistency with `get_system_resources()` (not direct psutil calls)
- Psutil is optional — all resource functions gracefully degrade to zeroed metrics when unavailable

### Added

- **Port Allocation Strategy (Section 2.C)**
  - Reserved port range: 60000–65535 (mirrors skill definition constraint).
  - Atomic lease mechanism: `assigned_port` + `status='processing'` updated in one SQLite transaction, paired with a `port_leases` table (`port`, `task_id`, `leased_at`).
  - Background stale-lease scanner: detects orphaned port bindings (bound port but no matching `processing` task) and purges them after unclean process deaths.
  - **Why:** Two tasks racing to pick the same port would cause bind failures. The lease table makes allocation deterministic and crash-resilient.

- **DAG Dependency Resolution (Section 2.D)**
  - `in_degree` integer column on `task_queue` tracks unresolved parent count.
  - Cascade UPDATE: when a parent transitions to `completed`, all children's `in_degree` decrements.
  - Ready queue: only tasks with `status='pending'` AND `in_degree=0` are polled by the lifecycle manager.
  - Topological sort validation on DAG construction: cycles are rejected before queue insertion.
  - **Why:** `parent_id` existed in the original schema but no algorithm described how the lifecycle manager determines which pending nodes are executable. This closes that gap.

- **Result Persistence & Output Retrieval (Section 2.E)**
  - `output` JSON column on `task_queue` stores task results.
  - Output contract: minimum schema — `{"status": "ok"|"error", "content": "<result>", "model_used": "<key>", "tokens_used": <int>}`.
  - Downstream DAG nodes read parent outputs via `dag_id` + `parent_id` join.
  - Output TTL: retained until entire DAG resolves; periodic cleanup job purges old DAGs.
  - **Why:** The original spec said the lifecycle manager "captures results" but didn't specify where they go or how downstream tasks consume them.

- **Structured Skill Definition Schema (`skills/llama_cpp_skill.json`)**
  - Typed flag definitions with defaults, types, and constraints (port range, etc.).
  - Placed alongside the prose reference (`skills/llama_cpp.md`) but is the source of truth.
  - **Why:** Relying on the supervisor's LLM to parse markdown prose for valid CLI args is fragile — ambiguity in the prose leads to malformed invocations. Structured JSON is inspectable, validateable, and modifiable by code.

- **`port_leases` table**
  - Schema: `port INTEGER PRIMARY KEY`, `task_id INTEGER`, `leased_at TEXT`.
  - **Why:** Decoupled from `task_queue` to enable independent stale-lease scanning without touching task state.

### Changed

- **`task_queue` schema:**
  - Added `in_degree INTEGER DEFAULT 0`
  - Added `spawn_model_key TEXT` (nullable — records the actual model used at execution; may differ from `target_model_key` after escalation)
  - Added `spawn_args JSON` (structured CLI arguments — replaces raw `spawn_command TEXT`)
  - Added `output JSON` (nullable — task result for downstream DAG consumption)
  - **Why `spawn_args` over `spawn_command`:** Structured JSON enables inspection, validation, and programmatic modification (e.g., swapping model tiers during escalation) without string parsing.

- **Model Registry Entry:** Added `estimated_vram_mb`, `capability_tier`, `min_memory_headroom_mb` fields.
  - **Why:** The supervisor needs quantitative data to allocate tasks against available memory.

- **Section lettering:** Re-sequenced from A→I (was A→F, then D→F re-used letters). No content moved, only labels renumbered for clarity.

- **Self-Healing Escalation (Section 3):** Updated to reference `spawn_args` (structured) instead of `spawn_command`, and added that `spawn_model_key` records the actual model at each retry for auditability.
  - **Why:** Consistency with the new structured schema and provides traceability across model swaps.

- **Lifecycle Manager (Section 2.I):** Updated to reference `port_leases` table, `in_degree` checking, structured `spawn_args`, and `spawn_model_key` recording.
  - **Why:** The manager is the central executor — its description must reflect all the new infrastructure around it.

- **Supervisor Command Assembly (Section 2.H):** Updated to describe structured argument assembly validated against the JSON skill schema, with registry fallbacks.
  - **Why:** No longer "writes custom spawn arguments based on skill.md rules" — now validates against the JSON schema before insertion.

### Reordered

- **Warm Standby Pool:** Moved from **Phase 6 (Performance Optimization)** to **Phase 3.8 (Dynamic Lifecycle & Process Management)**.
  - **Why:** Spawn/teardown cycles cost 10–30 seconds each. Without a hot-resident model, Phases 3–5 produce a system that's unusably slow. This is a correctness-of-experience requirement, not a post-hoc optimization.

- **DAG Resolution & Port Allocation:** Promoted to **Phase 2** (before lifecycle management).
  - **Why:** These are foundational to the lifecycle manager — you can't implement polling (`in_degree=0`) or spawning (port lease) without them first.

- **Structured Skill Schema:** Promoted from Phase 5 to **Phase 1.2**.
  - **Why:** "What models can we run and how do we run them" must be established before any scheduling logic is built.

- **Result Output Contract:** Added to **Phase 6.14** (was implicitly covered but not explicitly planned).

### Removed / Deprecated

- **`spawn_command` TEXT field** — replaced by `spawn_args JSON` + `spawn_model_key TEXT`.
- **Phase 5.8 (`skills/llama_cpp.md` Context Integration)** — merged into Phase 1.2 alongside the structured schema. The prose file still exists as a supplement but is no longer the primary skill definition mechanism.
- **Phase 6.12 (Warm Standby Pool)** — moved to Phase 3.8, not removed.

### Impact on Implementation Order

Original: Registry → Resources → Lifecycle → Sandboxing → Supervisor Routing → Performance
New:      Registry + Skill Schema → DAG + Port Safety → Lifecycle + Warm Standby → Sandboxing → Supervisor + Self-Healing → Output Cleanup

The new order reflects dependencies: you can't poll for ready tasks (Phase 3) without DAG resolution (Phase 2), and you can't build a usable system (Phase 3) without the warm pool (moved from Phase 6).

### Phase 1.1, 1.2 & 2 — Completed

Phases 1 and 2 are now fully implemented: the database schema, Pydantic models, QueueStore (with atomic operations), model registry, skill schema, validation, CLI assembly, DAG resolution engine, port allocation, and system resource introspection are all coded and tested (120 tests total, all passing). Phase 3 (Dynamic Lifecycle & Process Management) builds directly on these primitives.
