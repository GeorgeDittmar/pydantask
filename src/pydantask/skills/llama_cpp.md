# llama.cpp Execution Skill — Supervisor Reference

> **NOTE:** This is the human-readable reference. The structured skill schema
> (`skills/llama_cpp_skill.json`) is the authoritative source of truth. Always
> validate spawn arguments against the JSON schema before queue insertion.

## Overview

When spawning local execution instances via `llama-server`, optimize for
M-series unified memory using `--mmap true` to ensure instant mapping. Set
context size using `--ctx-size <val>` according to task requirements. Always
assign an open local port via `--port <port>`. Terminate processes immediately
after task batch completion.

## Key Flags

- `--model <path>` — Required. Path to the GGUF model file.
- `--port <port>` — Required. Local port for the server instance. Use ports
  60000–65535 to avoid conflicts with system services.
- `--ctx-size <val>` — Context window length. Default 4096. Increase for
  tasks that need large context (code generation, long document analysis).
- `--mmap true` — Memory-map the model file. Always use this on M-series
  chips for instant loading and efficient memory usage.
- `--n-gpu-layers <val>` — Layers to offload to Metal (Apple Silicon GPU).
  Default 99 (all layers). Set to 0 for CPU-only mode.
- `--temp <val>` — Sampling temperature. Default 0.7. Lower values produce
  more deterministic output (good for reasoning tasks).
- `--top-p <val>` — Nucleus sampling threshold. Default 0.9. Controls the
  diversity of generated tokens.

## Memory Guidelines

| Model Size | Approx. VRAM (4-bit) | Recommended ctx-size |
|------------|----------------------|---------------------|
| 1B–3B      | 0.5–1.5 GB           | 4096                |
| 7B–8B      | 3–5 GB               | 8192                |
| 13B–14B    | 7–9 GB               | 8192                |
| 30B–34B    | 18–22 GB             | 4096                |
| 70B+       | 40+ GB               | 2048                |

On a 64GB unified memory machine, keep at least 4GB headroom for the OS and
llama-server overhead. If memory pressure exceeds 85%, drop to a smaller model
or reduce context size.

## Lifecycle

1. Supervisor selects a model from the registry based on task complexity and
   available memory (checked via `get_system_resources`).
2. Supervisor assembles a `SpawnArgs` object and validates it against the
   skill schema.
3. Task is queued with the structured spawn arguments.
4. The lifecycle manager spawns `llama-server` using the arguments.
5. After execution, the process is terminated with SIGTERM.
6. The port is released back to the pool.

## Warm Standby

The smallest model in the registry is kept hot between tasks via context
reset (KV cache clear) instead of full teardown. This eliminates cold-start
latency (~10–30s) for sequential micro-tasks.
