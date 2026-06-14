# Observability and Tracing

Pydantask includes optional tracing support so you can inspect and debug
`DeepAgent` runs in external observability tools.

Supported backends (as implemented today):

- **Langfuse**
- **Logfire**
- **NONE** (no tracing)

Tracing is completely optional and is off by default.

---

## Enabling Tracing via `DeepAgent`

The simplest way to turn on tracing is to pass `trace=True` to `DeepAgent` and
set the appropriate environment variables for your chosen backend.

```python

from pydantask.agents import DeepAgent

agent = DeepAgent(
    objective="Research the best open source LLMs of 2024.",
    model="gpt-4.1-mini",
    trace=True,  # enable tracing with auto-detected backend
)

# Later, in your async code:
# result = await agent.run()
```

When `trace=True`:

- `DeepAgent` calls `init_tracing_backend(autodetect_tracing_backend())`.
- `autodetect_tracing_backend()` inspects environment variables and selects a
  backend (Langfuse, Logfire, or NONE).
- Key orchestration methods (such as `DeepAgent.run`) are traced and reported
  to the active backend.

Most users only need this setting plus the right environment variables.

---

## Backend Auto-detection

`autodetect_tracing_backend()` chooses a backend based on environment
variables, in the following order:

1. **Langfuse** if both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.
2. **Logfire** if `LOGFIRE_API_KEY` is set.
3. **NONE** if none of the above are present.

Examples:

```bash
# Langfuse
export LANGFUSE_PUBLIC_KEY="lf_pub_..."
export LANGFUSE_SECRET_KEY="lf_sec_..."

# or, Logfire
export LOGFIRE_API_KEY="logfire_..."
```

With these variables set, constructing `DeepAgent(..., trace=True)` is enough
to have traces sent to your chosen backend.

---

## Advanced: Manual Backend Selection

If you prefer to select a backend manually (instead of relying on
environment-based auto-detection), you can call `init_tracing_backend(...)`
yourself and leave `trace=False` (the default) when constructing `DeepAgent`.

```python 

from pydantask.agents import DeepAgent
from pydantask.observe.tracing import init_tracing_backend
from pydantask.models import TracingBackend

# Explicitly choose Langfuse or Logfire
init_tracing_backend(TracingBackend.LANGFUSE)
# or: init_tracing_backend(TracingBackend.LOGFIRE)

agent = DeepAgent(
    objective="...",
    model="gpt-5.2",
    trace=False,  # keep False so your manual choice is not overridden
)

# result = await agent.run()
```

In this pattern:

- Your call to `init_tracing_backend(...)` sets the active backend and performs
  any necessary SDK initialization for that backend.
- Because `trace=False`, `DeepAgent` does not change the active backend.

Use this if you want full control over when and how tracing is configured.

---

## Backend-specific Notes

### Langfuse

- Requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.
- Pydantask uses `langfuse.get_client()` and verifies credentials with
  `auth_check()`.
- When enabled, `pydantic_ai.Agent.instrument_all()` is called so that
  Pydantic AI internals (LLM calls, tool calls) emit spans that Langfuse can
  capture.

### Logfire

- Requires `LOGFIRE_API_KEY` (and any other configuration expected by
  `logfire.configure()`).
- Pydantask calls `logfire.configure()`, `logfire.instrument_pydantic_ai()`,
  and `logfire.instrument_httpx()` so that:
  - Pydantic AI agents
  - HTTPX calls used by model providers
  are traced automatically.

---

For an end-to-end example of using tracing in a deep research workflow, see
[Use Cases](use_cases.md), which shows a `DeepAgent` constructed with
`trace=True` and tracing-related environment variables set.
