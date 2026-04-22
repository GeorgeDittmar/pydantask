from __future__ import annotations

import os
import functools
import atexit
from contextvars import ContextVar
from typing import Any, Callable, TypeVar, Literal, ParamSpec, Coroutine, cast

from loguru import logger
from langfuse import get_client

from pydantask.models import TracingBackend

_langfuse_instrumented = False
_logfire_instrumented = False
_langsmith_instrumented = False


def flush_tracing() -> None:
    """Best-effort flush for whichever backend is active.

    Useful for short-lived scripts (e.g. `python example.py`) where the process
    may exit before background exporters finish.
    """
    try:
        backend = get_active_tracing_backend()

        if backend == TracingBackend.LANGFUSE:
            # Langfuse SDK buffers events; flush ensures they're sent before exit.
            get_client().flush()
            return

        if backend == TracingBackend.LOGFIRE:
            import logfire

            logfire.force_flush()
            return

        # LangSmith is typically controlled by its own SDK; no universal flush.
    except Exception as e:
        logger.debug(f"Tracing flush failed (ignored): {e}")


def autodetect_tracing_backend() -> TracingBackend:
    """
    Choose a tracing backend automatically based on environment variables.

    Order of precedence:
      1) Langfuse if LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY present
      2) Logfire if LOGFIRE_API_KEY present (example name)
      3) LangSmith if LANGCHAIN_API_KEY or LANGSMITH_API_KEY present
      4) Otherwise, NONE
    """
    # Langfuse
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return TracingBackend.LANGFUSE

    # Logfire (replace with whatever their SDK expects)
    if os.getenv("LOGFIRE_API_KEY"):
        return TracingBackend.LOGFIRE

    # LangSmith (example; adjust to your actual config)
    if (
        os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or os.getenv("LANGCHAIN_TRACING_V2") in {"1", "true", "True"}
    ):
        return TracingBackend.LANGSMITH

    return TracingBackend.NONE


def init_langfuse_tracing() -> None:
    """Initialize Langfuse tracing once, if credentials are valid.

    Note: If you want *deep* nested traces for PydanticAI internals (LLM calls,
    tool calls), enable PydanticAI's OpenTelemetry instrumentation.
    """
    global _langfuse_instrumented
    if _langfuse_instrumented:
        return

    try:
        lf = get_client()
        logger.info("Attempting to enable Langfuse tracing...")
        if not lf.auth_check():
            logger.warning(
                "Langfuse auth_check failed. LANGFUSE_* env vars missing/invalid; "
                "Langfuse tracing will remain disabled."
            )
            return

        # Enable PydanticAI OpenTelemetry instrumentation so Langfuse can show
        # nested agent/model/tool spans (requires Langfuse OTEL support).
        from pydantic_ai.agent import Agent

        Agent.instrument_all()

        _langfuse_instrumented = True
        # Ensure traces are flushed at interpreter shutdown for short-lived scripts.
        atexit.register(flush_tracing)
        logger.info("Langfuse tracing enabled.")

    except Exception as e:
        logger.exception(f"Failed to initialize Langfuse tracing: {e}")


def init_logfire_tracing() -> None:
    """Initialize Logfire tracing once."""
    global _logfire_instrumented 
    if _logfire_instrumented:
        return

    try:
        logger.info("Attempting to enable Logfire tracing...")
        import logfire
        logfire.configure()
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx()
        _logfire_instrumented = True
        # Ensure spans are flushed at interpreter shutdown for short-lived scripts.
        atexit.register(flush_tracing)
        logger.info("Logfire tracing enabled.")
    except Exception as e:
        logger.exception(f"Failed to initialize Logfire tracing: {e}")


def init_langsmith_tracing() -> None:
    """Initialize LangSmith tracing once."""
    global _langsmith_instrumented
    if _langsmith_instrumented:
        return

    try:
        logger.info("Attempting to enable LangSmith tracing...")
        # TODO: import and configure LangSmith client here.
        # e.g. from langsmith import Client; client = Client(api_key=..., ...).
        # Then register it with your LLM / tool stack as needed.
        _langsmith_instrumented = True
        logger.info("LangSmith tracing enabled.")
    except Exception as e:
        logger.exception(f"Failed to initialize LangSmith tracing: {e}")


def init_tracing_backend(backend: TracingBackend) -> None:
    set_active_tracing_backend(backend)

    if backend == TracingBackend.NONE:
        logger.info("No tracing configured.")
        return

    if backend == TracingBackend.LANGFUSE:
        init_langfuse_tracing()
        return

    if backend == TracingBackend.LOGFIRE:
        init_logfire_tracing()
        return

    if backend == TracingBackend.LANGSMITH:
        # os.environ.setdefault("LANGSMITH_TRACING", "true")
        # os.environ.setdefault("LANGSMITH_PROJECT", "pydantask")
        init_langsmith_tracing()
        # LangSmith runs are generally sent synchronously, but register anyway for symmetry.
        atexit.register(flush_tracing)
        return
    


P = ParamSpec("P")
R = TypeVar("R")
AsyncFn = Callable[P, Coroutine[Any, Any, R]]

# LangSmith's supported run_type values are Literals; keeping this typed avoids
# pyright/mypy complaints when passing run_type into `traceable`.
LangSmithRunType = Literal[
    "tool",
    "chain",
    "llm",
    "retriever",
    "embedding",
    "prompt",
]

_ACTIVE_BACKEND: ContextVar[TracingBackend] = ContextVar(
    "pydantask_active_tracing_backend", default=TracingBackend.NONE
)


def set_active_tracing_backend(backend: TracingBackend) -> None:
    _ACTIVE_BACKEND.set(backend)


def get_active_tracing_backend() -> TracingBackend:
    return _ACTIVE_BACKEND.get()


def traced(
    name: str | None = None,
    *,
    run_type: LangSmithRunType = "chain",
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[AsyncFn], AsyncFn]:
    """Route tracing to exactly one backend (selected elsewhere).

    Note: this decorator is intended to be used as `@traced()` or
    `@traced(run_type="tool")`.

    Backends:
      - Langfuse: observe(name=...)
      - LangSmith: traceable(name=..., run_type=...)
      - Logfire: logfire.span(name)
      - NONE: no-op
    """

    def _decorator(fn: AsyncFn) -> AsyncFn:
        span_name = name or fn.__qualname__  # e.g. "DeepAgent.run"

        @functools.wraps(fn)
        async def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            backend = get_active_tracing_backend()

            if backend == TracingBackend.LANGFUSE:
                from langfuse import observe

                # Avoid serialization issues (e.g. passing a PydanticAI Agent object)
                # by allowing callers to disable input/output capture.
                observed_fn = observe(
                    name=span_name,
                    capture_input=capture_input,
                    capture_output=capture_output,
                )(fn)
                return await cast(AsyncFn, observed_fn)(*args, **kwargs)

            if backend == TracingBackend.LANGSMITH:
                try:
                    from langsmith import traceable
                except Exception:
                    from langsmith.run_helpers import traceable  # type: ignore

                traced_fn = traceable(name=span_name, run_type=run_type)(fn)
                return await cast(AsyncFn, traced_fn)(*args, **kwargs)

            if backend == TracingBackend.LOGFIRE:
                import logfire

                with logfire.span(span_name):
                    return await fn(*args, **kwargs)

            return await fn(*args, **kwargs)

        return cast(AsyncFn, _wrapped)

    return _decorator