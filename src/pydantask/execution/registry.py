"""Static model registry for the dynamic model routing system.

Contains a ``MODEL_REGISTRY`` dict keyed by model key, with per-model
configuration that the supervisor consults before task insertion:

- ``estimated_vram_mb`` — Expected memory footprint when loaded.
- ``min_memory_headroom_mb`` — Additional headroom required beyond VRAM.
- ``capability_tier`` — Human-readable capability label.
- ``default_port`` — Preferred local port.
- ``default_ctx`` — Default context window.

The registry works with the structured skill schema
(``skills/llama_cpp_skill.json``): the supervisor validates a selected model's
spawn args against the skill schema before queue insertion.

Usage::

    from pydantask.execution.registry import MODEL_REGISTRY

    model = MODEL_REGISTRY["qwen2.5-coder-7b"]
    print(model.total_required_mb)  # 6200
"""

from __future__ import annotations

from pydantask.execution.schema import ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# Model registry — static dictionary of available models
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ── Small/fast models (good for quick tasks, sequential execution) ──
    "qwen2.5-coder-1.5b": ModelConfig(
        model_key="qwen2.5-coder-1.5b",
        path="/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        default_port=60000,
        default_ctx=4096,
        estimated_vram_mb=1200,
        capability_tier="fast",
        min_memory_headroom_mb=1500,
    ),
    "gemma3-1b": ModelConfig(
        model_key="gemma3-1b",
        path="/models/gemma-3-1b-it-q4_k_m.gguf",
        default_port=60001,
        default_ctx=4096,
        estimated_vram_mb=1000,
        capability_tier="fast",
        min_memory_headroom_mb=1200,
    ),
    # ── Mid-size models (good balance of speed and capability) ──
    "qwen2.5-coder-7b": ModelConfig(
        model_key="qwen2.5-coder-7b",
        path="/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        default_port=60002,
        default_ctx=8192,
        estimated_vram_mb=4200,
        capability_tier="coder",
        min_memory_headroom_mb=2000,
    ),
    "llama3.2-3b": ModelConfig(
        model_key="llama3.2-3b",
        path="/models/llama-3.2-3b-instruct-q4_k_m.gguf",
        default_port=60003,
        default_ctx=4096,
        estimated_vram_mb=2000,
        capability_tier="general",
        min_memory_headroom_mb=1500,
    ),
    "gemma3-4b": ModelConfig(
        model_key="gemma3-4b",
        path="/models/gemma-3-4b-it-q4_k_m.gguf",
        default_port=60004,
        default_ctx=8192,
        estimated_vram_mb=2500,
        capability_tier="general",
        min_memory_headroom_mb=1500,
    ),
    # ── Larger models (good for complex reasoning, parallel execution) ──
    "qwen2.5-14b": ModelConfig(
        model_key="qwen2.5-14b",
        path="/models/qwen2.5-14b-instruct-q4_k_m.gguf",
        default_port=60005,
        default_ctx=8192,
        estimated_vram_mb=8000,
        capability_tier="reasoner",
        min_memory_headroom_mb=3000,
    ),
    "llama3.1-8b": ModelConfig(
        model_key="llama3.1-8b",
        path="/models/llama-3.1-8b-instruct-q4_k_m.gguf",
        default_port=60006,
        default_ctx=8192,
        estimated_vram_mb=5000,
        capability_tier="general",
        min_memory_headroom_mb=2000,
    ),
    # ── Heavy models (complex reasoning, only when memory allows) ──
    "qwen2.5-32b": ModelConfig(
        model_key="qwen2.5-32b",
        path="/models/qwen2.5-32b-instruct-q4_k_m.gguf",
        default_port=60007,
        default_ctx=4096,
        estimated_vram_mb=18000,
        capability_tier="reasoner",
        min_memory_headroom_mb=4000,
    ),
    "llama3.1-70b": ModelConfig(
        model_key="llama3.1-70b",
        path="/models/llama-3.1-70b-instruct-q4_k_m.gguf",
        default_port=60008,
        default_ctx=2048,
        estimated_vram_mb=38000,
        capability_tier="reasoner",
        min_memory_headroom_mb=5000,
    ),
}

# Convenience lookups
_MODEL_KEYS = list(MODEL_REGISTRY.keys())
_FAST_MODELS = [k for k, v in MODEL_REGISTRY.items() if v.capability_tier == "fast"]
_CODE_MODELS = [k for k, v in MODEL_REGISTRY.items() if v.capability_tier == "coder"]
_REASONER_MODELS = [k for k, v in MODEL_REGISTRY.items() if v.capability_tier == "reasoner"]


def get_model(model_key: str) -> ModelConfig:
    """Get a model config by key.

    Raises:
        KeyError: If the model key is not in the registry.
    """
    return MODEL_REGISTRY[model_key]


def find_models_by_tier(tier: str) -> list[str]:
    """Find model keys matching a capability tier.

    Args:
        tier: Capability tier string (``"fast"``, ``"coder"``, ``"general"``,
            ``"reasoner"``).

    Returns:
        List of model keys sorted by estimated memory usage (smallest first).
    """
    return sorted(
        [k for k, v in MODEL_REGISTRY.items() if v.capability_tier == tier],
        key=lambda k: MODEL_REGISTRY[k].total_required_mb,
    )


def find_fit_models(available_mb: int) -> list[str]:
    """Find models that fit within available memory.

    A model fits if its ``total_required_mb`` (estimated_vram + headroom) is
    less than or equal to the available memory.

    Args:
        available_mb: Available unified memory in megabytes.

    Returns:
        List of fitting model keys sorted by memory efficiency
        (smallest required first).
    """
    return sorted(
        [k for k, v in MODEL_REGISTRY.items() if v.total_required_mb <= available_mb],
        key=lambda k: MODEL_REGISTRY[k].total_required_mb,
    )


def get_smallest_model() -> ModelConfig | None:
    """Return the smallest model in the registry (by total_required_mb).

    Used by the warm standby pool to keep the most memory-efficient model
    hot between task executions.
    """
    if not MODEL_REGISTRY:
        return None
    return min(MODEL_REGISTRY.values(), key=lambda m: m.total_required_mb)
