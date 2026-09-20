"""Tests for the model registry and structured skill schema system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantask.execution.registry import (
    MODEL_REGISTRY,
    find_fit_models,
    find_models_by_tier,
    get_model,
    get_smallest_model,
)
from pydantask.execution.schema import SpawnArgs
from pydantask.skills import (
    FlagDefinition,
    SkillRegistry,
    assemble_llama_server_command,
    default_registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelRegistry:
    def test_registry_has_entries(self) -> None:
        assert len(MODEL_REGISTRY) > 0

    def test_key_access(self) -> None:
        model = get_model("qwen2.5-coder-7b")
        assert model.model_key == "qwen2.5-coder-7b"

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model("nonexistent-model")

    def test_total_required_mb(self) -> None:
        model = get_model("qwen2.5-coder-7b")
        assert model.total_required_mb == 4200 + 2000  # 6200

    def test_smallest_model(self) -> None:
        smallest = get_smallest_model()
        assert smallest is not None
        # The smallest should be 1B class
        assert smallest.total_required_mb <= get_model("qwen2.5-coder-7b").total_required_mb

    def test_find_models_by_tier(self) -> None:
        fast = find_models_by_tier("fast")
        assert len(fast) > 0
        # All returned models should be "fast" tier
        for key in fast:
            assert MODEL_REGISTRY[key].capability_tier == "fast"
        # Should be sorted by memory (smallest first)
        for i in range(len(fast) - 1):
            assert (
                MODEL_REGISTRY[fast[i]].total_required_mb
                <= MODEL_REGISTRY[fast[i + 1]].total_required_mb
            )

    def test_find_models_by_unknown_tier(self) -> None:
        result = find_models_by_tier("nonexistent")
        assert result == []

    def test_find_fit_models(self) -> None:
        # 64GB machine
        fits_64gb = find_fit_models(64000)
        assert len(fits_64gb) > 0
        # All should fit
        for key in fits_64gb:
            assert MODEL_REGISTRY[key].total_required_mb <= 64000

        # 2GB — only fast models should fit
        fits_2gb = find_fit_models(2000)
        for key in fits_2gb:
            assert MODEL_REGISTRY[key].total_required_mb <= 2000

    def test_find_fit_models_no_fit(self) -> None:
        result = find_fit_models(100)
        assert result == []

    def test_default_ports_are_unique(self) -> None:
        ports = [m.default_port for m in MODEL_REGISTRY.values()]
        assert len(ports) == len(set(ports)), "Default ports must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# Skill schema tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFlagDefinition:
    def test_basic(self) -> None:
        flag = FlagDefinition(
            type="integer",
            default=4096,
            min=128,
            max=131072,
            description="Context size",
            spawn_args_field="ctx_size",
        )
        assert flag.type == "integer"
        assert flag.min == 128
        assert flag.spawn_args_field == "ctx_size"

    def test_no_bounds(self) -> None:
        flag = FlagDefinition(
            type="boolean",
            default=True,
            description="Memory map",
            spawn_args_field="mmap",
        )
        assert flag.min is None
        assert flag.max is None


class TestSkillSchema:
    def test_loads_from_json_file(self) -> None:
        """The llama_cpp skill schema loads successfully from disk."""
        assert default_registry.has("llama_cpp")
        schema = default_registry.get("llama_cpp")
        assert schema.engine == "llama-server"

    def test_required_flags(self) -> None:
        schema = default_registry.get("llama_cpp")
        assert "--model" in schema.required_flags
        assert "--port" in schema.required_flags

    def test_optional_flags_have_defaults(self) -> None:
        schema = default_registry.get("llama_cpp")
        assert "--ctx-size" in schema.optional_flags
        assert schema.optional_flags["--ctx-size"].default == 4096

        assert "--mmap" in schema.optional_flags
        assert schema.optional_flags["--mmap"].default is True

        assert "--n-gpu-layers" in schema.optional_flags
        assert schema.optional_flags["--n-gpu-layers"].default == 99

    def test_constraints(self) -> None:
        schema = default_registry.get("llama_cpp")
        assert schema.constraints.port_range == [60000, 65535]
        assert schema.constraints.max_concurrent_per_port == 1
        assert schema.constraints.health_check_timeout_seconds == 120


class TestSkillRegistry:
    def test_discovery(self) -> None:
        """SkillRegistry auto-discovers *_skill.json files."""
        registry = SkillRegistry()
        assert "llama_cpp" in registry.list_skills()

    def test_missing_skill_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent_skill")

    def test_has_method(self) -> None:
        registry = SkillRegistry()
        assert registry.has("llama_cpp") is True
        assert registry.has("missing") is False

    def test_custom_skills_dir(self, tmp_path: Path) -> None:
        """Registry can load from a custom directory."""
        # Create a dummy skill schema
        schema_data = {
            "engine": "test-engine",
            "engine_url": "http://localhost:9999/test",
            "health_endpoint": "http://localhost:9999/health",
            "required_flags": ["--test"],
            "optional_flags": {},
            "constraints": {
                "port_range": [50000, 59999],
            },
        }
        (tmp_path / "test_skill.json").write_text(json.dumps(schema_data))

        registry = SkillRegistry(skills_dir=tmp_path)
        assert "test" in registry.list_skills()
        assert registry.get("test").engine == "test-engine"


# ─────────────────────────────────────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_spawn_args(self) -> None:
        """SpawnArgs within bounds pass validation."""
        args = SpawnArgs(ctx_size=8192, temp=0.5, n_gpu_layers=32)
        default_registry.validate_spawn_args("llama_cpp", args)

    def test_default_spawn_args(self) -> None:
        """Default SpawnArgs pass validation."""
        args = SpawnArgs()
        default_registry.validate_spawn_args("llama_cpp", args)

    def test_ctx_size_above_max(self) -> None:
        """ctx_size above the skill's max is rejected."""
        args = SpawnArgs(ctx_size=200000)
        with pytest.raises(ValueError, match="above maximum"):
            default_registry.validate_spawn_args("llama_cpp", args)

    def test_ctx_size_below_min(self) -> None:
        """ctx_size below the skill's min is rejected."""
        args = SpawnArgs(ctx_size=10)
        with pytest.raises(ValueError, match="below minimum"):
            default_registry.validate_spawn_args("llama_cpp", args)

    def test_temp_above_max(self) -> None:
        """temp above 2.0 is rejected by Pydantic (SpawnArgs ge/le bounds)."""
        with pytest.raises(ValueError):  # Pydantic ValidationError
            SpawnArgs(temp=3.0)

    def test_top_p_above_max(self) -> None:
        """top_p above 1.0 is rejected by Pydantic (SpawnArgs ge/le bounds)."""
        with pytest.raises(ValueError):  # Pydantic ValidationError
            SpawnArgs(top_p=1.5)

    def test_n_gpu_layers_above_max(self) -> None:
        """n_gpu_layers above 999 is rejected by Pydantic (SpawnArgs ge/le bounds)."""
        with pytest.raises(ValueError):  # Pydantic ValidationError
            SpawnArgs(n_gpu_layers=1000)

    def test_skill_schema_bounds_enforce_via_pydantic(self) -> None:
        """SpawnArgs fields have ge/le constraints that shadow the skill schema.

        The skill schema's min/max bounds are the authoritative contract
        (documented for the supervisor and lifecycle manager), but SpawnArgs
        enforces them at construction time via Pydantic. The skill validation
        layer re-checks bounds as a belt-and-suspenders defense.
        """
        # Edge values that are within both Pydantic and skill bounds
        args = SpawnArgs(ctx_size=128, n_gpu_layers=0, temp=0.0, top_p=0.0)
        default_registry.validate_spawn_args("llama_cpp", args)

        # Values at the top of Pydantic bounds
        args = SpawnArgs(ctx_size=131072, n_gpu_layers=999, temp=2.0, top_p=1.0)
        default_registry.validate_spawn_args("llama_cpp", args)

    def test_unknown_skill_key(self) -> None:
        """Validation raises for unknown skill keys."""
        args = SpawnArgs()
        with pytest.raises(KeyError):
            default_registry.validate_spawn_args("nonexistent", args)


# ─────────────────────────────────────────────────────────────────────────────
# CLI command assembly tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCommandAssembly:
    def test_minimal_command(self) -> None:
        """With only required fields, assemble minimal command."""
        args = SpawnArgs(model_path="/models/test.gguf")
        cmd = assemble_llama_server_command(args, port=60001)
        assert cmd == ["llama-server", "--model", "/models/test.gguf", "--port", "60001"]

    def test_command_with_custom_ctx(self) -> None:
        """Custom ctx_size appears in the command."""
        args = SpawnArgs(model_path="/models/test.gguf", ctx_size=16384)
        cmd = assemble_llama_server_command(args, port=60001)
        assert "--ctx-size" in cmd
        assert "16384" in cmd

    def test_command_with_custom_gpu_layers(self) -> None:
        args = SpawnArgs(model_path="/models/test.gguf", n_gpu_layers=0)
        cmd = assemble_llama_server_command(args, port=60001)
        assert "--n-gpu-layers" in cmd
        assert "0" in cmd

    def test_command_with_mmap_false(self) -> None:
        args = SpawnArgs(model_path="/models/test.gguf", mmap=False)
        cmd = assemble_llama_server_command(args, port=60001)
        assert "--mmap" in cmd
        assert "false" in cmd

    def test_defaults_not_included(self) -> None:
        """Default values are omitted from the command (keep it short)."""
        args = SpawnArgs(
            model_path="/models/test.gguf",
            ctx_size=4096,  # default — should not appear
            n_gpu_layers=99,  # default — should not appear
            mmap=True,  # default — should not appear
            temp=0.7,  # default — should not appear
            top_p=0.9,  # default — should not appear
        )
        cmd = assemble_llama_server_command(args, port=60001)
        assert "--ctx-size" not in cmd
        assert "--n-gpu-layers" not in cmd
        assert "--mmap" not in cmd
        assert "--temp" not in cmd
        assert "--top-p" not in cmd

    def test_model_path_override(self) -> None:
        """Explicit model_path overrides spawn_args.model_path."""
        args = SpawnArgs(model_path="/models/override.gguf")
        cmd = assemble_llama_server_command(args, port=60001, model_path="/models/explicit.gguf")
        assert "--model" in cmd
        assert "/models/explicit.gguf" in cmd

    def test_model_path_none_omits_flag(self) -> None:
        """No model_path means --model is omitted (registry will provide it)."""
        args = SpawnArgs()
        cmd = assemble_llama_server_command(args, port=60001, model_path=None)
        assert "--model" not in cmd
        assert cmd == ["llama-server", "--port", "60001"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration: registry + skill schema together
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistrySkillIntegration:
    def test_model_registry_entries_have_valid_ports(self) -> None:
        """All model default ports fall within the skill's constraint range."""
        schema = default_registry.get("llama_cpp")
        port_min, port_max = schema.constraints.port_range
        for model in MODEL_REGISTRY.values():
            assert port_min <= model.default_port <= port_max, (
                f"Model {model.model_key} port {model.default_port} "
                f"outside range [{port_min}, {port_max}]"
            )

    def test_can_validate_spawn_args_for_a_registered_model(self) -> None:
        """End-to-end: pick a model, build spawn args, validate against skill."""
        model = get_model("qwen2.5-coder-7b")
        args = SpawnArgs(
            model_path=model.path,
            ctx_size=model.default_ctx,
            n_gpu_layers=99,
            mmap=True,
        )
        # Should not raise
        default_registry.validate_spawn_args("llama_cpp", args)

        # Should assemble a valid command
        cmd = assemble_llama_server_command(args, port=model.default_port)
        assert cmd[0] == "llama-server"
        assert "--port" in cmd
        assert str(model.default_port) in cmd
