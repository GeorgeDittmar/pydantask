"""Structured skill definitions for the dynamic model routing system.

Skills are the authoritative source for how to invoke a specific execution
engine (e.g. ``llama-server``). Each skill has:

- A **schema** (JSON file with typed flags, defaults, constraints)
- A **reference** (markdown file with human-readable guidance)

The supervisor validates every ``SpawnArgs`` object against the skill schema
before queue insertion. This prevents malformed CLI invocations from reaching
the lifecycle manager.

Usage::

    from pydantask.skills import SkillRegistry

    registry = SkillRegistry()
    llama_skill = registry.get("llama_cpp")
    llama_skill.validate_spawn_args(spawn_args)
    llama_skill.assemble_command(spawn_args, port=60001)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

from pydantask.execution.schema import SpawnArgs


# ─────────────────────────────────────────────────────────────────────────────
# Skill schema models
# ─────────────────────────────────────────────────────────────────────────────


class FlagDefinition(BaseModel):
    """Definition of a single CLI flag from the skill schema.

    Attributes:
        type: Python type name (``"integer"``, ``"boolean"``, ``"number"``).
        default: Default value for the flag.
        min: Minimum allowed value (integers and floats only).
        max: Maximum allowed value (integers and floats only).
        description: Human-readable description for the supervisor's guidance.
        spawn_args_field: Name of the corresponding ``SpawnArgs`` field.
    """

    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: str
    spawn_args_field: str


class SkillConstraints(BaseModel):
    """Global constraints for a skill.

    Attributes:
        port_range: Allowed port range [min, max].
        max_concurrent_per_port: Max tasks per port (usually 1).
        max_concurrent_models: Max simultaneous model instances (None = unlimited).
        health_check_timeout_seconds: Max seconds to wait for /health readiness.
        health_check_interval_seconds: Seconds between health check polls.
    """

    port_range: list[int]
    max_concurrent_per_port: int = 1
    max_concurrent_models: Optional[int] = None
    health_check_timeout_seconds: int = 120
    health_check_interval_seconds: float = 0.5


class SkillSchema(BaseModel):
    """Complete skill definition loaded from the JSON schema file.

    Attributes:
        engine: Name of the execution engine (e.g. ``"llama-server"``).
        engine_url: URL template for the engine's API endpoint.
        health_endpoint: URL template for the /health check.
        required_flags: Flags that must be present in every invocation.
        optional_flags: Flags that can be overridden per-task.
        constraints: Global constraints for the skill.
    """

    engine: str
    engine_url: str
    health_endpoint: str
    required_flags: list[str]
    optional_flags: Dict[str, FlagDefinition]
    constraints: SkillConstraints


# ─────────────────────────────────────────────────────────────────────────────
# Skill registry — loads and validates skill schemas
# ─────────────────────────────────────────────────────────────────────────────


# Default path to the skills directory (relative to the package root).
_SKILLS_DIR: Path = Path(__file__).parent


def _load_schema(path: Path) -> SkillSchema:
    """Load a skill schema from a JSON file.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        ValidationError: If the JSON does not match the expected structure.
    """
    with open(path) as f:
        data = json.load(f)
    return SkillSchema.model_validate(data)


class SkillRegistry:
    """Registry of structured skill definitions.

    Loads skill schemas from JSON files in the skills directory. Each JSON file
    (minus the extension) becomes a skill key. For example,
    ``llama_cpp_skill.json`` → key ``"llama_cpp"``.

    Provides validation and CLI assembly methods that the supervisor calls
    before queue insertion and that the lifecycle manager calls at execution.

    Attributes:
        skills: Mapping from skill key to ``SkillSchema``.
    """

    def __init__(self, skills_dir: Optional[Path] = None) -> None:
        self._skills_dir = skills_dir or _SKILLS_DIR
        self.skills: Dict[str, SkillSchema] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Discover and load all skill schema JSON files."""
        if not self._skills_dir.exists():
            return
        for path in sorted(self._skills_dir.glob("*.json")):
            # Match files like "llama_cpp_skill.json" → key "llama_cpp"
            if path.name.endswith("_skill.json"):
                key = path.name[: -len("_skill.json")]
                self.skills[key] = _load_schema(path)

    def get(self, skill_key: str) -> SkillSchema:
        """Get a skill schema by key.

        Raises:
            KeyError: If the skill key is not found.
        """
        if skill_key not in self.skills:
            raise KeyError(
                f"Unknown skill '{skill_key}'. Available: {list(self.skills.keys())}"
            )
        return self.skills[skill_key]

    def has(self, skill_key: str) -> bool:
        """Check if a skill key is registered."""
        return skill_key in self.skills

    def list_skills(self) -> list[str]:
        """Return all registered skill keys."""
        return list(self.skills.keys())

    def validate_spawn_args(
        self, skill_key: str, spawn_args: SpawnArgs
    ) -> None:
        """Validate a SpawnArgs object against a skill's optional flags schema.

        Checks each provided field against the skill's flag definitions for
        type compatibility, min/max bounds, and port range constraints.

        Raises:
            ValueError: If validation fails.
            KeyError: If the skill key is not found.
        """
        skill = self.get(skill_key)

        # Build a mapping from spawn_args_field to its value
        args_dict = spawn_args.model_dump(exclude_none=True)

        for flag_name, flag_def in skill.optional_flags.items():
            field_value = args_dict.get(flag_def.spawn_args_field)
            if field_value is None:
                continue  # Not provided — defaults apply, that's fine

            # Check type
            expected_type = _type_from_flag_string(flag_def.type)
            if not isinstance(field_value, expected_type):
                raise ValueError(
                    f"Flag '{flag_name}': expected {flag_def.type} "
                    f"({flag_def.spawn_args_field}), got {type(field_value).__name__}"
                )

            # Check bounds
            if flag_def.min is not None and field_value < flag_def.min:
                raise ValueError(
                    f"Flag '{flag_name}' ({flag_def.spawn_args_field}): "
                    f"value {field_value} below minimum {flag_def.min}"
                )
            if flag_def.max is not None and field_value > flag_def.max:
                raise ValueError(
                    f"Flag '{flag_name}' ({flag_def.spawn_args_field}): "
                    f"value {field_value} above maximum {flag_def.max}"
                )

        # Check port range constraints
        # (Ports are not in SpawnArgs, but the lifecycle manager assigns them.
        # This method documents the constraint for the supervisor.)
        port_range = skill.constraints.port_range
        if port_range[0] > port_range[1]:
            raise ValueError(
                f"Skill '{skill_key}': invalid port_range "
                f"[{port_range[0]}, {port_range[1]}]"
            )


def _type_from_flag_string(type_str: str) -> type:
    """Convert a flag type string to a Python type."""
    mapping = {
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "string": str,
    }
    return mapping.get(type_str, object)


# ─────────────────────────────────────────────────────────────────────────────
# CLI command assembly
# ─────────────────────────────────────────────────────────────────────────────


def assemble_llama_server_command(
    spawn_args: SpawnArgs,
    port: int,
    model_path: Optional[str] = None,
) -> list[str]:
    """Assemble a ``llama-server`` CLI command from structured spawn args.

    The supervisor writes structured ``SpawnArgs``; the lifecycle manager calls
    this function to build the actual ``asyncio.create_subprocess_exec`` argument
    list.

    Args:
        spawn_args: Structured arguments from the queue.
        port: The assigned port for this instance.
        model_path: Explicit model path override. Falls back to
            ``spawn_args.model_path`` then to the registry default.

    Returns:
        A list of command tokens suitable for ``asyncio.create_subprocess_exec``.
    """
    cmd = ["llama-server"]

    # Required flags
    if model_path is None:
        model_path = spawn_args.model_path
    if model_path:
        cmd.extend(["--model", model_path])
    cmd.extend(["--port", str(port)])

    # Optional flags
    if spawn_args.ctx_size != 4096:
        cmd.extend(["--ctx-size", str(spawn_args.ctx_size)])
    if spawn_args.n_gpu_layers != 99:
        cmd.extend(["--n-gpu-layers", str(spawn_args.n_gpu_layers)])
    if not spawn_args.mmap:
        cmd.extend(["--mmap", "false"])
    if spawn_args.temp != 0.7:
        cmd.extend(["--temp", str(spawn_args.temp)])
    if spawn_args.top_p != 0.9:
        cmd.extend(["--top-p", str(spawn_args.top_p)])

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built singleton (used by the supervisor and lifecycle manager)
# ─────────────────────────────────────────────────────────────────────────────


default_registry = SkillRegistry()
