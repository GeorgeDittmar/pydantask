"""Tests for system resource introspection."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from pydantask.execution.schema import QueueStore
from pydantask.execution.resources import (
    HAS_PSUTIL,
    SystemResources,
    get_system_resources,
    get_memory_headroom,
    should_throttle_pressure,
)
from pydantask.execution.registry import get_model, MODEL_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> QueueStore:
    s = QueueStore(tempfile.mktemp(suffix=".db"))
    s.init()
    yield s
    s.close()


# ─────────────────────────────────────────────────────────────────────────────
# SystemResources dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemResources:
    def test_basic_snapshot(self) -> None:
        resources = SystemResources(
            available_memory_mb=32000,
            total_memory_mb=64000,
            used_memory_mb=32000,
            memory_pressure_percent=50,
        )
        assert resources.available_memory_mb == 32000
        assert resources.parallel_capacity == 0  # No methods attached

    def test_zero_memory(self) -> None:
        resources = SystemResources(
            available_memory_mb=0,
            total_memory_mb=0,
            used_memory_mb=0,
            memory_pressure_percent=0,
        )
        assert resources.parallel_capacity == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_system_resources
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSystemResources:
    def test_returns_resources(self) -> None:
        resources = get_system_resources()
        assert isinstance(resources, SystemResources)
        if HAS_PSUTIL:
            assert resources.total_memory_mb > 0
        assert resources.available_memory_mb >= 0
        assert 0 <= resources.memory_pressure_percent <= 100

    def test_available_plus_used_equals_total(self) -> None:
        resources = get_system_resources()
        if HAS_PSUTIL:
            # psutil available + used ~= total (some reserved for kernel)
            assert resources.available_memory_mb + resources.used_memory_mb >= resources.total_memory_mb
        else:
            assert resources.available_memory_mb == 0
            assert resources.total_memory_mb == 0

    def test_can_fit_model_method(self) -> None:
        resources = get_system_resources()
        smallest = get_model("gemma3-1b")
        if resources.available_memory_mb > 0:
            assert resources.can_fit_model(smallest) is True
        else:
            assert resources.can_fit_model(smallest) is False

    def test_best_fitting_model(self) -> None:
        resources = get_system_resources()
        if resources.available_memory_mb > 0:
            model_key = resources.best_fitting_model()
            assert model_key is not None
            assert model_key in MODEL_REGISTRY
            model = MODEL_REGISTRY[model_key]
            assert model.total_required_mb <= resources.available_memory_mb

    def test_best_fitting_model_max_parallel(self) -> None:
        resources = get_system_resources()
        if resources.available_memory_mb > 0:
            model_1 = resources.best_fitting_model(max_parallel=1)
            model_8 = resources.best_fitting_model(max_parallel=8)
            if model_1 and model_8:
                assert MODEL_REGISTRY[model_1].total_required_mb >= MODEL_REGISTRY[model_8].total_required_mb
            elif model_1 and not model_8:
                smallest = min(MODEL_REGISTRY.values(), key=lambda m: m.total_required_mb)
                assert smallest.total_required_mb * 8 > resources.available_memory_mb

    def test_with_store_tracked_ports(self, store: QueueStore) -> None:
        """Active ports from store are included."""
        task_id = store.insert_task(dag_id="dag")
        store.start_task(task_id, 60001)

        resources = get_system_resources(store)
        assert 60001 in resources.active_ports

    def test_without_store(self) -> None:
        """Without store, active_ports may be empty or system-wide."""
        resources = get_system_resources()
        assert isinstance(resources.active_ports, set)


class TestGetMemoryHeadroom:
    def test_available_memory(self) -> None:
        """When psutil is available, returns True/False."""
        model = get_model("qwen2.5-coder-7b")
        result = get_memory_headroom(model)
        if HAS_PSUTIL:
            assert isinstance(result, bool)
        else:
            assert result is None

    def test_psutil_unavailable(self) -> None:
        """When psutil is unavailable, returns None."""
        with patch("pydantask.execution.resources.HAS_PSUTIL", False):
            model = get_model("qwen2.5-coder-7b")
            result = get_memory_headroom(model)
            assert result is None


class TestShouldThrottlePressure:
    def test_low_pressure(self) -> None:
        """At normal memory usage, should not throttle."""
        with patch("pydantask.execution.resources.HAS_PSUTIL", True):
            with patch("pydantask.execution.resources._get_memory_metrics") as mock_metrics:
                mock_metrics.return_value = {"percent": 30}
                assert should_throttle_pressure() is False

    def test_high_pressure(self) -> None:
        """At high memory usage, should throttle."""
        # Need to patch HAS_PSUTIL to True so _get_memory_metrics actually
        # calls our mock instead of returning the psutil-unavailable fallback
        with patch("pydantask.execution.resources.HAS_PSUTIL", True):
            with patch("pydantask.execution.resources._get_memory_metrics") as mock_metrics:
                mock_metrics.return_value = {"percent": 90}
                assert should_throttle_pressure() is True

    def test_at_threshold(self) -> None:
        """Just below threshold — no throttle."""
        with patch("pydantask.execution.resources.HAS_PSUTIL", True):
            with patch("pydantask.execution.resources._get_memory_metrics") as mock_metrics:
                mock_metrics.return_value = {"percent": 84.9}
                assert should_throttle_pressure() is False

    def test_psutil_unavailable(self) -> None:
        """When psutil is not available at all, should not throttle."""
        with patch("pydantask.execution.resources.HAS_PSUTIL", False):
            assert should_throttle_pressure() is False
