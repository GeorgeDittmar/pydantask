import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from httpx import AsyncClient

import pydantask.agents.agent as agent_mod


@pytest.fixture(autouse=True)
def env_vars(monkeypatch):
    """Ensure required API keys are present so DeepAgent doesn't raise on init."""
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


@pytest.fixture
def patched_client():
    """Patch DeepAgent._create_retrying_client to avoid real HTTP wiring."""
    with patch.object(
        agent_mod.DeepAgent,
        "_create_retrying_client",
        return_value=AsyncClient(),
    ):
        yield


def test_deep_agent_initialization(patched_client):
    # Instantiate DeepAgent with patched HTTP client
    deep_agent = agent_mod.DeepAgent(prompt="Test Goal")

    assert deep_agent.prompt == "Test Goal"
    assert isinstance(deep_agent.agent_registry, dict)

    # These keys must match the implementation in _setup_default_sub_agents
    assert "research_agent" in deep_agent.agent_registry
    assert "producer_agent" in deep_agent.agent_registry

    # Planner agent should be created during __init__
    assert deep_agent._planner_agent is not None


@pytest.mark.asyncio
async def test_deep_agent_run_returns_runtime_state(patched_client):
    """Verify that DeepAgent.run uses the planner and supervisor and
    returns the initialized runtime state.
    """
    with patch.object(
        agent_mod.DeepAgent,
        "_initialize_runtime_state",
        return_value=MagicMock(name="runtime_state"),
    ) as mock_init_state:
        # Configure the mocked runtime_state to look like a minimal RuntimeState
        mock_state = mock_init_state.return_value
        mock_state.plan = {}
        mock_state.agent_registry = {}
        mock_state.document_store = {}
        mock_state.runtime_steps = 0

        # Instantiate DeepAgent with patched dependencies
        deep_agent = agent_mod.DeepAgent(prompt="Test Goal")

        # Create async planner and supervisor mocks
        planner_mock = MagicMock(name="planner_agent")
        planner_mock.run = AsyncMock(name="planner_run")

        supervisor_mock = MagicMock(name="supervisor_agent")
        supervisor_mock.run = AsyncMock(name="supervisor_run")

        # Planner should return an object with .output.tasks
        task_item_mock = MagicMock(name="task_item")
        task_item_mock.task_id = 1

        plan_output_mock = MagicMock(name="plan_output")
        plan_output_mock.tasks = [task_item_mock]

        planner_result_mock = MagicMock(name="planner_result")
        planner_result_mock.output = plan_output_mock
        planner_mock.run.return_value = planner_result_mock

        # Supervisor should immediately signal all_tasks_completed=True
        supervisor_output = MagicMock(name="supervisor_output")
        supervisor_output.all_tasks_completed = True

        supervisor_result_mock = MagicMock(name="supervisor_result")
        supervisor_result_mock.output = supervisor_output
        supervisor_mock.run.return_value = supervisor_result_mock

        # Override the internally created agents with our mocks
        deep_agent._planner_agent = planner_mock
        deep_agent._supervisor_agent = supervisor_mock

        # Call the async run method
        runtime_state = await deep_agent.run()

        # Assertions: run was called, runtime_state came from _initialize_runtime_state
        planner_mock.run.assert_awaited_once()
        supervisor_mock.run.assert_awaited_once()
        mock_init_state.assert_called_once()

        assert runtime_state is mock_init_state.return_value
