import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from httpx import AsyncClient

import pydantask.agents.agent as agent_mod


class TestDeepAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Ensure Tavily API key is present so DeepAgent doesn't raise on init
        self.env_patcher = patch.dict(os.environ, {"TAVILY_API_KEY": "fake-key"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_deep_agent_initialization(self):
        # Avoid creating a custom mock HTTP client; use a real AsyncClient so
        # OpenAIProvider / AsyncOpenAI accept it without TypeError.
        with patch.object(
            agent_mod.DeepAgent,
            "_create_retrying_client",
            return_value=AsyncClient(),
        ):
            deep_agent = agent_mod.DeepAgent(prompt="Test Goal")

        self.assertEqual(deep_agent.prompt, "Test Goal")
        self.assertIsInstance(deep_agent.agent_registry, dict)

        # These keys must match the implementation in _setup_default_sub_agents
        self.assertIn("research_agent", deep_agent.agent_registry)
        self.assertIn("synthesizer_agent", deep_agent.agent_registry)

        # Planner agent should be created during __init__
        self.assertIsNotNone(deep_agent._planner_agent)

    async def test_deep_agent_run_returns_runtime_state(self):
        """Verify that DeepAgent.run uses the planner and supervisor and
        returns the initialized runtime state.
        """
        with (
            patch.object(
                agent_mod.DeepAgent,
                "_create_retrying_client",
                return_value=AsyncClient(),
            ),
            patch.object(
                agent_mod.DeepAgent,
                "_initialize_runtime_state",
                return_value=MagicMock(name="runtime_state"),
            ) as mock_init_state,
        ):
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

            self.assertIs(runtime_state, mock_init_state.return_value)


if __name__ == "__main__":
    unittest.main()
