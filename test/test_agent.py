# test/test_agent.py

import os
import unittest
from unittest.mock import patch, MagicMock

import pydantask.agent as agent_mod


class TestDeepAgent(unittest.TestCase):
    def setUp(self):
        # Patch environment variable for API key
        self.env_patcher = patch.dict(os.environ, {"TAVILY_API_KEY": "fake-key"})
        self.env_patcher.start()

        # Patch Agent to prevent actual network/model calls
        self.agent_patch = patch("pydantask.agent.Agent", autospec=True)
        self.mock_agent_cls = self.agent_patch.start()
        # Patch Planner agent so .run_sync returns a "plan" object as expected
        mock_plan = MagicMock()
        task_item = MagicMock()
        task_item.id = "1"
        task_item.description = "Test task"
        task_item.status = "pending"
        mock_plan.tasks = [task_item]
        self.mock_agent_cls.return_value.run_sync.return_value.output = mock_plan

        # Patch RuntimeState and TaskItem for initializing the plan
        self.runtimestate_patch = patch("pydantask.agent.RuntimeState", autospec=True)
        self.mock_runtimestate = self.runtimestate_patch.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.agent_patch.stop()
        self.runtimestate_patch.stop()

    def test_deep_agent_initialization(self):
        deep_agent = agent_mod.DeepAgent(prompt="Test Goal")
        self.assertEqual(deep_agent.prompt, "Test Goal")
        self.assertIsInstance(deep_agent.agent_registry, dict)
        self.assertIn("researcher", deep_agent.agent_registry)
        self.assertIn("synthesizer", deep_agent.agent_registry)
        # Check the planner agent is set up
        self.assertIsNotNone(deep_agent._planner_agent)

    def test_deep_agent_run(self):
        # Patch _initialize_runtime_state to use a MagicMock as well
        with patch.object(agent_mod.DeepAgent, "_initialize_runtime_state", return_value=MagicMock()) as mock_init_state, \
             patch.object(agent_mod.DeepAgent, "_create_supervisor") as mock_create_supervisor:
            # Return a supervisor agent mock with run_sync returning a MagicMock for each iteration
            supervisor_mock = MagicMock()
            supervisor_mock.run_sync.return_value = "supervisor-response"
            mock_create_supervisor.return_value = supervisor_mock

            deep_agent = agent_mod.DeepAgent(prompt="Test Goal")
            # Should run its loop and eventually return the supervisor_response
            result = deep_agent.run()
            self.assertEqual(result, "supervisor-response")


if __name__ == "__main__":
    unittest.main()