from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

import pydantask.agents.agent as agent_mod
from pydantask.models import (
    RuntimeState,
    TaskItem,
    TaskQAResult,
    TaskResult,
    TaskStatus,
    SupervisorDecision,
)


@pytest.fixture(autouse=True)
def env_vars(monkeypatch: pytest.MonkeyPatch):
    # Keep init() happy if a test *does* instantiate DeepAgent.
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


@pytest.fixture
def runtime_state() -> RuntimeState:
    return RuntimeState(objective="obj", agent_registry={}, next_task_id=1)


def make_minimal_deep_agent(prompt: str = "obj") -> agent_mod.DeepAgent:
    """Create a DeepAgent without running its heavy __init__."""
    da = agent_mod.DeepAgent.__new__(agent_mod.DeepAgent)
    da.prompt = prompt
    da.agent_registry = {}
    da._max_steps = 3
    da.checkpoint = False
    return da


def test_autodetect_tracing_backend_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.NONE

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.LANGSMITH

    monkeypatch.setenv("LOGFIRE_API_KEY", "x")
    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.LOGFIRE

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.LANGFUSE


def test_deep_agent_init_sets_registry_keys(monkeypatch: pytest.MonkeyPatch):
    # pydantic-ai inspects tools as callables and expects `__name__`.
    def _fake_tavily_tool(*args, **kwargs):
        return {"ok": True}

    with (
        patch.object(
            agent_mod.DeepAgent, "_create_retrying_client", return_value=AsyncClient()
        ),
        patch.object(agent_mod, "OpenAIProvider", autospec=True),
        patch.object(agent_mod, "OpenAIChatModel", autospec=True),
        patch.object(agent_mod, "tavily_search_tool", return_value=_fake_tavily_tool),
        # Avoid pulling in pydantic-ai's tool schema machinery for this unit test.
        patch.object(agent_mod, "Agent", autospec=True) as agent_cls,
    ):
        deep_agent = agent_mod.DeepAgent(prompt="Test Goal", trace=False)

    assert deep_agent.prompt == "Test Goal"
    assert "research_agent" in deep_agent.agent_registry
    assert "producer_agent" in deep_agent.agent_registry

    # Soft-disabled filesystem tools should not be registered by default.
    all_tools: list[object] = []
    for c in agent_cls.call_args_list:
        tools = c.kwargs.get("tools") or []
        all_tools.extend(tools)

    all_tool_names = {getattr(t, "__name__", str(t)) for t in all_tools}
    assert "read_from_file_system" not in all_tool_names
    assert "write_to_file_system" not in all_tool_names
    assert "save_task_context" not in all_tool_names
    assert "read_task_context" not in all_tool_names


@pytest.mark.asyncio
async def test_add_cancel_patch_task(runtime_state: RuntimeState):
    da = make_minimal_deep_agent()
    ctx = SimpleNamespace(deps=runtime_state)

    task_id = await da.add_task(
        ctx,
        sub_task_objective="do the thing",
        capability="worker_agent",
        dependencies=[123],
        metadata={"k": "v"},
    )

    assert task_id == 1
    assert runtime_state.next_task_id == 2

    task = runtime_state.plan[task_id]
    assert task.status == TaskStatus.READY
    assert task.sub_task_dependencies == [123]
    assert task.metadata == {"k": "v"}

    msg = await da.patch_task(
        ctx, task_id=task_id, sub_task_objective="new", dependencies=[1, 2]
    )
    assert "updated successfully" in msg
    assert runtime_state.plan[task_id].sub_task_objective == "new"
    assert runtime_state.plan[task_id].sub_task_dependencies == [1, 2]

    msg = await da.cancel_task(ctx, task_id=task_id, reason="no longer needed")
    assert "cancelled" in msg
    assert runtime_state.plan[task_id].status == TaskStatus.CANCELLED


def test_dependencies_satisfied_only_completed(runtime_state: RuntimeState):
    da = make_minimal_deep_agent()

    t1 = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="dep",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
    )
    t2 = TaskItem(
        task_id=2,
        overall_objective=runtime_state.objective,
        sub_task_objective="blocked",
        capability="worker_agent",
        status=TaskStatus.READY,
        sub_task_dependencies=[1, 999],
    )

    runtime_state.plan = {1: t1, 2: t2}

    assert da._dependencies_satisfied(t1, runtime_state) is True
    assert da._dependencies_satisfied(t2, runtime_state) is False  # dep 999 missing

    runtime_state.plan[999] = TaskItem(
        task_id=999,
        overall_objective=runtime_state.objective,
        sub_task_objective="present but not complete",
        capability="worker_agent",
        status=TaskStatus.RUNNING,
    )
    assert da._dependencies_satisfied(t2, runtime_state) is False

    runtime_state.plan[999].status = TaskStatus.COMPLETED
    assert da._dependencies_satisfied(t2, runtime_state) is True


def test_handle_critic_result_transitions():
    da = make_minimal_deep_agent()

    task = TaskItem(
        task_id=1,
        overall_objective="obj",
        sub_task_objective="do it",
        capability="worker_agent",
        status=TaskStatus.NEEDS_REVIEW,
    )

    da.handle_critic_result(task, TaskQAResult(task_id=1, passed=True, reasoning="ok"))
    assert task.status == TaskStatus.COMPLETED

    task2 = TaskItem(
        task_id=2,
        overall_objective="obj",
        sub_task_objective="do it",
        capability="worker_agent",
        status=TaskStatus.NEEDS_REVIEW,
        attempt_count=0,
        max_attempts=2,
    )
    da.handle_critic_result(
        task2, TaskQAResult(task_id=2, passed=False, reasoning="not good")
    )
    assert task2.status == TaskStatus.READY
    assert task2.attempt_count == 1
    assert "Previous attempt failed review" in task2.sub_task_objective

    task2.attempt_count = 2
    da.handle_critic_result(
        task2, TaskQAResult(task_id=2, passed=False, reasoning="still bad")
    )
    assert task2.status == TaskStatus.FAILED
    assert "Max retries reached" in (task2.error_msg or "")


@pytest.mark.asyncio
async def test_execute_sets_result_and_needs_review(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")

    sub_agent = MagicMock(name="sub_agent")
    sub_agent.run = AsyncMock(name="run")

    tr = TaskResult(task_id=1, summary="done")
    sub_agent.run.return_value = SimpleNamespace(output=tr)

    step = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="do",
        capability="worker_agent",
        status=TaskStatus.READY,
    )

    out_step = await da.execute(sub_agent, step, runtime_state)

    assert out_step.status == TaskStatus.NEEDS_REVIEW
    assert out_step.result == tr
    sub_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_ready_tasks_filters_deps_and_injects_feedback(
    runtime_state: RuntimeState,
):
    da = make_minimal_deep_agent(prompt="overall")

    # plan: task 1 completed, task 2 ready (depends on 1), task 3 blocked (depends on missing)
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="dep",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
    )
    runtime_state.plan[2] = TaskItem(
        task_id=2,
        overall_objective=runtime_state.objective,
        sub_task_objective="ready",
        capability="worker_agent",
        status=TaskStatus.READY,
        sub_task_dependencies=[1],
    )
    runtime_state.plan[3] = TaskItem(
        task_id=3,
        overall_objective=runtime_state.objective,
        sub_task_objective="blocked",
        capability="worker_agent",
        status=TaskStatus.READY,
        sub_task_dependencies=[999],
    )

    worker_impl = MagicMock(name="worker_impl")
    da.agent_registry = {
        "worker_agent": agent_mod.CapabilityDescription(
            name="worker_agent",
            description="",
            tool_func=worker_impl,
        )
    }

    async def _execute_side_effect(sub_agent, step: TaskItem, ctx: RuntimeState):
        # mimic DeepAgent.execute returning the (mutated) step
        step.result = TaskResult(task_id=step.task_id, summary=f"ran {step.task_id}")
        step.status = TaskStatus.NEEDS_REVIEW
        return step

    da.execute = AsyncMock(side_effect=_execute_side_effect)

    decision = SupervisorDecision(
        reasoning="",
        tasks_to_execute=[2, 3],
        feedback_to_subagents={2: "focus on X"},
        all_tasks_completed=False,
    )

    results = await da._execute_ready_tasks(decision, runtime_state)

    assert [t.task_id for t in results] == [2]
    assert runtime_state.plan[2].parameters["supervisor_feedback"] == "focus on X"
    da.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_task_status_and_view_qa_report(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")
    ctx = SimpleNamespace(deps=runtime_state)

    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="x",
        capability="worker_agent",
        status=TaskStatus.READY,
    )

    msg = await da.update_task_status(ctx, task_id=1, status=TaskStatus.COMPLETED)
    assert "now" in msg
    assert runtime_state.plan[1].status == TaskStatus.COMPLETED

    runtime_state.plan[1].task_feedback = TaskQAResult(
        task_id=1, passed=True, reasoning="ok"
    )
    report = await da.view_qa_report(ctx, task_id=1)
    assert '"passed": true' in report


@pytest.mark.asyncio
async def test_run_stops_when_supervisor_says_done(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")
    da._initialize_runtime_state = MagicMock(return_value=runtime_state)
    da._format_supervisor_input_prompt = MagicMock(return_value="prompt")

    supervisor = MagicMock(name="supervisor")
    supervisor.run = AsyncMock(
        return_value=SimpleNamespace(
            output=SupervisorDecision(
                reasoning="done",
                tasks_to_execute=[],
                feedback_to_subagents=None,
                all_tasks_completed=True,
            )
        )
    )
    da._supervisor_agent = supervisor

    result = await da.run()

    assert result.objective == "overall"
    assert result.final_result is None
    assert result.plan == runtime_state.plan
    supervisor.run.assert_awaited_once()
