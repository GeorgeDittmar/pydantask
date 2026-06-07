from __future__ import annotations

import textwrap

import pytest

from pydantask.agents.utils import import_yaml_workflow
from pydantask.models import TaskStatus


@pytest.mark.asyncio
async def test_import_yaml_workflow_valid_strict_contract(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Top level objective"
        reasoning_steps: "Because I said so"
        tasks:
          - task_id: 1
            sub_task_objective: "Research X"
            capability: "research_agent"
            sub_task_dependencies: []

          - task_id: 2
            sub_task_objective: "Synthesize"
            capability: "producer_agent"
            sub_task_dependencies: [1]
            is_final: true
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    plan = await import_yaml_workflow(path)

    assert plan.reasoning_steps == "Because I said so"
    assert len(plan.tasks) == 2

    t1, t2 = plan.tasks
    assert t1.task_id == 1
    assert t1.overall_objective == "Top level objective"
    assert t1.sub_task_objective == "Research X"
    assert t1.sub_task_dependencies == []
    assert t1.status == TaskStatus.PENDING

    assert t2.task_id == 2
    assert t2.overall_objective == "Top level objective"
    assert t2.sub_task_dependencies == [1]
    assert t2.is_final is True


@pytest.mark.asyncio
async def test_import_yaml_workflow_rejects_alias_keys(tmp_path):
    # These keys were supported in an earlier ergonomic draft, but the YAML contract
    # is intentionally strict now.
    yaml_text = textwrap.dedent(
        """
        objective: "Top level objective"
        tasks:
          - id: 1
            objective: "Research X"
            capability: "research_agent"
            dependencies: []
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        await import_yaml_workflow(path)

    msg = str(e.value)
    assert "Invalid workflow YAML" in msg
    # Either missing required keys or extra keys forbidden.
    assert (
        "Extra inputs are not permitted" in msg
        or "Field required" in msg
        or "task_id" in msg
        or "sub_task_objective" in msg
    )


@pytest.mark.asyncio
async def test_import_yaml_workflow_auto_marks_final_highest_task_id(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Obj"
        tasks:
          - task_id: 10
            sub_task_objective: "Later task"
            capability: "producer_agent"
            sub_task_dependencies: []

          - task_id: 2
            sub_task_objective: "Earlier task"
            capability: "research_agent"
            sub_task_dependencies: []
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    plan = await import_yaml_workflow(path, auto_mark_final=True)

    finals = [t.task_id for t in plan.tasks if t.is_final]
    assert finals == [10]


@pytest.mark.asyncio
async def test_import_yaml_workflow_multiple_final_tasks_raises(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Obj"
        tasks:
          - task_id: 1
            sub_task_objective: "A"
            capability: "research_agent"
            sub_task_dependencies: []
            is_final: true

          - task_id: 2
            sub_task_objective: "B"
            capability: "producer_agent"
            sub_task_dependencies: [1]
            is_final: true
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        await import_yaml_workflow(path)

    assert "multiple tasks" in str(e.value).lower()
    assert "final" in str(e.value).lower()


@pytest.mark.asyncio
async def test_import_yaml_workflow_missing_dependency_raises(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Obj"
        tasks:
          - task_id: 1
            sub_task_objective: "A"
            capability: "research_agent"
            sub_task_dependencies: [999]
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        await import_yaml_workflow(path)

    assert "depends on missing" in str(e.value).lower()


@pytest.mark.asyncio
async def test_import_yaml_workflow_cycle_dependency_raises(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Obj"
        tasks:
          - task_id: 1
            sub_task_objective: "A"
            capability: "research_agent"
            sub_task_dependencies: [2]

          - task_id: 2
            sub_task_objective: "B"
            capability: "producer_agent"
            sub_task_dependencies: [1]
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        await import_yaml_workflow(path)

    assert "cycle" in str(e.value).lower()


@pytest.mark.asyncio
async def test_import_yaml_workflow_duplicate_task_id_raises(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Obj"
        tasks:
          - task_id: 1
            sub_task_objective: "A"
            capability: "research_agent"
            sub_task_dependencies: []

          - task_id: 1
            sub_task_objective: "B"
            capability: "producer_agent"
            sub_task_dependencies: []
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        await import_yaml_workflow(path)

    assert "duplicate" in str(e.value).lower()
    assert "task_id" in str(e.value).lower()


@pytest.mark.asyncio
async def test_import_yaml_workflow_forbids_extra_task_fields(tmp_path):
    yaml_text = textwrap.dedent(
        """
        objective: "Obj"
        tasks:
          - task_id: 1
            sub_task_objective: "A"
            capability: "research_agent"
            sub_task_dependencies: []
            made_up_key: "nope"
        """
    ).lstrip()

    path = tmp_path / "workflow.yml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        await import_yaml_workflow(path)

    msg = str(e.value)
    assert "Invalid workflow YAML" in msg
    assert "Extra inputs are not permitted" in msg
