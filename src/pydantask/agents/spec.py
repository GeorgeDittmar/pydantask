from abc import ABC, abstractmethod
from typing import Callable, Any, Union
from pydantic_ai.agent import Agent
from pydantic_ai import RunContext
from src.pydantask.models import RuntimeState
from src.pydantask.prompts import (
    RESEARCH_AGENT_SYS_PROMPT,
    SUPERVISOR_INPUT_PROMPT,
    DYNAMIC_SUPERVISOR_SYS_PROMPT,
)
from src.pydantask.prompts.prompts import PRODUCER_SYS_PROMPT


class BaseAgentSpec(ABC):
    @abstractmethod
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        raise NotImplementedError("Must implement system_prompt method in subclass.")


class SupervisorSpec(BaseAgentSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:

        return DYNAMIC_SUPERVISOR_SYS_PROMPT

    def format_input_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        plan_display_lines = []
        for t in ctx.deps.plan.values():
            line = (
                f"- Task ID: {t.task_id} | Status: [{t.status}] "
                f"| Objective: {t.sub_task_objective} "
                f"| Dependencies: {t.sub_task_dependencies}"
            )

            fb = getattr(t, "task_feedback", None)
            if fb is not None:
                # Adjust these fields to match TaskQAResult
                # verdict = getattr(fb, "passed", None)
                verdict = getattr(fb, "passed", None)
                summary = getattr(fb, "reasoning", None)

                line += "\n  QA: "
                if verdict is not None:
                    line += f"verdict={verdict} "
                if summary:
                    line += f"\n    summary: {summary}"

            plan_display_lines.append(line)

        plan_display = "\n".join(plan_display_lines)
        # Simplify the registry so the Supervisor sees "Tools" not "Agent Objects"
        agent_display = "\n".join(
            [
                f"- {uuid}: {info.description}"
                for uuid, info in ctx.deps.agent_registry.items()
            ]
        )
        return SUPERVISOR_INPUT_PROMPT.format(
            objective=ctx.deps.objective,
            plan_display=plan_display,
            agent_display=agent_display,
        )


class ResearcherSpec(BaseAgentSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return RESEARCH_AGENT_SYS_PROMPT


class CoderSpec(BaseAgentSpec):
    pass


class SynthesizerSpec(BaseAgentSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return super().system_prompt(ctx)


class ProducerSpec(BaseAgentSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return PRODUCER_SYS_PROMPT
