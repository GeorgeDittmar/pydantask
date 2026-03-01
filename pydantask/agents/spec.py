from abc import ABC, abstractmethod
from typing import Callable, Any, Union
from pydantic_ai.agent import Agent
from pydantic_ai import RunContext
from pydantask.models import RuntimeState
from pydantask.prompts import RESEARCH_AGENT_SYS_PROMPT, SUPERVISOR_SYS_PROMPT
from pydantask.prompts.prompts import PRODUCER_SYS_PROMPT


class BaseAgentSpec(ABC):
    @abstractmethod
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        raise NotImplementedError("Must implement system_prompt method in subclass.")


class SupervisorSpec(BaseAgentSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        plan_display = "\n".join(
            [
                f"- Task ID: {t.task_id} | Status: [{t.status}] |Task Objective: {t.sub_task_objective} | Task Dependencies: {t.sub_task_dependencies}"
                for t in ctx.deps.plan.values()
            ]
        )

        # Simplify the registry so the Supervisor sees "Tools" not "Agent Objects"
        agent_display = "\n".join(
            [
                f"- {uuid}: {info.description}"
                for uuid, info in ctx.deps.agent_registry.items()
            ]
        )

        return SUPERVISOR_SYS_PROMPT.format(
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
