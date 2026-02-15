from abc import ABC, abstractmethod
from pydantic_ai import RunContext
from pydantask.models import RuntimeState
from pydantask.prompts import RESEARCH_AGENT_SYS_PROMPT, SUPERVISOR_SYS_PROMPT


class BaseSpec(ABC):
    @abstractmethod
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        raise NotImplementedError("Must implement system_prompt method in subclass.")


class SupervisorSpec(BaseSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        plan_display = "\n".join(
            [
                f"- Task ID: {t.task_id} | Status: [{t.status}] |Task Objective: {t.task_objective} | Task Dependencies: {t.task_dependencies}"
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


class ResearcherSpec(BaseSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return RESEARCH_AGENT_SYS_PROMPT


class CoderSpec(BaseSpec):
    pass


class SynthesizerSpec(BaseSpec):
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return super().system_prompt(ctx)
