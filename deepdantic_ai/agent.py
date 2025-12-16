from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from json import load
from typing import Any, Dict, List, Optional, Callable

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# ============================================================
# Core Agent State
# ============================================================


class AgentState(BaseModel):
    goal: str
    knowledge: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None


class StepResult(BaseModel):
    new_knowledge: List[str] = Field(default_factory=list)
    reasoning_step: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None


# ============================================================
# Tool Helpers
# ============================================================


def tool(fn: Callable) -> Callable:
    """Marks a function as a tool, auto-registerable by PydanticAI"""
    fn.__deepdantic_tool__ = True
    return fn


def collect_tools(obj: Any) -> Dict[str, Callable]:
    """Collect all @tool methods in an object"""
    return {
        name: fn
        for name, fn in inspect.getmembers(obj)
        if callable(fn) and getattr(fn, "__deepdantic_tool__", False)
    }


# ============================================================
# Step Abstraction
# ============================================================


class Step(ABC):
    name: str

    @abstractmethod
    async def run(self, state: AgentState) -> StepResult:
        pass


# ============================================================
# Research Step
# ============================================================


class ResearchOutput(BaseModel):
    reasoning: str
    facts: List[str]


class ResearchStep(Step):
    name = "research"

    def __init__(self):
        self.agent = Agent(
            model="gpt-4.1-mini",
            output_type=ResearchOutput,
            system_prompt="You are a research step in a deep agent. Produce structured output only.",
        )

    async def run(self, state: AgentState) -> StepResult:
        prompt = f"""
GOAL:
{state.goal}

KNOWN FACTS:
{state.knowledge}

Produce:
- One concise reasoning step
- New facts only
"""
        run_result = await self.agent.run(prompt)
        output: ResearchOutput = run_result.output

        return StepResult(
            new_knowledge=output.facts,
            reasoning_step=output.reasoning,
        )


# ============================================================
# Tool Step
# ============================================================


class ToolOutput(BaseModel):
    reasoning: str
    tool_used: Optional[str] = None
    tool_result: Optional[Any] = None


class ToolStep(Step):
    name = "tool"

    def __init__(self):
        self.agent = Agent(
            model="gpt-4.1-mini",
            output_type=ToolOutput,
            system_prompt="You are a tool-using step. Decide if a tool call is useful.",
        )
        for fn in collect_tools(self).values():
            self.agent.tool(fn)

    async def run(self, state: AgentState) -> StepResult:
        prompt = f"""
GOAL:
{state.goal}

KNOWN FACTS:
{state.knowledge}

Decide if a tool call is useful.
"""
        run_result = await self.agent.run(prompt)
        output: ToolOutput = run_result.output

        artifacts = {}
        if output.tool_used:
            artifacts[output.tool_used] = output.tool_result

        return StepResult(
            reasoning_step=output.reasoning,
            artifacts=artifacts,
        )

    @tool
    async def store_fact(self, ctx: RunContext[AgentState], fact: str) -> str:
        ctx.state.artifacts.setdefault("stored_facts", []).append(fact)
        return "Fact stored successfully."


# ============================================================
# Synthesis Step (produces final_answer)
# ============================================================


class SynthesisOutput(BaseModel):
    reasoning: str
    answer: str


class SynthesisStep(Step):
    name = "synthesize"

    def __init__(self):
        self.agent = Agent(
            model="gpt-4.1-mini",
            output_type=SynthesisOutput,
            system_prompt="You are the synthesis step. Produce the final answer based on knowledge and reasoning.",
        )

    async def run(self, state: AgentState) -> StepResult:
        prompt = f"""
GOAL:
{state.goal}

KNOWN FACTS:
{state.knowledge}

REASONING TRACE:
{state.reasoning}

Produce a final, concise answer.
"""
        run_result = await self.agent.run(prompt)
        output: SynthesisOutput = run_result.output

        return StepResult(reasoning_step=output.reasoning, final_answer=output.answer)


# ============================================================
# Planner Step (controls flow)
# ============================================================


class PlannerOutput(BaseModel):
    reasoning: str
    next_action: str  # must match a step name or "stop"


class PlannerStep(Step):
    name = "planner"

    def __init__(self, available_steps: List[str]):
        self.available_steps = available_steps
        self.agent = Agent(
            model="gpt-4.1-mini",
            output_type=PlannerOutput,
            system_prompt="You are a planner. Decide the next step for the agent based on state. If you think you have enough information to answer the goal, choose 'synthesize'. If you have already synthesized or there are no new facts, choose 'stop'.",
        )

    async def run(self, state: AgentState) -> StepResult:
        prompt = f"""
GOAL:
{state.goal}

KNOWN FACTS:
{state.knowledge}

REASONING TRACE:
{state.reasoning}

Available steps: {self.available_steps + ['stop']}

Decide what to do next.
"""
        run_result = await self.agent.run(prompt)
        output: PlannerOutput = run_result.output

        return StepResult(
            reasoning_step=output.reasoning,
            artifacts={"next_action": output.next_action},
        )


# ============================================================
# Deep Agent Executor
# ============================================================


class DeepAgent:
    def __init__(self, steps: Dict[str, Step], max_depth: int = 15):
        self.steps = steps
        self.max_depth = max_depth

    async def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        current_step = "planner"

        for _ in range(self.max_depth):
            print(f"\n--- Executing step: {current_step} ---")
            step = self.steps[current_step]
            result = await step.run(state)

            state.knowledge.extend(result.new_knowledge)
            if result.reasoning_step:
                state.reasoning.append(f"[{step.name}] {result.reasoning_step}")
            state.artifacts.update(result.artifacts)
            if result.final_answer:
                state.final_answer = result.final_answer
                break

            next_action = state.artifacts.get("next_action")
            if not next_action or next_action == "stop":
                break
            if next_action not in self.steps:
                print(f"Warning: unknown next_action '{next_action}', stopping.")
                break
            current_step = next_action

        return state


# ============================================================
# Example Run
# ============================================================


async def main():
    steps = {
        "planner": PlannerStep(available_steps=["research", "tool", "synthesize"]),
        "research": ResearchStep(),
        "tool": ToolStep(),
        "synthesize": SynthesisStep(),
    }

    agent = DeepAgent(steps=steps, max_depth=15)
    state = await agent.run("Explain transformers at a high level")

    print("\nFINAL STATE")
    print("=" * 40)
    print(state.model_dump())


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
