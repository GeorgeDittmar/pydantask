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
            system_prompt="You are a synthesis step in a deep agent. Produce the final answer based on knowledge and reasoning collected.",
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
            system_prompt="""
            You are an expert task planning agent. 
            You must plan and keep track of TODOs you come up with to perform a task based on state of work and available tools. 
            
            If you think you have enough information to answer the given task, choose 'synthesize'. 
            If you have already synthesized or there are no new facts, choose 'stop'. 
            If you see repeated steps without progress, choose 'stop'.
            If you do not have a way to answer the question, choose 'stop'.
            Produce structured output only.
            """,
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
    def __init__(self, steps: Dict[str, Step], tools, max_depth: int = 15):
        self.steps = steps
        self.tools = tools  # TODO: integrate tools into steps / agents
        self.max_depth = max_depth

    async def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)

        print(f"\n--- Executing step: Planning ---")
        for _ in range(self.max_depth):

            # 1. Planner ALWAYS runs
            planner = self.steps["planner"]
            plan = await planner.run(state)

            if plan.reasoning_step:
                state.reasoning.append(f"[planner] {plan.reasoning_step}")

            next_action = plan.artifacts.get("next_action")
            print(f"\n--- Next Action: {next_action} ---")

            if next_action in ("stop", None):
                break

            if next_action not in self.steps:
                raise ValueError(f"Unknown step: {next_action}")

            # 2. Execute the chosen step
            step = self.steps[next_action]
            result = await step.run(state)
            print(result)

            state.knowledge.extend(result.new_knowledge)
            if result.reasoning_step:
                state.reasoning.append(f"[{step.name}] {result.reasoning_step}")

            if result.final_answer:
                state.final_answer = result.final_answer
                break

            if not next_action or next_action == "stop":
                print("STOPPING")
                break

            if next_action not in self.steps:
                print(f"Warning: unknown next_action '{next_action}', stopping.")
                break
            current_step = next_action

        return state


@tool
def todo_write(tasks: List[str]) -> str:
    """Tool to write a todo list from given tasks."""
    formatted_tasks = "\n".join([f"- {task}" for task in tasks])
    return f"Todo list created:\n{formatted_tasks}"


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


# ============================================================
# Example Run
# ============================================================


async def main():

    # TODO: find a way to allow folks to not have to worry about all these other pieces. maybe build it into the DeepAgent constructor?
    steps = {
        "planner": PlannerStep(available_steps=["research", "tool", "synthesize"]),
        "research": ResearchStep(),
        "tool": ToolStep(),
        "synthesize": SynthesisStep(),
    }

    agent = DeepAgent(steps=steps, max_depth=15)
    state = await agent.run(
        "Explain the significance of the Higgs boson in particle physics."
    )

    print("\nFINAL STATE")
    print("=" * 40)
    print(state.model_dump())


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
