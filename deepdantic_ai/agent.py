from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from json import load
from typing import Any, Dict, List, Optional, Callable

from annotated_types import T
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# ============================================================
# Core Agent State
# ============================================================

# Initialize Pydantic AI instrumentation
Agent.instrument_all()


class AgentState(BaseModel):
    goal: str
    knowledge: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary artifacts collected during the agent's execution such as Todo Lists, reasoning steps etc.",
    )
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
            instrument=True,
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
            system_prompt="You are a tool-using step. Decide if a tool call is useful for solving.",
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

    @tool
    async def reflect(self, ctx: RunContext[AgentState], reflection: str) -> str:
        ctx.state.artifacts.setdefault("reflections", []).append(reflection)
        return "Reflection recorded successfully."


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


class ToDo(BaseModel()):
    task: str
    status: str = Field(description="e.g., 'not_done', 'in_progress', 'completed")


class ToDoList(BaseModel):
    reasoning: str
    tasks_list: list[ToDo] = Field(
        description="A list of todo tasks with their status."
    )


class ToDoStep(Step):
    name = "todo"

    def __init__(self):
        self.agent = Agent(
            model="gpt-4.1-mini",
            output_type=ToDoList,
            system_prompt="You are an expert todo list creator for a deep agent system. Create a concise todo list of steps to solve a goal or task. Use the current knowledge and reasoning to inform your todo list.",
        )

        # regsiter tools this agent can use
        # self.agent.tool(self.store_todos)
        # self.agent.tool(self.update_todo)

    async def run(self, state: AgentState) -> StepResult:
        prompt = f"""
                GOAL:
                {state.goal}

                KNOWN FACTS:
                {state.knowledge}
                
                TODO LIST:
                {state.artifacts['todos'] if 'todos' in state.artifacts else '{}'}

                Create a concise todo list of tasks to achieve the goal.
                """
        run_result = await self.agent.run(prompt)
        output: ToDoOutput = run_result.output

        return StepResult(
            reasoning_step=output.reasoning,
            artifacts={"todo": output.tasks_list},
        )

    async def update_todo(
        self, ctx: RunContext[AgentState], todo: str, status: str
    ) -> str:
        todos = ctx.state.artifacts.setdefault("todos", [])
        for i, t in enumerate(todos):
            if t.startswith(todo):
                todos[i] = f"{todo} - {status}"
                return "Todo updated successfully."
        return "Todo not found."

    async def store_todos(self, ctx: RunContext[AgentState], todos: List[str]) -> str:
        ctx.state.artifacts.setdefault("todos", []).extend(todos)
        return "Todos stored successfully."


class PlannerOutput(BaseModel):
    reasoning: str = Field(description="The reasoning behind the chosen next action.")
    next_action: str = Field(
        description="The next action to take, must match a step name or 'stop'."
    )
    todo: str = Field(description="TODO list to accomplish the goal.", default="")


class PlannerStep(Step):
    name = "planner"

    def __init__(self, available_steps: List[str]):
        self.available_steps = available_steps
        self.agent = Agent(
            model="gpt-4.1-mini",
            output_type=PlannerOutput,
            system_prompt="""
            You are an expert task planner for a deep agent system.
            You are not to provide an answer yet, only plan the next step given the current state of things needing to be done.

            Your goal is to decide the next best action to take to progress towards the overall goal.
        

            Create a concise reasoning trace explaining why you chose the next action.
            Additionally, create or update a TODO list to help keep track of tasks needed to accomplish the goal.
            Have the TODO list reflect the current state of knowledge and reasoning.
            Have the TODO list be actionable and specific.
            Have the TODO list help guide future steps towards completing the goal.
            Do not use vague or generic tasks in the TODO list.
            
            The todo list should be in simple text format, with one task per line.
            Each line should start with a dash (-) followed by the task description and if it is done or not.
            ex. 
                - Research the history of the topic - done
                - Find relevant examples - not done
            
            You must choose one of the available steps to take next from the list provided under 'Available steps'.
            
            You have the following default step options:
                
                - If you think you have enough information to complete the task, choose 'synthesize'. 
                - If you have already synthesized or there are no new facts, choose 'stop'. 
                - If you see repeated steps without progress, choose 'stop'.
                - If you do not have a way to solve the given task, choose 'stop'.
            """,
        )

    async def run(self, state: AgentState) -> StepResult:
        prompt = f"""
            AGENT GOAL:
            {state.goal}
            
            KNOWN FACTS:
            {state.knowledge}

            REASONING TRACE:
            {state.reasoning}

            Available steps to take: {self.available_steps + ['stop']}
"""
        run_result = await self.agent.run(prompt)
        output: PlannerOutput = run_result.output

        return StepResult(
            reasoning_step=output.reasoning,
            artifacts={
                "next_action": output.next_action,
                "todo": output.todo,
            },  # might be overwriting dictionary entries we dont want that
        )

    async def write_todo(self, ctx: RunContext[AgentState], todos: ToDoList]) -> str:

        ctx.state.artifacts["todos"] = todos
        return "Todo list created."

    async def think_tool(self, ctx: RunContext[AgentState], reflection: str) -> str:
        ctx.state.artifacts.setdefault("reflections", []).append(reflection)
        return "Reflection recorded for decision-making."

    async def read_todo(self, ctx: RunContext[AgentState]) -> str:
        return ctx.state.artifacts.get("todos", "No todo list found.")


# ============================================================
# Deep Agent Executor
# ============================================================


class DeepAgent:
    def __init__(self, steps: Dict[str, Step], tools=None, max_depth: int = 15):
        self.steps = steps
        self.tools = tools  # TODO: integrate tools into steps / agents
        self.max_depth = max_depth

    async def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)

        # Create todo list at the start for the agent to work from
        # todo_step = ToDoStep()
        # todo_result = await todo_step.run(state)
        # state.reasoning.append(f"[{todo_step.name}] {todo_result.reasoning_step}")

        # todo_list = todo_result.artifacts.get("ToDo")
        # print(state)
        # if todo_list:
        #     state.artifacts["todos"] = todo_list
        #     print(f"\n--- Initial ToDo List ---\n{todo_list}")

        for _ in range(self.max_depth):
            print("\n==============================")
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
    from langfuse import get_client

    langfuse = get_client()

    # Verify connection
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Authentication failed. Please check your credentials and host.")

    asyncio.run(main())
