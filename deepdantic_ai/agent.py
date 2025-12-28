from asyncio import tasks
import json
import uuid
from os import system

from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


SUPERVISOR_SYSTEM_PROMPT = """
You are the supervisor agent in a deep agent system.
You own todos, delegate tasks to sub-agents, and ensure progress towards the overall goal.

Rules:
- Only you may create or update todos.
- You must reason over provided goal, todo_state, and last_event.
- Decide on next action: create_todo, update_todo, delegate, or complete.
- When creating a todo, provide a description.
- When updating a todo, provide the todo_id and new status.
- When delegating, specify target_agent and payload for said agent.
- When completing, ensure all todos are done.

Rules for Todo Management:
- A todo has id, description, status (pending, in_progress, done), and optional agent owner.
- You may create multiple todos at once if you see fit.
- You may also add new todos based on progress, but no more than one at a time.
- You may only update existing todos by id.
- You must ensure todos are progressing towards the overall goal.
- You must keep the todo list organized and relevant to the goal.
- You must make todos actionable and clear.
- You must prioritize todos that unblock progress towards the goal.
- You must not create duplicate todos.

Output a valid NextAction object only.
"""


# =========================
# Runtime state models
# =========================


class TaskItem(BaseModel):
    id: str
    description: str
    status: Literal["pending", "in_progress", "done"]
    owner: Optional[str] = None
    capability: Literal[
        "research",
        "analysis",
        "synthesis",
        "external_interaction",
        "creative_problem_solving",
        "collaboration",
    ] = "research"


# =========================
# Planner state models
# =========================
class Plan(BaseModel):
    tasks: List[TaskItem] = Field(default_factory=list)


# class RuntimeState(BaseModel):
#     goal: str
#     plan: PlanState
#     last_event: Optional[str] = None


class ResearchResult(BaseModel):
    findings: list[str]
    sources: list[str]
    confidence: float


class RuntimeState(BaseModel):
    plan: Plan
    completed_steps: set[int]
    research_results: dict[int, ResearchResult]
    iteration: int = 0
    max_iterations: int = 20


class NextAction(BaseModel):
    """Next action to be taken by the supervisor agent."""

    reasoning_summary: str
    action_type: Literal["", "delegate", "complete"]
    # create: Optional[list[CreateTodo]] = None
    # update: Optional[UpdateTodo] = None
    target_agent: Optional[str] = None
    payload: Optional[dict] = None


# =========================
# Supervisor input/output
# =========================
class SupervisorDeps(BaseModel):
    goal: str
    plan: Plan
    last_event: Optional[str] = None


# =========================
# planner agent to determine tasks to perform to achieve the goal
# =========================

PLANNER_SYSTEM_PROMPT = """
Your job is to break down the goal into manageable discrete tasks that can be delegated to sub agents.

Rules:
- You must prioritize tasks that unblock progress towards the goal.
- You must not create duplicate tasks.  
- Each task should be clear and specific.
- You must make tasks actionable and clear.
- Tasks should be concise, ideally under 10 words.
- When creating multiple tasks, ensure they are distinct and cover different aspects of the goal.

Capabilities:
- You can create tasks that require 'research', 'analysis', or 'synthesis' of information.
- You can create tasks that require interaction with external systems or APIs.
- You can create tasks that require creative problem solving or ideation.
- You can create tasks that require collaboration with other agents.
"""


class DeepAgent:
    """DeepDantic AI Agent that manages sub-agents to achieve complex goals."""

    def __init__(self, prompt, model="gpt-4.1-mini", max_steps=20, token_budget=20000):
        """
        DeepAgent constructor.

        :param prompt: The overall goal or prompt for the agent.
        :param model: The language model to use. default is "gpt-4.1-mini".
        :param max_steps: Maximum steps to prevent infinite loops. defaults to 20 steps
        :param token_budget: Token budget for the agent's operations. default is 20000 tokens.
        """
        self.model = model
        self.prompt = prompt  # Goal Prompt
        self._max_steps = 20  # Max steps to prevent infinite loops

        self._planner_agent = Agent(
            model=model, system_prompt=PLANNER_SYSTEM_PROMPT, output_type=Plan
        )
        self._supervisor_agent = Agent(
            model=model,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            output_type=NextAction,
            deps_type=RuntimeState,
        )

    def _apply_action(self, action: NextAction, state: RuntimeState):
        """Apply the chosen action to the runtime state."""
        if action.action_type == "create_todo" and action.create:
            for todo in action.create:
                new_todo = TodoItem(
                    id=str(uuid.uuid4()),
                    description=todo.description,
                    status="pending",
                )
                state.todos.todos.append(new_todo)

        elif action.action_type == "update_todo" and action.update:
            for todo in state.todos:
                if todo.id == action.update.todo_id:
                    todo.status = action.update.status
                    break
        # Additional action types like 'delegate' and 'complete' would be handled here

    def _run_supervisor(self):
        # Logic to run the supervisor agent and manage sub-agents
        pass

    def run(self):
        # Start the supervisor agent to manage sub-agents
        # state = RuntimeState(goal=self.prompt)
        agent_plan = self._planner_agent.run_sync(self.prompt).output

        # Run the supervisor to decide next actions from the plan
        # self._apply_action(supervisor_response, state)
        for todo in agent_plan.tasks:
            print(
                f"- [{todo.status}] {todo.description} (id: {todo.id}) capability: {todo.capability})"
            )
        # print(
        #     "Planner Response:",
        #     agent_plan,
        # )

        # for _step in range(self._max_iterations):  # Limit to 5 steps for demo purposes
        #     print(f"\n--- Supervisor Step {step + 1} ---")
        #     supervisor_response = self._supervisor_agent.run_sync(
        #         "Decide the next action.", deps=agent_plan
        #     ).output
        #     print("Supervisor Response:", supervisor_response)
        # Here you would parse the supervisor response and create/manage sub-agents accordingly
        # For simplicity, we will just print the response for now


from dotenv import load_dotenv

load_dotenv()
agent = DeepAgent(
    "Help me plan a trip to japan. I want to see cultural sites, tourist sites, and eat good food.",
    "gpt-4.1-mini",
)
agent.run()
