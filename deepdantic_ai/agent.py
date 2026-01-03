# from asyncio import tasks
import json
import uuid
from os import system

from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from transformers import RobertaForQuestionAnswering


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
        # "external_interaction",
        # "creative_problem_solving",
        # "collaboration",
    ] = "research"
    task_dependencies: Optional[List[int]] = Field(default_factory=list)


# =========================
# Planner state models
# =========================
class Plan(BaseModel):
    tasks: List[TaskItem] = Field(default_factory=list)


# class RuntimeState(BaseModel):
#     goal: str
#     plan: PlanState
#     last_event: Optional[str] = None


# class ResearcherAgent:
#     """Researcher Agent that performs research tasks."""

#     class AgentState(BaseModel):
#         goal: str
#         completed_steps: List[str] = Field(default_factory=list)
#         step_results: dict[str, dict] = Field(default_factory=dict)
#         steps: List["ResearchStep"] = Field(default_factory=list)
#         hypotheses: List[str] = Field(default_factory=list)
#         evidence: List[dict] = Field(default_factory=list)
#         claims: List[str] = Field(default_factory=list)
#         confidence: float = 0.0


class ResearchResult(BaseModel):
    findings: list[str]
    sources: list[str]
    confidence: float


class RuntimeState(BaseModel):
    plan: Plan = Field(default_factory=Plan)
    completed_steps: set[int] = Field(default_factory=set)
    research_results: dict[int, ResearchResult] = Field(default_factory=dict)
    knowledge_store: dict[str, str] = Field(
        default_factory=dict
    )  # store for accumulated knowledge
    iteration: int = 0
    tokens_used: int = 0
    tool_available: Literal["research_agent", "writer_agent", "web_search"] = (
        "web_search"
    )
    goal: str


class NextAction(BaseModel):
    """Next action to be taken by the supervisor agent."""

    reasoning_summary: str
    action_type: Literal["delegate", "complete"]
    target_agent: Optional[str] = None
    task_spec: Optional[TaskItem] = None


def _mocked_websearch(query: str) -> dict:
    """Mocked web search function for demonstration purposes."""
    return {
        "query": query,
        "results": [
            {
                "title": "Japan has many interesting palces to visit",
                "url": "http://example.com/1",
                "snippet": "Tokyo Disneyland is a popular tourist destination.",
            },
            {
                "title": "Japanese food places are top",
                "url": "http://example.com/2",
                "snippet": "Jiro Ono's sushi restaurant in Tokyo is world-renowned. The ramen shops in Fukuoka are also a must-visit.",
            },
        ],
    }


# =========================
# Supervisor input/output
# =========================
class SupervisorDeps(BaseModel):
    goal: str
    plan: Plan
    last_event: Optional[str] = None


def _default_tool_registry():
    return {
        "research_agent": {
            "description": "An agent that can perform research tasks using web search and data retrieval.",
            "capabilities": ["research", "analysis"],
        },
        "writer_agent": {
            "description": "An agent that can synthesize information and generate written content.",
            "capabilities": ["synthesis", "creative_problem_solving"],
        },
    }


# =========================
# planner agent to determine tasks to perform to achieve the goal
# =========================

PLANNER_SYSTEM_PROMPT = """
Your job is to break down the goal into manageable discrete tasks that can be delegated to sub agents.
You will output a Plan object containing a list of TaskItems.
Be sure to follow these rules when creating the tasks:

###Rules:
- You must prioritize tasks that unblock progress towards the goal.
- You must not create duplicate tasks.  
- Each task should be clear and specific.
- You must make tasks actionable and clear.
- Tasks should be concise, ideally under 10 words.
- When creating multiple tasks, ensure they are distinct and cover different aspects of the goal.
- If the goal is complex, break it down into at least 5 distinct tasks.
- If any task depends on another, specify the dependency using task_dependencies using the task ids.
- Tasks should be ordered in a way that respects dependencies.
- Be sure that the synthesis of all tasks leads to achieving the overall goal and is the final task.

There are several types of tasks you can create based on the capabilities of your sub-agents.

###Capabilities:
- You can create tasks that require 'research' or 'synthesis' of information.
- You can create tasks that require interaction with external systems or APIs.
- You can create tasks that require creative problem solving or ideation.
- You can create tasks that require collaboration with other agents.
"""


SUPERVISOR_SYSTEM_PROMPT = """
You are the supervisor agent in a deep agent system.
Your job is to manage and delegate tasks to sub-agents to achieve the overall goal.

You will be provided with the current runtime state, including the plan of tasks to be completed, and the results of any completed tasks.
Based on this information, you must decide the next action to take. You must not modify the tasks if you decide to delegate a task. You must pick the task as is from the plan.

Be sure to follow these rules when deciding the next action:

###Rules:
- You must check if there are any pending tasks in the plan.
- You must check if any tasks have been completed.
- You must check the results of completed tasks to inform your decision.
- You must check task dependencies before delegating a task.
- You must always consider the overall goal when deciding the next action.
- You must prioritize tasks that unblock progress towards the goal.
- You must delegate tasks to sub-agents based on their capabilities.
- You must not delegate tasks that have already been completed.
- You must provide a clear reasoning summary for your decision.
- You must only output a single NextAction object in your response. 

Do not include any additional text or explanation.
Do not output anything other than the NextAction object.
Do not modify the plan directly.


If you decide to delegate a task, specify the target_agent and any necessary payload for the agent. When delegating, pick a task from the plan that is pending and whose dependencies have been met.

Output a valid NextAction object only.
"""

GENERIC_SUB_AGENT_SYSTEM_PROMPT = """
You are a sub-agent in a deep agent system.
Your job is to complete the task assigned to you by the supervisor agent.
You will be provided with the task description and any necessary context.
You must complete the task to the best of your ability and report your findings back to the supervisor agent.

###Rules:
- You must always consider the overall goal when completing your task.
- You must use your capabilities to complete the task effectively.
- You must report your findings or results back to the supervisor agent.
- You must ensure that your work aligns with the overall goal.
- If you encounter any challenges, think creatively to overcome them.
- You must only output the results of your task in a clear and concise manner.
"""


class SubAgentInstruction(BaseModel):
    reasoning: Optional[str] = None
    instructions: str


class TaskSpec(BaseModel):
    task_id: str
    objective: str
    capability: Literal["research", "analysis", "synthesis"]
    inputs: dict
    success_criteria: str
    constraints: list[str]
    overall_goal: str


def __subagent_instruction_writer(goal: str, taskspec: TaskSpec) -> SubAgentInstruction:
    """Write instructions for the sub-agent that explains its task in detail and how it acheives the overall goal.
    return str: Instructions for the sub-agent.
    """
    subagent_task_instructions = f"""
    You are a sub-agent task writer in a deep agent system.
    You take the goal and a task specification
    and use your capabilities to write clear instructions that the supervisor can give to a sub-agent to complete the task.

    The overall goal is: {goal}
    
    The task specification is: {taskspec.model_dump_json(indent=2)}
    
    To achieve this task, you should:
    1. Understand the overall goal and how your task contributes to it.
    2. Use your capabilities to complete the task effectively.
    3. Report your findings or results back to the supervisor agent.
    4. Ensure that your work aligns with the overall goal.
    5. If you encounter any challenges, think creatively to overcome them.
    Good luck!"""

    __subagent_writer_agent = Agent(
        model="gpt-4.1-mini",
        output_type=SubAgentInstruction,
        deps_type=TaskSpec,
    )

    return __subagent_writer_agent.run_sync(subagent_task_instructions).output


class DeepAgent:
    """DeepDantic AI Agent that manages sub-agents to achieve complex goals."""

    def __init__(self, prompt, model="gpt-4.1-mini", max_steps=20, token_budget=20000):
        """
        DeepAgent constructor. Creates a DeepDantic AI Agent.

        :param prompt: The overall goal or prompt for the agent.
        :param model: The language model to use. default is "gpt-4.1-mini".
        :param max_steps: Maximum steps to prevent infinite loops. defaults to 20 steps
        :param token_budget: Token budget for the agent's operations. default is 20000 tokens.
        """
        self.model = model
        self.prompt = prompt  # Goal Prompt
        self._max_steps = max_steps  # Max steps to prevent infinite loops
        self.token_budget = token_budget  # Token budget for the agent's operations
        self.tool_registry = _default_tool_registry()
        self.agent_registry = {}
        # Initialize planner and supervisor agents
        # Planner agent to break down the goal into tasks
        # Supervisor agent to manage todos and delegate tasks
        self._planner_agent = Agent(
            model=model, system_prompt=PLANNER_SYSTEM_PROMPT, output_type=Plan
        )
        self._supervisor_agent = Agent(
            model=model,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            output_type=NextAction,
            deps_type=RuntimeState,
        )

    def _generate_subagent_instructions(
        self, goal: str, task: TaskSpec
    ) -> SubAgentInstruction:
        """Generate instructions for a sub-agent based on the goal and task."""
        return __subagent_instruction_writer(goal, task.description)

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

    def _initialize_runtime_state(self, plan: Plan, goal: str) -> RuntimeState:
        # Logic to initialize and manage the runtime state
        return RuntimeState(plan=plan, goal=goal)

    def run(self):
        # Start the supervisor agent to manage sub-agents
        # state = RuntimeState(goal=self.prompt)
        agent_plan = self._planner_agent.run_sync(self.prompt).output

        # Run the supervisor to decide next actions from the plan
        # self._apply_action(supervisor_response, state)
        for todo in agent_plan.tasks:
            print(
                f"- [{todo.status}] {todo.description} (id: {todo.id}) capability: {todo.capability}) owner: {todo.owner} dependencies: {todo.task_dependencies}"
            )

        # now save the plan to the agent state
        runtime_state = self._initialize_runtime_state(
            plan=agent_plan, goal=self.prompt
        )

        print("\n--- Initial Runtime State ---")
        print(runtime_state)

        step = 0
        while step < self._max_steps or runtime_state.tokens_used < self.token_budget:
            print(f"\n--- Supervisor Step {step + 1} ---")
            supervisor_response = self._supervisor_agent.run_sync(
                deps=runtime_state
            ).output

            # if supervisor_response.action_type == "complete":
            #     print("Goal completed by supervisor.")
            #     break
            # if supervisor_response.action_type == "delegate":
            #     task_spec = supervisor_response.task_spec
            #     if task_spec is None:
            #         print("No task specification provided for delegation.")
            #         break

            #     # Generate sub-agent instructions
            #     subagent_instructions = self._generate_subagent_instructions(
            #         goal=runtime_state.goal,
            #         task=TaskItem(
            #             id=task_spec.task_id,
            #             description=task_spec.objective,
            #             status="pending",
            #             capability=task_spec.capability,
            #         ),
            #     )
            #     print(
            #         f"Delegating Task ID {task_spec.task_id} to {supervisor_response.target_agent}"
            #     )
            #     print("Sub-Agent Instructions:", subagent_instructions.instructions)
            # Here you would create and run the sub-agent based on the instructions
            # For simplicity, we will just print the instructions for now
            print("Supervisor Response:", supervisor_response)
            step += 1
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
    max_steps=2,
)
agent.run()
