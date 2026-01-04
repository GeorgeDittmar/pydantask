# from asyncio import tasks
import json
import uuid
import os

from os import system


from pydantic_ai import Agent, RunContext, FunctionToolset
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict

import asyncio
from pydantic_ai import RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from prompts import PLANNER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT

# =========================
# Runtime state models
# =========================


from dotenv import load_dotenv

load_dotenv()


class TaskItem(BaseModel):
    id: str
    description: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Plan = Field(default_factory=Plan)
    agent_registry: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    completed_steps: set[int] = Field(default_factory=set)
    research_results: Dict[int, ResearchResult] = Field(default_factory=dict)
    knowledge_store: Dict[str, str] = Field(
        default_factory=dict
    )  # store for accumulated knowledge
    iteration: int = 0
    tokens_used: int = 0
    tool_available: Literal["research_agent", "writer_agent", "web_search"] = (
        "web_search"
    )
    goal: str = Field(
        default="Perform a websearch for microsoft and tell me what you know."
    )


class NextAction(BaseModel):
    """Next action to be taken by the supervisor agent."""

    reasoning_summary: str
    action_type: Literal["delegate", "complete"]
    target_agent: Optional[str] = None
    task_spec: Optional[TaskItem] = None


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


def supervisor_system_prompt(ctx: RunContext[RuntimeState]) -> str:
    return f"""You are the supervisor agent in a deep agent system.
Your job is to manage and delegate tasks to sub-agents to achieve the overall goal.

You have the following runtime state:

{ctx.agent_state.model_dump_json(indent=2)}

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


class SupervisorSpec:
    def system_prompt(self, ctx: RunContext) -> str:
        return f"""You are the supervisor agent in a deep agent system.
Your job is to manage and delegate tasks to sub-agents and tools to achieve the overall goal.

Job plan:

{ctx.deps.plan}

You have the following capabilities available to use:

{ctx.dep.agent_registory.keys()}

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
"""

    def tools(self):
        return [
            self.mark_task_done,
        ]

    def mark_task_done(self, ctx, task_id: str):
        ctx.deps.tasks[task_id].done = True
        return f"Task {task_id} marked done."


# Your Registry stays clean
# registry = {
#     "researcher": deep_research_agent,  # Heavy reasoning
#     "coder": coding_agent,  # Heavy reasoning
#     "files": file_agent,  # Atomic / Utility
# }


# default_registry = {"websearch": search_web}

from pydantic_ai.common_tools.tavily import (
    TavilySearchTool,
    TavilySearchResult,
    tavily_search_tool,
)


class SearchQuery(BaseModel):
    search_query: str = Field(default_factory="")


api_key = os.getenv("TAVILY_API_KEY")
assert api_key is not None

# A "Thin" Agent that just wraps a tool
web_agent = Agent(
    "gpt-4.1-mini",  # Use a cheap model for simple tasks
    system_prompt="You are a search specialist. When given a task, follow these steps: "
    "1. Generate 3 specific search queries to cover the topic. "
    "2. Call the search tool for each query. "
    "3. Summarize the findings into a clear report.",
    tools=[tavily_search_tool(api_key)],
    output_type=str,
)


class DeepAgent:
    """DeepDantic AI Agent that manages sub-agents to achieve complex goals."""

    def __init__(
        self,
        prompt,
        model="gpt-4.1-mini",
        max_steps=20,
        token_budget=20000,
        agent_registry={},
    ):
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
        self.agent_registry = agent_registry
        # Initialize planner and supervisor agents
        # Planner agent to break down the goal into tasks
        # Supervisor agent to manage todos and delegate tasks
        self._planner_agent = Agent(
            model=model, system_prompt=PLANNER_SYSTEM_PROMPT, output_type=Plan
        )

    def _setup_default_agents(self):

        return {"researcher": }
    
    def _create_supervisor(self, tools=None, model="openai:gpt-4.1-mini") -> Agent:
        spec = SupervisorSpec()
        agent = Agent(
            model="openai:gpt-4.1-mini",
            deps_type=RuntimeState,
            # output_type=RuntimeState,
            tools=tools,
        )

        @agent.system_prompt
        def _prompt(ctx):
            return spec.system_prompt(ctx)

        for tool in spec.tools():
            agent.tool(tool)

        return agent

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
        supervisor_agent = self._create_supervisor(tools=[self.call_worker])

        # plan = planner(current_state)
        # for step in plan:
        #     result = executor.run(step)
        #     supervisor.observe(step, result)
        #     if supervisor.detects_issue(result):
        #         supervisor.correct(step, current_state)

        step = 0
        while step < self._max_steps or runtime_state.tokens_used < self.token_budget:
            print(f"\n--- Supervisor Step {step + 1} ---")
            supervisor_response = supervisor_agent.run_sync(
                "Execute the plan given the runtime state and knowledge you know.",
                deps=runtime_state,
            ).output
            print(supervisor_response)
            break
            # Here you would create and run the sub-agent based on the instructions
            # For simplicity, we will just print the instructions for now
            step += 1

    # Tools used by the supervisor or planner agents
    async def get_system_time(ctx: RunContext[RuntimeState]) -> str:
        """Use this to get the current server time for scheduling or needing to know the date"""
        from datetime import datetime

        return datetime.now().isoformat()

    async def execute_ready_tasks_tool(self, ctx: RunContext[RuntimeState]) -> str:
        """Finds all tasks ready to run and executes them in parallel."""
        plan = ctx.deps.plan

        # 1. Identify "Ready" tasks
        ready_steps = [
            step
            for step in plan.steps
            if step.status == "pending"
            and all(
                plan.get_step(d_id).status == "completed" for d_id in step.depends_on
            )
        ]

        if not ready_steps:
            return "No tasks are currently ready for execution."

        # 2. Prepare the concurrent coroutines
        tasks = []
        for step in ready_steps:
            # grab the tool that the plan or supervisor decides
            worker = ctx.deps.registry.get(step.assigned_to)
            if worker:
                step.status = "running"
                # We wrap the agent run in a small wrapper to update the step status after
                tasks.append(self.run_and_update_step(worker, step))

        # 3. Execute all at once
        results = await asyncio.gather(*tasks)

        return f"Executed {len(results)} tasks in parallel."

    async def search_web(self, search_query):
        """Take a search query and search the internet for information to solve a research / task need."""
        return web_agent.run(search_query).output

    async def run_and_update_step(self, worker, step):
        """Helper to run an agent and capture its output into the step object."""
        try:
            result = await worker.run(step.description)
            step.output = result.data
            step.status = "completed"
            return f"Step {step.label} success"
        except Exception as e:
            step.status = "failed"
            return f"Step {step.label} failed: {str(e)}"

    def update_step_status(
        ctx: RunContext[RuntimeState],
        step_id: int,
        status: Literal["pending", "completed", "failed"],
        output: str | None = None,
    ) -> str:
        """Updates the status and output of a specific step in the plan."""
        plan = ctx.deps.plan
        for step in plan.steps:
            if step.id == step_id:
                step.status = status
                if output:
                    step.output = output
                return f"Step {step_id} updated to {status}."
        return f"Error: Step {step_id} not found."

    # 1. ATOMIC TOOLS (The Manager's "Desk Tools")
    def update_task_status(ctx: RunContext[RuntimeState], step_id: int, status: str):
        """Directly updates the plan. No other agent needed."""
        ctx.deps.plan.get_step(step_id).status = status
        return f"Status for {step_id} is now {status}."

    # 2. THE BRIDGE TOOL (The Manager's "Phone")
    async def call_worker(
        ctx: RunContext[RuntimeState], capability: str, instruction: str
    ):
        """The Supervisor calls this to hand off a task to the Registry."""
        # Find the agent in the registry
        worker_agent = ctx.deps.registry.get(capability)

        if not worker_agent:
            return f"Error: No specialist found for '{capability}'."

        # Trigger the deep reasoning loop of the sub-agent
        result = await worker_agent.run(instruction)

        # Return just the result to the Supervisor
        return result.data


print(web_agent.run_sync("Look up top tourist locations japan").output)

agent = DeepAgent(
    "Help me plan a trip to japan. I want to see cultural sites, tourist sites, and eat good food.",
    "gpt-4.1-mini",
    max_steps=2,
)
agent.run()
