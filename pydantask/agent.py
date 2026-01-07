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
from .prompts import PLANNER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT

# =========================
# Runtime state models
# =========================


from dotenv import load_dotenv

load_dotenv()
from langfuse import get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

Agent.instrument_all()


class TaskItem(BaseModel):
    id: str
    description: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    owner: Optional[str] = None
    capability: Literal[
        "researcher",
        "writer",
        "synthesiser",
        # "external_interaction",
        # "creative_problem_solving",
        # "collaboration",
    ] = "research"
    task_dependencies: Optional[List[int]] = Field(default_factory=list)


# =========================
# Planner state models
# =========================
class Plan(BaseModel):
    tasks: list[TaskItem]


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


def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
    return f"""You are the supervisor agent in a deep agent system.
    Your job is to manage and delegate tasks to sub-agents and tools to achieve the overall goal.

    Job plan:

    {ctx.deps.plan}

    You have the following capabilities available to use:

    {ctx.deps.agent_registry.keys()}

    Based on this information, you must decide the next action to take. You must not modify the tasks if you decide to delegate a task. You must pick the task as is from the plan.

    Be sure to follow these rules when deciding what to do:

    ###Rules:
    - You must check if there are any pending tasks in the plan.
    - You must check if any tasks have been completed.
    - You must check the results of completed tasks to inform your decision.
    - You must check task dependencies before delegating a task.
    - You must always consider the overall goal when deciding the next action.
    - You must prioritize tasks that unblock progress towards the goal.
    - You must delegate tasks to sub-agents based on their capabilities.
    - You must not delegate tasks that have already been completed.

    Decide which capability to use that are available to you.
    """


class SubAgentInstruction(BaseModel):
    reasoning: Optional[str] = None
    instructions: str


class TaskSpec(BaseModel):
    task_id: str
    objective: str
    capability: Literal["researcher", "writer", "synthesizer"]
    inputs: dict
    success_criteria: str
    constraints: list[str]
    overall_goal: str


# def __subagent_instruction_writer(goal: str, taskspec: TaskSpec) -> SubAgentInstruction:
#     """Write instructions for the sub-agent that explains its task in detail and how it acheives the overall goal.
#     return str: Instructions for the sub-agent.
#     """
#     subagent_task_instructions = f"""
#     You are a sub-agent task writer in a deep agent system.
#     You take the goal and a task specification
#     and use your capabilities to write clear instructions that the supervisor can give to a sub-agent to complete the task.

#     The overall goal is: {goal}

#     The task specification is: {taskspec.model_dump_json(indent=2)}

#     To achieve this task, you should:
#     1. Understand the overall goal and how your task contributes to it.
#     2. Use your capabilities to complete the task effectively.
#     3. Report your findings or results back to the supervisor agent.
#     4. Ensure that your work aligns with the overall goal.
#     5. If you encounter any challenges, think creatively to overcome them.
#     Good luck!"""

#     __subagent_writer_agent = Agent(
#         model="gpt-4.1-mini",
#         output_type=SubAgentInstruction,
#         deps_type=TaskSpec,
#     )

#     return None


class SupervisorSpec:
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        print("SYSTEM PROMPT")
        return f"""You are the supervisor agent in a deep agent system.
Your job is to manage and delegate tasks to sub-agents and tools to achieve the overall goal.

Job plan from planning agent:

{ctx.deps.plan}

You have the following capabilities tied to agents available to use:

{ctx.deps.agent_registry}

You must reason out and decide which task from the plan to run and use the call_worker tool to call the worker to run said task.
If a task is completed use the update_state 

Think step by step keeping the following rules in mind:
1. Reason out why you need to call a capability
2. Come up with an instruction to give to the capability
3. Make sure that the instruction you give is related to the task.
4. When reviewing a sub agent or tools response be sure it solves the task that was given. 
5. If the response from a sub agent or tool does not solve the task, attempt to retry, otherwise set the task to 'failed'
"""

    # def tools(self):
    #     return [
    #         self.mark_task_done,
    #     ]

    # def mark_task_done(self, ctx, task_id: str):
    #     ctx.deps.tasks[task_id].done = True
    #     return f"Task {task_id} marked done."


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
    search_query: str


synth_agent_sys_prompt = """
You take information from various sources and synthesize a response for the goal
"""


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
        self.agent_registry = self._setup_default_agents()
        # Initialize planner and supervisor agents
        # Planner agent to break down the goal into tasks
        # Supervisor agent to manage todos and delegate tasks
        self._planner_agent = Agent(
            model=self.model, system_prompt=PLANNER_SYSTEM_PROMPT, output_type=Plan
        )

    def _setup_default_agents(self):

        api_key = os.getenv("TAVILY_API_KEY")
        assert api_key is not None

        synthesizer = Agent(
            "gpt-4.1-mini",
            instructions=synth_agent_sys_prompt,
            output_type=str,
        )

        # A "Thin" Agent that just wraps a tool
        researcher_agent = Agent(
            "gpt-4.1-mini",  # Use a cheap model for simple tasks
            instructions="You are a research specialist. When given a task, follow these steps: "
            "1. Generate 3 specific search queries to cover the topic. "
            "2. Call the search tool for each query. "
            "3. Summarize the findings into a clear report.",
            tools=[tavily_search_tool(api_key)],
            output_type=str,
        )
        return {"researcher": researcher_agent, "synthesizer": synthesizer}

    def _create_supervisor(self, tools=None, model="openai:gpt-4.1-mini") -> Agent:
        spec = SupervisorSpec()
        agent = Agent(
            model=model,
            deps_type=RuntimeState,
            tools=tools,
        )

        @agent.system_prompt
        def _prompt(ctx):
            return spec.system_prompt(ctx)

        return agent

    def _initialize_runtime_state(
        self, plan: Plan, goal: str, registry: dict
    ) -> RuntimeState:
        # Logic to initialize and manage the runtime state
        return RuntimeState(plan=plan, goal=goal, agent_registry=registry)

    def run(self):
        # Start the supervisor agent to manage sub-agents
        # state = RuntimeState(goal=self.prompt)
        agent_plan = self._planner_agent.run_sync(self.prompt).output
        print(agent_plan)
        # Run the supervisor to decide next actions from the plan
        # self._apply_action(supervisor_response, state)
        for todo in agent_plan.tasks:
            print(
                f"- [{todo.status}] {todo.description} (id: {todo.id}) capability: {todo.capability}) owner: {todo.owner} dependencies: {todo.task_dependencies}"
            )

        # now save the plan to the agent state
        runtime_state = self._initialize_runtime_state(
            plan=agent_plan, goal=self.prompt, registry=self.agent_registry
        )

        print("\n--- Initial Runtime State ---")
        print(runtime_state)
        supervisor_agent = self._create_supervisor(
            tools=[self.call_worker, self.update_task_status]
        )
        # plan = planner(current_state)
        # for step in plan:
        #     result = executor.run(step)
        #     supervisor.observe(step, result)
        #     if supervisor.detects_issue(result):
        #         supervisor.correct(step, current_state)

        supervisor_response = supervisor_agent.run_sync(
            "Execute the plan given the runtime state and knowledge you know.",
            deps=runtime_state,
        ).output
        print(supervisor_response)

        # Here you would create and run the sub-agent based on the instructions
        # For simplicity, we will just print the instructions for now

    # Tools used by the supervisor or planner agents
    async def get_system_time(self, ctx: RunContext[RuntimeState]) -> str:
        """Use this to get the current server time for scheduling or needing to know the date"""
        from datetime import datetime

        return datetime.now().isoformat()

    async def review_plan(self, ctx: RunContext[RuntimeState]):
        """Tool to review the current state of the plan."""
        return f"Current state of the plan: {ctx.deps.plan}"

    async def update_knowledge(
        self, capabiliity, answer, ctx: RunContext[RuntimeState]
    ):
        """Updates the knowledge runtime state with any new knowledge that is needed to answer a goal or task"""

    async def execute_ready_tasks(self, ctx: RunContext[RuntimeState]) -> str:
        """Finds all tasks that are ready to run and executes them in parallel."""
        plan = ctx.deps.plan

        # 1. Identify "Ready" tasks
        ready_steps = [
            step
            for step in plan.tasks
            if step.status == "pending"
            and all(plan.get(d_id).status == "completed" for d_id in step.depends_on)
        ]

        if not ready_steps:
            return "No tasks are currently ready for execution."

        # 2. Prepare the concurrent coroutines
        tasks = []
        for step in ready_steps:
            # grab the tool that the plan or supervisor decides
            worker = ctx.deps.agent_registry.get(step.capability)
            if worker:
                step.status = "running"
                # We wrap the agent run in a small wrapper to update the step status after
                tasks.append(self.run_and_update_step(worker, step))

        # 3. Execute all at once
        results = await asyncio.gather(*tasks)

        return f"Executed {len(results)} tasks in parallel."

    async def search_web(self, search_query):
        """Take a search query and search the internet for information to solve a research / task need."""
        return await res.run(search_query).output

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

    # 1. ATOMIC TOOLS (The Manager's "Desk Tools")
    async def update_task_status(
        self, ctx: RunContext[RuntimeState], step_id: int, status: str
    ):
        """Directly updates the plan. No other agent needed."""
        for step_id in ctx.deps.plan.tasks:
            ctx.deps.plan.tasks.get(step_id).status = status
            return f"Status for {step_id} is now {status}."
        return f"Error: No step with {step_id} found in plan. Be sure status_id actually exists."

    # 2. THE BRIDGE TOOL (The Manager's "Phone")
    async def call_worker(
        self, ctx: RunContext[RuntimeState], capability: str, instruction: str
    ):
        """The Supervisor calls this to hand off a task to the Registry."""
        # Find the agent in the registry
        worker_agent = ctx.deps.agent_registry.get(capability, None)

        if not worker_agent:
            return f"Error: No specialist found for '{capability}'."
        print(f"AGENT: {capability} INSTRUCTION: {instruction}")
        # Trigger the deep reasoning loop of the sub-agent
        result = await worker_agent.run(instruction)
        print(result)
        # Return just the result to the Supervisor
        return result


# print(web_agent.run_sync("Look up top tourist locations japan").output)

# agent = DeepAgent(
#     "Help me plan a trip to japan. I want to see cultural sites, tourist sites, and eat good food.",
#     "gpt-4.1-mini",
#     max_steps=2,
# )
# agent.run()
