# from asyncio import tasks
from email import errors
from json import tool

from langfuse import get_client
import json
import uuid
import os

import logging
from os import system

from enum import Enum
from pydantic_ai import Agent, RunContext, FunctionToolset
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict, Callable, Union

import asyncio
from asyncio import TaskGroup
from pydantic_ai import RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from .prompts import PLANNER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
from .models import (
    ResearchResult,
    RuntimeState,
    TaskItem,
    Plan,
    TaskQAResult,
    TaskStatus,
    NextAction,
    SupervisorDecision,
    ToolDescription,
    TaskResult,
)
from .default_tools import (
    write_to_file_system,
    read_from_file_system,
    think_tool,
    ask_user,
)

from pydantic_ai.common_tools.tavily import (
    tavily_search_tool,
)

from pprint import pprint

# =========================
# Runtime state models
# =========================


from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

Agent.instrument_all()


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

    {ctx.deps.agent_registry}

    You must look at the current state of the job plan, and decide which task or tasks should be run next.
    Honor task dependencies and do not start tasks whose dependencies are not 'COMPLETE'.

    If any task is given the state 'ERROR' or 'REPLAN' delegate to the planner to attempt to rework the plan
    for that task.

    When delegating a task set the status to 'DELEGATE'. You may delegate multiple tasks if they can be run independently.
    """


class SupervisorSpecv2:
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return f"""
        You are a supervisor of a plan to perform a goal. You must execute the 
        plan as described and only make modifications or update the plan
        if there are issues in completing steps or needing to do more work.
        
        Goal: {ctx.deps.goal}

        --- OPERATIONAL STATE ---
        CURRENT PLAN:
        {json.dumps({k: v.model_dump() for k, v in ctx.deps.plan.items()}, indent=2)}

        KNOWLEDGE STORE (What we know so far):
        {ctx.deps.knowledge_store}

        --- YOUR OPERATIONAL MANDATE ---
        1. Identify the next unblocked task in the Plan.
        2. Delegate that task using the 'call_worker' tool.
        3. Once you receive the result (Verified/Failed), update the state.
        4. IMPORTANT: After updating the state for ONE task, stop and provide a brief status update. 
        5. DO NOT attempt to run the entire plan in one turn.

        --- YOUR RESPONSIBILITIES ---
        1. ANALYZE: Which tasks are 'pending' and have their dependencies 'completed'?
        2. EVALUATE: Look at the Knowledge Store. Is there enough info to skip a task or does a task need revision?
        3. DELEGATE: Use 'call_worker' for the next logical task. 
        4. REFLECT: When a worker returns data, you must decide: Is it good enough? If not, use feedback to ask for a retry.
        5. EVOLVE: If you discover a new sub-goal is needed, use 'add_task_to_plan'.

        --- RULES ---
        - Do not perform work yourself. Delegate it to the appropriate capability worker.
        - If a worker fails, analyze the error and decide to retry or fail the task.
        - Once the goal is fully satisfied, provide the final synthesis.

        --- TERMINATION CRITERIA ---
        - When ALL tasks in the plan are 'completed', call 'generate_final_output' and return the result as your final answer.
        - If a critical task is 'failed' and cannot be recovered, explain the blocker to the user.
        """


class SupervisorSpec:
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        return f"""You are the supervisor agent in a deep agent system.
        Your job is to manage and delegate tasks to sub-agents and tools to achieve the overall goal.

        You have the following capabilities available to delegate work to

        {ctx.deps.agent_registry}

        Current State of job plan:

        {ctx.deps.plan}

        You must look at the current state of the job plan, and decide which task or tasks should be run next.
        Honor task dependencies and do not start tasks whose dependencies are not COMPLETED.

        If any task is given the state ERROR or REPLAN delegate to the planner to attempt to rework the plan
        for that task.

        You may delegate multiple tasks if they can be run independently. When delegating a task set the status to READY.
        """

    def tools(self):
        return [
            self.mark_task_done,
        ]

    def mark_task_done(self, ctx, task_id: str):

        if task_id not in ctx.deps.tasks:
            return f"Could not find {task_id} in plan. Try again and make sure you provide the correct task id."

        ctx.deps.tasks[task_id].done = True
        return f"Task {task_id} marked complete."


# Your Registry stays clean
# registry = {
#     "researcher": deep_research_agent,  # Heavy reasoning
#     "coder": coding_agent,  # Heavy reasoning
#     "files": file_agent,  # Atomic / Utility
# }


# default_registry = {"websearch": search_web}


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
        model="openai:gpt-4.1-mini",
        critic_model="openai:gpt-4.1-mini",
        max_steps=3,
        set_token_budget: int = None,
        tools: Union[None, list[ToolDescription]] = None,
        human_feedback: bool = False,
    ):
        """
        DeepAgent constructor. Creates a DeepDantic AI Agent.

        :param prompt: The overall goal or prompt for the agent.
        :param model: The language model to use. default is "gpt-4.1-mini".
        :param max_steps: Maximum steps to prevent infinite loops. defaults to 20 steps
        :param set_token_budget: Token budget for the agent's operation. Defaults to None (no limit).
        :param tools: List of ToolDescription objects representing the agent's capabilities. Defaults to None.
        :param human_feedback: Whether to incorporate human feedback in the agent's decision-making. Defaults to False.
        """
        self.model = model
        self.prompt = prompt  # Goal Prompt
        self._max_steps = max_steps  # Max steps to prevent infinite loops
        self.token_budget = set_token_budget
        self.agent_registry = self._setup_default_tools(tools=tools)
        self.critic_model = critic_model
        self._planner_agent = Agent(
            model=self.model, system_prompt=PLANNER_SYSTEM_PROMPT, output_type=Plan
        )

        self._critic_agent = Agent(
            self.critic_model,
            system_prompt="Evaluate the following output from work done on a task. Output a detailed report and if it meets the task requirements.",
            output_type=TaskQAResult,
            tools=[read_from_file_system, think_tool],
        )

    def _setup_default_tools(self, tools: Union[None, list[ToolDescription]] = None):
        """
        Setup default tools along with any custom tools that may be provided by the caller.
        Default tools available are synthesizer, researcher, and file_system agents

        Args:
            tools (Union[None, list[ToolDescription]], optional): Any custom tools to include in the agent. Defaults to None.

        Returns:
            dict[str, ToolDescription]: Mapping of toolId's to the tool description and function
        """
        api_key = os.getenv("TAVILY_API_KEY", None)

        if not api_key:
            raise ValueError(
                "Tavily search api key not found or provided in env variables"
            )

        synthesizer_agent = Agent(
            self.model,
            instructions=synth_agent_sys_prompt,
            output_type=str,
            tools=[write_to_file_system, read_from_file_system],
        )

        synthesizer = ToolDescription(
            description="Generate answers based on information collected from tasks.",
            tool_func=synthesizer_agent,
        )

        # A "Thin" Agent that just wraps a tool
        researcher_agent = Agent(
            self.model,  # Use a cheap model for simple tasks
            instructions="You are a research specialist. When given a task, follow these steps: "
            "1. Generate 5 search queries based on the task description. Start with broad queries and narrow down."
            "2. Call the search tool for each query. "
            "3. Analyze the results from the search tool and extract key information."
            "4. Use the think tool to reflect on the information gathered and determine if another search is needed."
            "5. If enough information has been gathered, summarize the findings into a clear report."
            " Be concise and focus on the most relevant information to the task."
            " When building the search report, generate a detailed report and a summary."
            "When genrating the detailed report, include references to sources used. Write the detailed report to a markdown file and return the file path in the detailed_report_path field of the output.",
            tools=[
                tavily_search_tool(api_key),
                think_tool,
                write_to_file_system,
                read_from_file_system,
            ],
            output_type=ResearchResult,
        )

        researcher = ToolDescription(
            description="Tool to research information. This could include searching the web or querying a data source.",
            tool_func=researcher_agent,
        )

        file_system_agent = Agent(
            self.model,
            instructions="You have access to a file system to use for tasks that need to be completed. \
            Use the file system to store long term information. \
            You may also write output for the user to the file system. \
            You also have an addtional think tool that you can use to reflect on your work and plan next steps.",
            tools=[write_to_file_system, read_from_file_system, think_tool],
        )

        file_system = ToolDescription(
            description="Agent to interact with the file system of host machine. Should be used to store information that needs to persist for further use or context.",
            tool_func=file_system_agent,
        )

        ask_user_agent = Agent(
            self.model,
            instructions="You ask the user clarifying questions when you need more information to complete a task. Once you have the information you need, you provide it back to the supervisor agent as a summary for it to then reason over. Do not return a question in that summary since the user will not see it. When done, set the status to REVIEW. If you runinto errors set the status to ERROR",
            tools=[ask_user, think_tool],
            output_type=TaskResult,
        )

        ask_user_tool = ToolDescription(
            description="Tool to ask the user a question and get input back from them.",
            tool_func=ask_user_agent,
        )

        thinking_tool_agent = Agent(
            self.model,
            instructions="You can use this tool to reflect on your progress and plan your next steps carefully.",
            tools=[think_tool],
        )

        thinking_tool = ToolDescription(
            description="Tool for strategic reflection on progress and decision-making.",
            tool_func=thinking_tool_agent,
        )

        _tools = [synthesizer, researcher, file_system, thinking_tool]

        # if additional tools ahve been supplied then add those to the tool list
        if tools:
            _tools.extend(tools)

        _tool_registry = {str(uuid.uuid4()): tool for tool in _tools}

        # each agent gets its own unique id
        return _tool_registry

    def _create_supervisor(self, tools=None, model="openai:gpt-4.1-mini") -> Agent:
        spec = SupervisorSpec()
        agent = Agent(
            model=model,
            deps_type=RuntimeState,
            tools=tools,
            output_type=SupervisorDecision,
        )

        @agent.system_prompt
        def _prompt(ctx):
            return spec.system_prompt(ctx)

        return agent

    def _initialize_runtime_state(
        self, plan: Dict[str, TaskItem], goal: str, registry: dict
    ) -> RuntimeState:
        # Logic to initialize and manage the runtime state
        return RuntimeState(plan=plan, goal=goal, agent_registry=registry)

    async def run(self):
        # Start the supervisor agent to manage sub-agents
        # state = RuntimeState(goal=self.prompt)
        planner_prompt = f"""
        Goal: {self.prompt}

        Capabilities: {self.agent_registry}
        """
        agent_plan = await self._planner_agent.run(planner_prompt)
        agent_plan_map = {v.id: v for v in agent_plan.output.tasks}
        print(f"--- Initial Plan Created with {len(agent_plan_map)} tasks ---")
        pprint(agent_plan_map)

        # now save the plan to the agent state
        runtime_state = self._initialize_runtime_state(
            plan=agent_plan_map, goal=self.prompt, registry=self.agent_registry
        )

        # setup supervisor whose job is to determine if work is done or if there are more things to do
        supervisor_agent = self._create_supervisor(tools=[self.update_task_status])
        task_queue = []
        step_count = 0
        while step_count < self._max_steps:

            print(f"\n--- DeepAgent Cycle {step_count} ---")
            # Setup up this iterations supervisor instruction
            supervisor_response = await supervisor_agent.run(
                "Execute the plan given the current runtime state and status of tasks.",
                deps=runtime_state,
            )

            print(f"--- Awaiting Task Results ---")
            # execute tasks that are ready to run and await responses
            task_results = await self.execute_ready_tasks(
                supervisor_response, runtime_state
            )
            print(f"--- Task Results ---")
            # go through responses and evaluate if they have completed the task
            for task_result in task_results or []:
                print(f"--- Evaluating Task Result for {task_result.id} ---")
                qa_prompt = f"""
                Goal: {runtime_state.goal}
                Task Description: {task_result.description}
                Worker Output: {task_result.result}

                Based on the goal and task description, evaluate if the worker output sufficiently completes the task.
                Provide a detailed analysis and TRUE or FALSE if it passed or not quality check.

                You have the ability to reflect on the work done and provide feedback for improvement if needed.
                1. If the work is sufficient, respond with TRUE.
                2. If the work is insufficient, respond with FALSE and provide detailed feedback on what needs to be improved.
                3. Consider any additional information we may need to complete the task successfully.

                You have the ability to read the detailed report file if it was generated by the worker. Use that to inform your decision.
                """
                qa_response = await self._critic_agent.run(qa_prompt)
                print(f"--- QA Response ---")
                pprint(qa_response.output.model_dump())

                runtime_state.plan[task_result.id].iteration_history.append(
                    qa_response.output
                )

            # output qa report of task completion for supervisor
            # add any new knowledge to the runtime state knowledge store
            runtime_state.runtime_steps += 1
            # supervisor gets qa report and decides if work is done and if so marks it as done and decides what to do next.

            step_count += 1
            return supervisor_response

    # def run(self):
    #     # init the runtime state

    #     # planner agent creates the initial work plan
    #     agent_plan = self._planner_agent.run_sync(self.prompt).output
    #     agent_plan_map = {v.id: v for v in agent_plan.tasks}

    #     run_state = self._initialize_runtime_state(
    #         agent_plan_map, self.prompt, self.agent_registry
    #     )
    #     # runner looks at the plan and executes tasks that have no dependencies or all dependencies are taken care of
    #     run_results = self.execute_ready_tasks()
    #     # after agent runs call eval agent to determine if the task was compeleted successfully

    #     # if not it needs to decide to either replan

    # async def review_worker_output(
    #     self, ctx: RunContext[RuntimeState], task_id: str, worker_output: str
    # ):
    #     task = ctx.deps.plan[task_id]
    #     task.attempt_count += 1

    #     # Call the separate Critic Agent
    #     critic_report = await critic_agent.run(
    #         f"Goal: {task.description}\nResult: {worker_output}"
    #     )

    #     if "APPROVED" in critic_report.data.upper():
    #         ctx.deps.knowledge_store[task_id] = worker_output
    #         task.status = "completed"
    #         return f"Task {task_id} approved after {task.attempt_count} attempts."

    #     if task.attempt_count >= task.max_attempts:
    #         task.status = "failed"
    #         return f"Task {task_id} failed after maximum revision attempts."

    #     # Otherwise, set up for revision
    #     task.status = "requires_revision"
    #     task.review_feedback = critic_report.data
    #     return f"Revision requested for {task_id}. Feedback: {critic_report.data}"

    async def update_knowledge(
        self, capabiliity, answer, ctx: RunContext[RuntimeState]
    ):
        """Updates the knowledge runtime state with any new knowledge that is needed to answer a goal or task"""
        pass

    async def execute_ready_tasks(
        self, tasks: SupervisorDecision, ctx: RuntimeState
    ) -> Union[None, list]:
        """Finds all tasks that are ready to run and executes them in parallel."""

        # 1. Identify "Ready" tasks
        ready_steps = [
            ctx.plan[id]
            for id in tasks.tasks_to_execute
            # if step.status == TaskStatus.PENDING
            # and all(
            #     tasks_to_execute.get(d_id).status == "completed"
            #     for d_id in step.task_dependencies
            # )
        ]

        print("Ready Steps to Execute:")
        print(len(ready_steps))
        if not ready_steps:
            return None

        # 2. Prepare the concurrent coroutines
        ready_tasks = []
        for step in ready_steps:
            print(f"- {step.id}: {step.description} using {step.capability}")
            print(f"  Dependencies: {step.task_dependencies}")
            print(f"  Status: {step.status}")
            print(f"  Result: {step.result}")
            print()
            # grab the tool that the plan or supervisor decides
            worker = self.agent_registry.get(step.capability)
            if worker:
                step.status = TaskStatus.RUNNING
                # We wrap the agent run in a small wrapper to update the step status after
                ready_tasks.append(self.execute(worker.tool_func, step))

        # 3. Execute tasks and return exceptions to notify the supervisor
        print("--- Executing Ready Tasks ---")
        task_results = []
        async with TaskGroup() as tg:
            for task in ready_tasks:
                task_results.append(tg.create_task(task))

        results = [t.result() for t in task_results]
        print("--- All Ready Tasks Completed ---")
        print(len(results))
        pprint(results)
        return results

    async def execute(self, tool, step):
        """Helper to run an agent and capture its output into the step object."""
        try:
            result = await tool.run(step.description)
            step.result = result.output
            # Update the step status to REVIEW
            step.status = TaskStatus.REVIEW
            pprint(step.model_dump())
            return step
        except Exception as e:
            step.status = TaskStatus.ERROR
            step.error_msg = str(e)
            pprint(step.model_dump())
            return step

    async def reflect_on_work(self, reflection, ctx: RunContext[RuntimeState]):
        "Tool to reflect on if work has been completed."
        return f"{reflection}"

    async def update_task_status(
        self, ctx: RunContext[RuntimeState], step_id: str, status: TaskStatus
    ):
        """The supervisor uses this for updating a specific step in the plan to a new status."""
        if step_id in ctx.deps.plan:
            ctx.deps.plan.get(step_id).status = status
            return f"Status for {step_id} is now {status}."
        return f"Error: No step with {step_id} found in plan. Be sure status_id actually exists."

    async def call_worker(
        self, ctx: RunContext[RuntimeState], capability: str, instruction: str
    ):
        """The Supervisor calls this to hand off a task to one of the sub agent workers."""
        # Find the agent in the registry
        worker_agent = ctx.deps.agent_registry.get(capability, None)

        if not worker_agent:
            return f"Error: No specialist found for '{capability}'."
        # print(f"AGENT: {capability} INSTRUCTION: {instruction}")
        # Trigger the deep reasoning loop of the sub-agent
        result = await worker_agent.run(instruction)
        # Return just the result to the Supervisor
        return result
