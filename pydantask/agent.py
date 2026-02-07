# from asyncio import tasks
from email import errors
from json import tool
import json
from multiprocessing.connection import wait
import uuid
import os

from httpx import AsyncClient, HTTPStatusError
from langfuse import get_client
from langfuse import observe
from tenacity import (
    wait_exponential_jitter,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)
from loguru import logger
from os import system

from enum import Enum
from pydantic_ai import Agent, RunContext, FunctionToolset
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict, Callable, Union
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

import asyncio
from asyncio import TaskGroup
from pydantic_ai import RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport
from pydantic_ai.common_tools.tavily import tavily_search_tool
from .prompts import PLANNER_SYS_PROMPT, CRITIC_SYS_PROMPT
from .models import (
    ResearchResult,
    RuntimeState,
    TaskItem,
    Plan,
    TaskQAResult,
    TaskStatus,
    SupervisorDecision,
    ToolDescription,
    TaskResult,
)
from .default_tools import (
    write_to_file_system,
    read_from_file_system,
    think_tool,
    ask_user,
    get_current_datetime,
)

from pydantic_ai.common_tools.tavily import (
    tavily_search_tool,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


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


class SupervisorSpec:
    def system_prompt(self, ctx: RunContext[RuntimeState]) -> str:
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        plan_display = "\n".join(
            [
                f"- [{t.status}] ID: {t.task_id} | Task: {t.task_objective} | Deps: {t.task_dependencies}"
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

        return f"""
### ROLE
You are the Orchestrator/Supervisor. Your job is to manage the execution of a multi-step plan by delegating tasks to specialized sub-agents.

### MISSION OBJECTIVE
{ctx.deps.objective}

### CURRENT MISSION CONTROL BOARD
<plan_status>
{plan_display}
</plan_status>

### AVAILABLE SUB-AGENT CAPABILITIES
<capabilities>
{agent_display}
</capabilities>

### OPERATING PROCEDURES
1. **Dependency Check:** Only move tasks to 'READY' if all their `task_dependencies` are marked 'COMPLETED'.
2. **Parallel Execution:** You MAY delegate multiple independent 'READY' tasks simultaneously.
3. **Quality Assurance (QA):**
   - If a task is in 'REVIEW', check the `TaskQAReport` and verify if the task meets what is needed towards completing the objective. 
   - If QA passed: Mark task as 'COMPLETED'.
   - If QA failed: Mark task back to 'READY' and include the QA feedback in the task instructions.
4. **Error Handling:** If a task is 'FAILED', investigate the error and decide if the task needs to be reran or if the plan needs an update via your tools.
5. **Self-Reflection:** Use the `think_tool` before every decision to verify you aren't missing a dependency or misallocating a sub-agent.

### OUTPUT INSTRUCTIONS
Decide which tasks to execute now. Return your decision as a `SupervisorDecision` object.
"""


synth_agent_sys_prompt = """
Your sole task is to take all the information that has been gathered from research tasks and synthesize it into a coherent answer to the original goal.
You may not ask for more information. You must answer with the information you have. 
If you are confused you may use the think tool to reflect on your work and plan next steps or raise any issues you have with the supervisor
in the return object with a task status of ERROR. You cannot ask the user for more information.

Think through your task 

You have access to the following tools:
    - write_to_file_system: Use this to write long form answers to the file system for later retrieval. 
    - read_from_file_system: Use this to read any files that may have been previously saved for context by previous tasks or to read any long term memory you have kept.
    - think_tool: Use this to reflect on your work and plan next steps.
    
When you generate your answer, you must provide both a detailed report and a summary.
    The detailed report should be long form and include references to any sources used.
    The summary should be concise and to the point.

When you write the detailed report to the file system, return the file path in the detailed_report_path field of your output.
"""


class DeepAgent:
    """Pydantask AI DeepAgent that manages sub-agents to achieve complex goals."""

    def __init__(
        self,
        prompt,
        model="gpt-4.1-mini",
        critic_model="gpt-4.1-mini",
        max_steps=20,
        set_token_budget: int = None,
        tools: Union[None, list[ToolDescription]] = None,
        human_feedback: bool = False,
    ):
        """
        Create DeepAgent instance.

        :param prompt: The overall objective for the agent.
        :param model: The language model to use. default is "gpt-4.1-mini".
        :param max_steps: Maximum steps to prevent infinite loops. defaults to 20 steps
        :param set_token_budget: Token budget for the agent's operation. Defaults to None (no limit).
        :param tools: List of ToolDescription objects representing the agent's capabilities. Defaults to None.
        :param human_feedback: Whether to incorporate human feedback in the agent's decision-making. Defaults to False.
        """
        self.model = model
        self.prompt = prompt  # Objective for the agent
        self._max_steps = max_steps  # Max steps to prevent infinite loops
        self.token_budget = set_token_budget
        self.agent_registry = self._setup_default_tools(tools=tools)
        self.critic_model = critic_model

        client = self._create_retrying_client()
        _model = OpenAIChatModel(model, provider=OpenAIProvider(http_client=client))
        self._planner_agent = Agent(
            name="Planner Agent",
            model=_model,
            system_prompt=PLANNER_SYS_PROMPT,
            output_type=Plan,
            tools=[get_current_datetime, think_tool],
            end_strategy="exhaustive",
        )

        self._critic_agent = Agent(
            model=_model,
            name="Critic Agent",
            system_prompt="You are an expert critic. Evaluate the following output from work done on a task. Output a detailed report and if it meets the task requirements.",
            output_type=TaskQAResult,
            deps_type=RuntimeState,
            tools=[read_from_file_system, get_current_datetime, think_tool],
            end_strategy="exhaustive",
        )

    def _create_retrying_client(self):
        """Create a client with smart retry handling for multiple error types.
        https://ai.pydantic.dev/retries/
        """

        def should_retry_status(response):
            """Raise exceptions for retryable HTTP status codes."""
            if response.status_code in (429, 502, 503, 504):
                response.raise_for_status()  # This will raise HTTPStatusError

        transport = AsyncTenacityTransport(
            config=RetryConfig(
                # Retry on HTTP errors and connection issues
                retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
                # Smart waiting: respects Retry-After headers, falls back to exponential backoff
                wait=wait_retry_after(
                    fallback_strategy=wait_exponential(multiplier=1, max=60),
                    max_wait=300,
                ),
                # Stop after 5 attempts
                stop=stop_after_attempt(5),
                # Re-raise the last exception if all retries fail
                reraise=True,
            ),
            validate_response=should_retry_status,
        )

        return AsyncClient(transport=transport)

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
            name="Synthesizer Agent",
            system_prompt=synth_agent_sys_prompt,
            output_type=str,
            tools=[
                write_to_file_system,
                read_from_file_system,
                think_tool,
            ],
        )

        synthesizer = ToolDescription(
            description="Generate answers based on information from various sources and sub agents.",
            tool_func=synthesizer_agent,
        )

        # A "Thin" Agent that just wraps a tool
        researcher_agent = Agent(
            self.model,
            name="Research Agent",  # Use a cheap model for simple tasks
            system_prompt="You are a research specialist. When given a task, follow these steps: "
            "1. Generate 5 search queries based on the task description. Start with broad queries and narrow down."
            "2. Call the search tool for each query. "
            "3. Analyze the results from the search tool and extract detailed information."
            "4. Use the think tool to reflect on the information gathered and determine if another search is needed."
            "5. If enough information has been gathered, summarize the findings into a clear report."
            " Be concise and focus on the most relevant information to the task."
            " When building the search report, generate a detailed report and a summary."
            "When genrating the detailed report, include references to sources used with each section written. Write the detailed report to a markdown file and return the file path in the detailed_report_path field of the output.",
            tools=[
                tavily_search_tool(api_key),
                think_tool,
                write_to_file_system,
                read_from_file_system,
                get_current_datetime,
            ],
            deps_type=RuntimeState,
            output_type=TaskResult,
            end_strategy="exhaustive",
        )

        researcher = ToolDescription(
            description="Tool to research information. This could include searching the web or querying a data source.",
            tool_func=researcher_agent,
        )

        file_system_agent = Agent(
            self.model,
            system_prompt="You have access to a file system to use for tasks that need to be completed. \
            Use the file system to store long term information. \
            You may also write output for the user to the file system. \
            You also have an addtional think tool that you can use to reflect on your work and plan next steps.",
            tools=[write_to_file_system, read_from_file_system, think_tool],
            deps_type=RuntimeState,
        )

        file_system = ToolDescription(
            description="Agent to interact with the file system of host machine. Should be used to store information that needs to persist for further use or context.",
            tool_func=file_system_agent,
        )

        ask_user_agent = Agent(
            self.model,
            system_prompt="You ask the user clarifying questions when you need more information to complete a task. Once you have the information you need, you provide it back to the supervisor agent as a summary for it to then reason over. Do not return a question in that summary since the user will not see it. When done, set the status to REVIEW. If you runinto errors set the status to ERROR",
            tools=[ask_user, think_tool],
            output_type=TaskResult,
        )

        ask_user_tool = ToolDescription(
            description="Tool to ask the user a question and get input back from them.",
            tool_func=ask_user_agent,
        )

        thinking_tool_agent = Agent(
            self.model,
            system_prompt="You are a strategic reflection agent. Your job is to think deeply about the work that has been done so far and provide insights and next steps. Use this tool to reflect on progress, identify gaps, and plan future actions. Your reflection should be thorough and consider all aspects of the task at hand.",
            tools=[think_tool],
        )

        thinking_tool = ToolDescription(
            description="Tool for strategic reflection on progress and decision-making. Must be used after each task to reflect on work done and plan next steps.",
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
            name="Supervisor",
            deps_type=RuntimeState,
            tools=tools,
            output_type=SupervisorDecision,
            end_strategy="exhaustive",
        )

        @agent.system_prompt
        def _prompt(ctx):
            return spec.system_prompt(ctx)

        return agent

    def _initialize_runtime_state(
        self, plan: Dict[int, TaskItem], objective: str, registry: dict
    ) -> RuntimeState:
        # Logic to initialize and manage the runtime state
        return RuntimeState(plan=plan, objective=objective, agent_registry=registry)

    @observe
    async def run(self):
        # Start the supervisor agent to manage sub-agents
        # state = RuntimeState(goal=self.prompt)
        planner_prompt = f"""
        Goal: {self.prompt}

        Capabilities: {self.agent_registry}
        
        Come up with a plan for the above goal using the available capabilities.
        """

        agent_plan = await self._planner_agent.run(planner_prompt)

        agent_plan_map = {v.task_id: v for v in agent_plan.output.tasks}
        print("--- Generated Plan ---")
        # return
        # now save the plan to the agent state
        runtime_state = self._initialize_runtime_state(
            plan=agent_plan_map, objective=self.prompt, registry=self.agent_registry
        )

        # setup supervisor whose job is to determine if work is done or if there are more things to do
        supervisor_agent = self._create_supervisor(
            tools=[self.update_task_status, get_current_datetime, think_tool],
            model=self.model,
        )
        step_count = 0
        stop_execution = False
        while step_count < self._max_steps and not stop_execution:

            print(f"\n--- DeepAgent Cycle {step_count} ---")

            supervisor_response = await supervisor_agent.run(
                "Execute the plan given the current runtime plan state and status of tasks. Be sure to check if any tasks are ready for review for final acceptance or if a task is needing to be reran.",
                deps=runtime_state,
            )
            supervisor_response = supervisor_response.output

            if supervisor_response.all_tasks_completed:
                print(
                    f"--- All tasks completed according to supervisor. Ending execution loop. ---"
                )
                stop_execution = True
                break

            print(f"--- Awaiting Task Results ---")
            # execute tasks that are ready to run and await responses
            task_results = await self.execute_ready_tasks(
                supervisor_response, runtime_state
            )

            print(f"--- Task Results ---")
            # go through responses and evaluate if they have completed the task
            for task_result in task_results or []:
                print(f"--- Evaluating Task Result for {task_result.task_id} ---")
                qa_prompt = f"""
                Overall Objective: {runtime_state.objective}

                Sub Task Result: {task_result.model_dump_json(indent=2)}

                Based on the objetive and task description, evaluate if the worker output sufficiently completes the task.
                Provide a detailed analysis and TRUE or FALSE if it passed or not quality check.

                Abilities:
                You have the ability to reflect on the work done and provide feedback for improvement if needed.
                You have the ability to read the detailed report file if it was generated by the worker. Use that to inform your decision.

                Instructions:
                1. If the work is sufficient, respond with TRUE.
                2. If the work is insufficient, respond with FALSE and provide detailed feedback on what needs to be improved.
                3. Do not attempt to qa the whole GOAL, just the specific task assigned. The GOAL is meant to provide context only.
                4. Use the think tool to reflect on the work done and plan your evaluation carefully.

                Do not make assumptions. Base your evaluation strictly on the worker output and the task description.
                """
                qa_response = await self._critic_agent.run(
                    qa_prompt, deps=runtime_state
                )
                qa_response = qa_response.output
                print(f"--- QA Response ---")
                pprint(qa_response.model_dump_json())

                # add the qa report to the task result for the supervisor to review
                runtime_state.plan[task_result.task_id].task_feedback = qa_response

            runtime_state.runtime_steps += 1

            step_count += 1
        return runtime_state

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
        print(ready_steps)

        if not ready_steps:
            return None

        # 2. Prepare the concurrent coroutines
        ready_tasks = []
        for step in ready_steps:
            print(f"- {step.task_id}: {step.task_objective} using {step.capability}")
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

    @retry(wait=wait_exponential_jitter(), reraise=True, stop=stop_after_attempt(3))
    async def execute(self, tool, step: TaskItem) -> TaskItem:
        """Helper to run an agent and capture its output into the step object."""
        await asyncio.sleep(1.0)  # 1 second sleep to not hit throttling limits
        try:
            result = await tool.run(
                f"""Execute the following task: {step.task_objective}. Make sure to keep in mind the overall objective: {self.prompt}.
                                    Do not act on the goal act only on the singular task description provided."""
            )
            step.result = result.output
            step.status = TaskStatus.NEEDS_REVIEW
            return step
        except Exception as e:
            step.status = TaskStatus.ERRORED
            step.error_msg = str(e)
            return step

    async def update_task_status(
        self, ctx: RunContext[RuntimeState], task_id: int, status: TaskStatus
    ):
        """The supervisor uses this for updating a specific task in the plan to a new status."""
        if task_id in ctx.deps.plan:
            ctx.deps.plan.get(task_id).status = status
            return f"Status for {task_id} is now {status}."
        return f"Error: No task with {task_id} found in plan. Be sure task_id actually exists."

    async def call_worker(
        self, ctx: RunContext[RuntimeState], capability: str, instruction: str
    ):
        """The Supervisor calls this to hand off a task to one of the sub agent workers."""
        # Find the agent in the registry
        worker_agent = ctx.deps.agent_registry.get(capability, None)

        if not worker_agent:
            return f"Error: No specialist found for '{capability}'."
        # Trigger the deep reasoning loop of the sub-agent
        result = await worker_agent.run(instruction)
        return result
