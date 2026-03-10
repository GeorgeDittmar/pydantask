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
from typing import List, Optional, Literal, Any, Dict, Callable, Union, Type
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from datetime import datetime
import asyncio
from asyncio import TaskGroup
from pydantic_ai import RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport
from pydantic_ai.common_tools.tavily import tavily_search_tool
from loguru import logger
from pydantask.agents.spec import (
    BaseAgentSpec,
    SupervisorSpec,
    ProducerSpec,
    ResearcherSpec,
    SynthesizerSpec,
)
from pydantic_ai.models import Model
from pydantask.prompts.prompts import (
    PLANNER_SYS_PROMPT,
    CRITIC_SYS_PROMPT,
    PRODUCER_SYS_PROMPT,
    RESEARCH_AGENT_SYS_PROMPT,
    SUPERVISOR_SYS_PROMPT,
    SUPERVISOR_INPUT_PROMPT,
    WORKER_AGENT_SYS_PROMPT,
)
from pydantask.models import (
    RuntimeState,
    TaskItem,
    Plan,
    TaskQAResult,
    TaskStatus,
    SupervisorDecision,
    CapabilityDescription,
    TaskResult,
)

from pydantask.tools.default_tools import (
    write_to_file_system,
    read_from_file_system,
    think_tool,
    get_current_datetime,
    list_completed_tasks,
    list_documents,
    get_task_result,
    save_task_context,
    read_task_context,
)

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


from pprint import pprint


class DeepAgent:
    """Pydantic AI based DeepAgent that manages sub-agents to achieve complex goals."""

    def __init__(
        self,
        prompt: str,
        model: str | Model = "gpt-4.1-mini",
        critic_agent: Optional[Agent] = None,
        planner_agent: Optional[Agent] = None,
        supervisor_agent: Optional[Agent] = None,
        researcher_agent: Optional[Agent] = None,
        max_steps: int = 20,
        set_token_budget: Union[int, None] = None,
        sub_agents: Union[None, list[CapabilityDescription]] = None,
        human_feedback: bool = False,
        trace: bool = False,
        # default output type for the producer agent, can be set to a default type or custom pydantic model for better structure and validation of final output
        output_type: Type = TaskResult,
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
        # load_dotenv()

        if trace:
            langfuse = get_client()
            logger.info("Enabling Langfuse tracing...")
            # Verify connection
            if langfuse.auth_check():
                logger.info("Langfuse client is authenticated and ready!")
                Agent.instrument_all()
            else:
                logger.error(
                    "Authentication failed. Could not find TAVILY_API_KEY in environment variables."
                )

        self.model_name: str = model
        self.prompt: str = prompt  # Objective for the agent
        self._max_steps: int = max_steps  # Max steps to prevent infinite loops
        self.token_budget: Union[int, None] = set_token_budget

        self.output_type = output_type
        self._retry_client = self._create_retrying_client()

        # TODO: support other chatmodels and providers beyond openai by allowing custom model and provider classes to be passed in as arguments and used for each agent. For now we will just use the openai chat model with the retry transport for all agents since it is the most robust for long conversations and has built in support for function calling which is useful for tool use.

        # have a ChatModel factory or something so we can support full set of models
        self._retry_model = OpenAIChatModel(
            model, provider=OpenAIProvider(http_client=self._retry_client)
        )
        self._planner_agent = planner_agent or Agent(
            name="_default_Planner_Agent",
            model=self._retry_model,
            system_prompt=PLANNER_SYS_PROMPT,
            output_type=Plan,
            tools=[think_tool],
            end_strategy="exhaustive",
        )

        self._critic_agent = critic_agent or Agent(
            model=self._retry_model,
            name="_default_Critic_Agent",
            system_prompt=CRITIC_SYS_PROMPT,
            output_type=TaskQAResult,
            deps_type=RuntimeState,
            tools=[read_from_file_system, get_current_datetime, think_tool],
            end_strategy="exhaustive",
        )

        self._supervisor_agent = supervisor_agent or self._create_agent_from_spec(
            agent_spec=SupervisorSpec(),
            name="_default_Supervisor_Agent",
            tools=[
                self.update_task_status,
                get_current_datetime,
                think_tool,
                self.view_qa_report,
            ],
            output_type=SupervisorDecision,
            deps_type=RuntimeState,
            model=self._retry_model,
        )

        self._producer_agent = self._create_agent_from_spec(
            agent_spec=ProducerSpec(),
            name="_default_Producer_Agent",
            tools=[
                # Core FS and context tools
                write_to_file_system,
                read_from_file_system,
                save_task_context,
                read_task_context,
                # Reasoning / time / plan-inspection tools
                think_tool,
                get_current_datetime,
                list_documents,
                list_completed_tasks,
                get_task_result,
            ],
            output_type=output_type,
            deps_type=RuntimeState,
            model=self._retry_model,
        )

        # TODO: rework some of these tools
        api_key = os.getenv("TAVILY_API_KEY", None)

        if not api_key:
            raise ValueError(
                "Tavily search api key not found or provided in env variables"
            )

        self._researcher_agent = researcher_agent or Agent(
            model=self._retry_model,
            name="_default_Research_Agent",  # Use a cheap model for simple tasks
            system_prompt=RESEARCH_AGENT_SYS_PROMPT,
            tools=[
                tavily_search_tool(api_key),
                think_tool,
                # File-system and context tools; prefer save_task_context for reports
                write_to_file_system,
                read_from_file_system,
                save_task_context,
                read_task_context,
                get_current_datetime,
                list_documents,
            ],
            deps_type=RuntimeState,
            output_type=TaskResult,
            end_strategy="exhaustive",
        )

        self.agent_registry = self._setup_default_sub_agents(sub_agents=sub_agents)

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

    def _setup_default_sub_agents(
        self, sub_agents: Union[None, list[CapabilityDescription]] = None
    ):
        """
        Setup default sub agents along with any additional sub atgents that may be provided by the caller.
        Default suba gents available are producer, critic, researcher, and file_system.

        Args:
            tools (Union[None, list[ToolDescription]], optional): Any custom tools to include in the agent. Defaults to None.

        Returns:
            dict[str, ToolDescription]: Mapping of toolId's to the tool description and function
        """

        producer_agent = Agent(
            model=self._retry_model,
            name="_default_Producer_agent",
            system_prompt=PRODUCER_SYS_PROMPT,
            deps_type=RuntimeState,
            output_type=TaskResult,
            tools=[
                # FS & context tools for canonical final reports
                write_to_file_system,
                read_from_file_system,
                save_task_context,
                read_task_context,
                # Plan / history inspection
                list_documents,
                list_completed_tasks,
                get_task_result,
                # Reflection
                think_tool,
            ],
        )

        producer = CapabilityDescription(
            name="producer_agent",
            description="Produces output based on information from various sources and sub agents.",
            tool_func=producer_agent,
        )

        researcher = CapabilityDescription(
            name="research_agent",
            description="Tool to research information. This could include searching the web or querying a data source.",
            tool_func=self._researcher_agent,
        )

        general_worker_agent = Agent(
            model=self._retry_model,
            name="_default_General_Worker_Agent",
            system_prompt=WORKER_AGENT_SYS_PROMPT,
            deps_type=RuntimeState,
            output_type=TaskResult,
            tools=[
                write_to_file_system,
                read_from_file_system,
                save_task_context,
                read_task_context,
                list_documents,
                list_completed_tasks,
                get_task_result,
                think_tool,
                get_current_datetime,
            ],
        )

        worker = CapabilityDescription(
            name="worker_agent",
            description=(
                "General-purpose worker for analysis, summarization, document editing, "
                "code or log interpretation, and other non-web tasks that operate on "
                "existing context and files."
            ),
            tool_func=general_worker_agent,
        )

        # file_system_agent = Agent(
        #     model=self._retry_model,
        #     name="_default_File_System_Agent",
        #     system_prompt="You have access to a file system to use for tasks that need to be completed. \
        #     Use the file system to store long term information. \
        #     You may also write output for the user to the file system. \
        #     You also have an addtional think tool that you can use to reflect on your work and plan next steps.",
        #     tools=[write_to_file_system, read_from_file_system, think_tool],
        #     deps_type=RuntimeState,
        #     output_type=TaskItem,
        # )

        # file_system = CapabilityDescription(
        #     name="file_system_agent",
        #     description="Agent to interact with the file system of host machine. Should be used to store information that needs to persist for further use or context.",
        #     tool_func=file_system_agent,
        # )

        # ask_user_agent = Agent(
        #     self.model,
        #     system_prompt="You ask the user clarifying questions when you need more information to complete a task. Once you have the information you need, you provide it back to the supervisor agent as a summary for it to then reason over. Do not return a question in that summary since the user will not see it. When done, set the status to REVIEW. If you runinto errors set the status to ERROR",
        #     tools=[ask_user, think_tool],
        #     output_type=TaskResult,
        # )

        # ask_user_tool = ToolDescription(
        #     name="ask_user_agent",
        #     description="Tool to ask the user a question and get input back from them.",
        #     tool_func=ask_user_agent,
        # )

        _sub_agents_list = [producer, researcher]

        # if additional sub agents been supplied then add those to the registry
        if sub_agents:
            _sub_agents_list.extend(sub_agents)

        _sub_agent_registry = {
            sub_agent.name: sub_agent for sub_agent in _sub_agents_list
        }
        # each agent gets its own unique id
        return _sub_agent_registry

    def _create_agent_from_spec(
        self,
        model: Model,
        agent_spec: BaseAgentSpec,
        name: str = "Agent",
        deps_type: Type[RuntimeState] = RuntimeState,
        output_type=None,
        tools: list[Callable] = [],
        end_strategy="exhaustive",
    ) -> Agent:

        spec = agent_spec
        agent = Agent(
            model=model,
            name=name,
            deps_type=deps_type,
            output_type=output_type,
            tools=tools,
            end_strategy=end_strategy,
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

    def _format_capabilities(self) -> str:
        """Return a human-readable list of available sub-agent capabilities for the planner.

        Each line is of the form:
        - capability_name: description
        """
        lines = []
        for name, desc in self.agent_registry.items():
            description = getattr(desc, "description", "")
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def _build_producer_prompt(self, state: RuntimeState) -> str:
        completed = [t for t in state.plan.values() if t.status == TaskStatus.COMPLETED]
        lines = []
        for t in sorted(completed, key=lambda x: x.task_id):
            result = t.result
            summary = (
                getattr(result, "summary", str(result)) if result else "<no result>"
            )
            paths = getattr(result, "detailed_report_paths", []) if result else []
            sources = getattr(result, "sources", []) if result else []
            lines.append(
                f"- Task {t.task_id} ({t.capability})\n"
                f"  objective: {t.sub_task_objective}\n"
                f"  summary: {summary}\n"
                f"  report_paths: {paths}\n"
                f"  sources: {sources}"
            )
        completed_display = "\n".join(lines) or "<no completed tasks>"

        return f"""
    Overall objective:
    {state.objective}

    Completed sub-tasks (source material):
    {completed_display}

    Using only these results (and any files they point to), synthesize the final answer
    to the overall objective. Do not perform new research.
    """

    def _format_plan(self, plan: Plan):
        lines = []
        for task in plan.tasks:
            id = task.task_id
            sub_task_obj = task.sub_task_objective
            task_status = task.status
            metadata = task.metadata
            lines.append(
                f"- Task ID:{id}\n sub_task_obj: {sub_task_obj} \n task_status: {task_status}\n metadata: {metadata}"
            )
        return "\n".join(lines)

    # def _format_
    def format_input_prompt(self, ctx: RuntimeState) -> str:
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        plan_display_lines = []
        for t in ctx.plan.values():
            line = (
                f"- Task ID: {t.task_id} | Status: [{t.status}] "
                f"| Objective: {t.sub_task_objective} "
                f"| Dependencies: {t.sub_task_dependencies}"
            )

            fb = getattr(t, "task_feedback", None)
            if fb is not None:
                # Adjust these fields to match TaskQAResult
                # verdict = getattr(fb, "passed", None)
                verdict = getattr(fb, "passed", None)
                summary = getattr(fb, "reasoning", None)

                line += "\n  QA: "
                if verdict is not None:
                    line += f"verdict={verdict} "
                if summary:
                    line += f"\n    summary: {summary}"

            plan_display_lines.append(line)

        plan_display = "\n".join(plan_display_lines)
        # Simplify the registry so the Supervisor sees "Tools" not "Agent Objects"
        agent_display = "\n".join(
            [
                f"- {uuid}: {info.description}"
                for uuid, info in ctx.agent_registry.items()
            ]
        )
        return SUPERVISOR_INPUT_PROMPT.format(
            objective=ctx.objective,
            plan_display=plan_display,
            agent_display=agent_display,
        )

    @observe
    async def run(self):
        # Start the supervisor agent to manage sub-agents
        # state = RuntimeState(goal=self.prompt)
        now = await get_current_datetime()
        current_year = datetime.now().year
        capabilities_display = self._format_capabilities()
        planner_prompt = f"""
        Objective: {self.prompt}

        AVAILABLE CAPABILITIES:
        {capabilities_display}
        
        Example of what capabilities could be used for:
            -   "research_agent" → needs web/external info.
            -   "worker_agent" → general reasoning/transformation on existing info.
            -   "producer_agent" → only final answer step.
        
        
        Current Datetime (MUST be used verbatim if time is needed as context): {now}
        CURRENT_YEAR (authoritative numeric year): {current_year}
        
        Come up with a plan for the above objective using the available capabilities.
        Always include the above datetime in the plan metadata and any date-sensitive instructions.
        Use CURRENT_YEAR exactly as provided when resolving any relative time expressions.
        """

        agent_plan = await self._planner_agent.run(planner_prompt)

        agent_plan_map = {v.task_id: v for v in agent_plan.output.tasks}

        logger.info("--- Generated Plan ---")
        logger.info(self._format_plan(agent_plan.output))
        logger.info("--- Generated Plan ---")
        # now save the plan to the agent state
        runtime_state = self._initialize_runtime_state(
            plan=agent_plan_map, objective=self.prompt, registry=self.agent_registry
        )

        step_count = 0
        stop_execution = False
        while step_count < self._max_steps and not stop_execution:

            logger.info(f"\n--- DeepAgent Cycle {step_count} ---")

            supervisor_response = await self._supervisor_agent.run(
                self.format_input_prompt(runtime_state),
                deps=runtime_state,
            )
            supervisor_response = supervisor_response.output

            if supervisor_response.all_tasks_completed:
                logger.info(
                    f"--- All tasks completed according to supervisor. Ending execution loop. ---"
                )
                stop_execution = True
                continue

            logger.info(f"--- Executing Tasks ---")
            # execute tasks that are ready to run and await responses
            task_results = await self._execute_ready_tasks(
                supervisor_response, runtime_state
            )

            if len(task_results) == 0:
                # handle case if task_results are empty
                logger.info("NO TASK RESULTS")
                stop_execution = True
                continue

            logger.info("--- Task Results ---")
            logger.info(f"Number of tasks executed: {len(task_results)}")
            logger.info(f"Results: {task_results}")
            logger.info("--- Task Result ---")
            # go through responses and evaluate if they have completed the task
            for task_result in task_results or []:
                logger.info(f"--- Evaluating Task Result for {task_result.task_id} ---")
                qa_prompt = f"""
                
                Evaluate if the following worker output completed its task.
                
                Overall Objective:
                {runtime_state.objective}

                Sub Task Definition (TaskItem):
                {runtime_state.plan[task_result.task_id].model_dump_json(indent=2)}

                Worker Output (TaskResult):
                {task_result.result.model_dump_json(indent=2)}
                
                Any documents to review:
                {runtime_state.document_store}
                """
                qa_response = await self._critic_agent.run(
                    qa_prompt, deps=runtime_state
                )
                qa_response = qa_response.output

                logger.info(f"--- QA Response ---")
                logger.info(qa_response.model_dump_json())
                # if qa_response.do
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

    def _dependencies_satisfied(self, step: TaskItem, ctx: RuntimeState) -> bool:
        # Consider a dependency satisfied only if it's COMPLETED (or whatever set you like)
        required_statuses = {TaskStatus.COMPLETED}
        for dep_id in step.sub_task_dependencies or []:
            dep_task = ctx.plan.get(dep_id)
            if dep_task is None:
                # Be conservative: if the dependency is missing, treat it as unsatisfied
                return False
            if dep_task.status not in required_statuses:
                return False
        return True

    async def _execute_ready_tasks(
        self, tasks: SupervisorDecision, ctx: RuntimeState
    ) -> list[TaskItem]:
        """Finds all tasks that are ready to run and executes them in parallel."""
        candidate_steps = [ctx.plan[id] for id in tasks.tasks_to_execute]

        ready_steps = [
            step for step in candidate_steps if self._dependencies_satisfied(step, ctx)
        ]

        # if no ready steps return empty list
        if len(ready_steps) == 0:
            return []

        logger.info("Ready Steps to Execute:")
        logger.info(ready_steps)

        if not ready_steps:
            return []

        # 2. Prepare the concurrent coroutines
        ready_tasks = []
        for step in ready_steps:
            # get supervisor feedback if any for this task
            if (
                tasks.feedback_to_subagents
                and step.task_id in tasks.feedback_to_subagents
            ):
                if step.parameters is None:
                    # create if None
                    step.parameters = {}
                step.parameters["supervisor_feedback"] = (
                    tasks.feedback_to_subagents.get(step.task_id)
                )

            logger.info(
                f"- {step.task_id}: {step.sub_task_objective} using {step.capability}"
            )
            logger.info(f"  Dependencies: {step.sub_task_dependencies}")
            logger.info(f"  Status: {step.status}")
            logger.info(f"  Result: {step.result}")
            logger.info("\n")

            # grab the tool that the plan or supervisor  decides
            worker = self.agent_registry.get(step.capability)
            if worker:
                step.status = TaskStatus.RUNNING
                # We wrap the agent run in a small wrapper to update the step status after
                ready_tasks.append(self.execute(worker.tool_func, step, ctx))

        # 3. Execute tasks and return exceptions to notify the supervisor
        logger.info("--- Executing Ready Tasks ---")
        task_results = []
        async with TaskGroup() as tg:
            for task in ready_tasks:
                task_results.append(tg.create_task(task))

        results = [t.result() for t in task_results]
        logger.info("--- All Ready Tasks Completed ---")
        logger.info(results)
        return results

    @retry(wait=wait_exponential_jitter(), reraise=True, stop=stop_after_attempt(3))
    async def execute(
        self, sub_agent, step: TaskItem, runtime_state: RuntimeState
    ) -> TaskItem:
        """Helper to run an agent and capture its output into the step object."""

        _feedback_for_agent = None
        if isinstance(step.parameters, dict):
            _feedback_for_agent = step.parameters.get("supervisor_feedback")

        if step.capability == "producer_agent":
            # Build a synthesis-oriented prompt that summarizes all completed tasks,
            # including their summaries, report_paths, and sources.
            producer_context = self._build_producer_prompt(runtime_state)

            user_prompt = f"""
                        {producer_context}

                        You are now executing the FINAL synthesis TaskItem:

                        {step.model_dump_json(indent=2)}
                        """

            if _feedback_for_agent:

                user_prompt += f"""

                    Supervisor feedback / additional insturctions for this execution:
                    
                    {_feedback_for_agent}
                    """
            user_prompt += """
                    Your job:
                    - Use ONLY the completed sub-task results and any files they point to.
                    - Combine their findings into a single, coherent final answer.
                    - Follow your system prompt instructions for citations and final TaskResult structure.
                    - Do NOT request new research or create new sub-tasks.
                    """
        else:
            user_prompt = f"""
                You are executing TaskItem:

                {step.model_dump_json(indent=2)}

                Overall objective:
                {self.prompt}

                """
            if _feedback_for_agent:
                user_prompt += f"""

                Supervisor feedback for this execution:
                {_feedback_for_agent}
                """

            user_prompt += """

            ONLY act on this sub-task. Do not re-plan or change the task.
            """
        # try:
        result = await sub_agent.run(
            user_prompt,
            deps=runtime_state,
        )
        step.result = result.output
        step.status = TaskStatus.NEEDS_REVIEW
        return step
        # except Exception as e:
        #     step.status = TaskStatus.ERRORED
        #     step.error_msg = str(e)
        #     return step

    async def update_task_status(
        self, ctx: RunContext[RuntimeState], task_id: int, status: TaskStatus
    ):
        """The supervisor uses this for updating a specific task in the plan to a new status.

        When to use:
            - If a task has no dependencies and needs to be set to 'READY' state
            - If a task has a result which has been QA'd and passed.

        When not to use:
            - Setting a task to a status to complete the plan before it is actually done.
        """
        if task_id in ctx.deps.plan:
            ctx.deps.plan.get(task_id).status = status
            return f"Status for {task_id} is now {status}."
        return f"Error: No task with {task_id} found in plan. Be sure task_id actually exists."

    async def view_qa_report(self, ctx: RunContext[RuntimeState], task_id: int) -> str:
        """
        Tool Name: View QA Report
        Desription: Tool to view the specific detailed QA report for a given task_id.
        When to use:
            - If you need to review the full QA report from the critic.
        """
        task = ctx.deps.plan.get(task_id)
        if task is None:
            return f"No task with id {task_id}."

        fb = getattr(task, "task_feedback", None)
        if fb is None:
            return f"No QA feedback found for task {task_id}."

        # Return either a summary or full JSON depending on your needs
        return fb.model_dump_json(indent=2)
