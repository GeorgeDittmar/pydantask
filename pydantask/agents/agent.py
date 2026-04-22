# from asyncio import tasks
from email import errors
from json import tool
import json
from multiprocessing.connection import wait
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

import uuid

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
from pydantic_ai.usage import RunUsage, UsageLimits
from loguru import logger
from pydantask.agents.spec import (
    BaseAgentSpec,
    SupervisorSpec,
    ProducerSpec,
    ResearcherSpec,
    SynthesizerSpec,
)
from pydantask.agents import utils
from pathlib import Path
from pydantic_ai.models import Model
from pydantask.prompts.prompts import (
    PLANNER_SYS_PROMPT,
    CRITIC_SYS_PROMPT,
    PRODUCER_SYS_PROMPT,
    RESEARCH_AGENT_SYS_PROMPT,
    SUPERVISOR_INPUT_PROMPT,
    WORKER_AGENT_SYS_PROMPT,
    DYNAMIC_SUPERVISOR_SYS_PROMPT,
)
from pydantask.observe.tracing import _langfuse_instrumented, _langsmith_instrumented,_logfire_instrumented
from pydantask.models import (
    RuntimeState,
    TaskItem,
    Plan,
    TaskQAResult,
    TaskStatus,
    SupervisorDecision,
    CapabilityDescription,
    TaskResult,
    DeepAgentRunResult,
    TracingBackend
)

from pydantask.tools.default_tools import (
    read_scratch_notes,
    write_to_file_system,
    read_from_file_system,
    think_tool,
    get_current_datetime,
    list_completed_tasks,
    list_documents,
    get_task_result,
    save_task_context,
    read_task_context,
    append_scratch_note,
)

from pydantask.observe.tracing import traced, init_tracing_backend, autodetect_tracing_backend, flush_tracing
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


from pprint import pprint


class DeepAgent:
    """Pydantic AI based DeepAgent that manages sub-agents to achieve complex goals."""

    def __init__(
        self,
        prompt: str,
        model: str | Model = "gpt-4.1-mini",
        # thinking: bool = False,
        critic_agent: Optional[Agent] = None,
        planner_agent: Optional[Agent] = None,
        supervisor_agent: Optional[Agent] = None,
        researcher_agent: Optional[Agent] = None,
        max_steps: int = 20,
        set_token_budget: Union[int, None] = None,
        sub_agents: Union[None, list[CapabilityDescription]] = None,
        human_in_loop: bool = False,
        # default output type for the producer agent, can be set to a default type or custom pydantic model for better structure and validation of final output
        output_type: Type = TaskResult,
        # planning_mode: str = "dynamic",  # "static" | "dynamic"
        trace: bool = False,
        checkpoint: bool = False
    ):
        """
        Create DeepAgent instance.

        :param prompt: The overall objective for the agent.
        :param model: The language model to use. default is "gpt-4.1-mini".
        :param max_steps: Maximum steps to prevent infinite loops. defaults to 20 steps
        :param set_token_budget: Token budget for the agent's operation. Defaults to None (no limit).
        :param tools: List of ToolDescription objects representing the agent's capabilities. Defaults to None.
        :param human_in_loop: Whether to incorporate human feedback in the agent's decision-making. Defaults to False.
        """
        if trace:
            init_tracing_backend(autodetect_tracing_backend())

        self.model_name: str = model
        self.prompt: str = prompt  # Objective for the agent
        self._max_steps: int = max_steps  # Max steps to prevent infinite loops
        self.token_budget: Union[int, None] = set_token_budget

        self.output_type = output_type
        self._retry_client = self._create_retrying_client()

        self.checkpoint = checkpoint
        self.checkpoint_path = Path(f"_checkpoint/{uuid.uuid4()}/") 
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
            # end_strategy="exhaustive",
        )

        self._supervisor_agent = supervisor_agent or Agent(
            model=self._retry_model,
            name="_dynamic_Supervisor_Agent",
            system_prompt=DYNAMIC_SUPERVISOR_SYS_PROMPT,
            output_type=SupervisorDecision,
            deps_type=RuntimeState,
            tools=[
                self.update_task_status,
                self.add_task,
                self.cancel_task,
                self.patch_task,
                get_current_datetime,
                think_tool,
                self.view_qa_report,
            ],
            end_strategy="exhaustive",
        )

        self._producer_agent = self._create_agent_from_spec(
            agent_spec=ProducerSpec(),
            name="_default_Producer_Agent",
            tools=[
                # Reasoning / time / plan-inspection tools
                think_tool,
                get_current_datetime,
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
                append_scratch_note,
                read_scratch_notes,
                get_current_datetime,
            ],
            deps_type=RuntimeState,
            output_type=TaskResult,
        )

        self.agent_registry = self._setup_default_sub_agents(
            additonal_capabilities=sub_agents
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

    def _setup_default_sub_agents(
        self, additonal_capabilities: Union[None, list[CapabilityDescription]] = None
    ) -> Dict:
        """
        Setup default sub agents along with any additional sub atgents that may be provided by the caller.
        Default suba gents available are producer, critic, researcher, and file_system.

        Args:
            sub_agents (Union[None, list[CapabilityDescription]], optional): Any custom tools / agents to include. Defaults to None.

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
                read_task_context,
                # Plan / history inspection
                # list_documents,
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
                # list_documents,
                list_completed_tasks,
                get_task_result,
                think_tool,
                append_scratch_note,
                read_scratch_notes,
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

        _sub_agents_list = [producer, researcher]

        # if additional sub agents been supplied then add those to the registry
        if additonal_capabilities:
            _sub_agents_list.extend(additonal_capabilities)

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

    def _initialize_runtime_state(self, objective: str, registry: dict) -> RuntimeState:
        # Logic to initialize and manage the runtime state
        return RuntimeState(
            objective=objective, agent_registry=registry, next_task_id=1
        )
    
    def _checkpoint_state(self, runtime: RuntimeState):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_checkpoint = utils.get_incremented_path(f"state_{timestamp}", "json", directory=self.checkpoint_path)
        new_checkpoint.write_text(runtime.model_dump_json(indent=4), encoding="utf-8")

    def _format_capabilities(self) -> str:
        """
        Return a human-readable list of available sub-agent capabilities for the planner.

        Each line is of the form:
        - capability_name: description
        """
        lines = []
        for name, desc in self.agent_registry.items():
            description = getattr(desc, "description", "")
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

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

    def _format_supervisor_input_prompt(self, ctx: RuntimeState) -> str:
        """Function to format input prompt to the supervisor agent."""
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        capability_display = self._format_capabilities()

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

        return SUPERVISOR_INPUT_PROMPT.format(
            objective=ctx.objective,
            plan_display=plan_display,
            agent_display=capability_display,
            now=datetime.now(),
            current_year=datetime.now().year,
        )

    def _format_critic_input_prompt(self, task_result: TaskResult, ctx: RuntimeState):
        _prompt = f"""
            
            Evaluate if the following worker output completed the specified task task.
            
            Overall Objective:
            {ctx.objective}

            Sub Task Definition (TaskItem):
            {ctx.plan[task_result.task_id].model_dump_json(indent=2)}

            Worker Output (TaskResult):
            {task_result.result.model_dump_json(indent=2)}
            
            Any documents to review:
            {ctx.document_store}
            
            """
        return _prompt

    async def add_task(
        self,
        ctx: RunContext[RuntimeState],
        sub_task_objective: str,
        capability: str,
        dependencies: list[int] | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Tool: Add Task
        Description: Add a new TaskItem to the current DAG if more work is needed to complete the objective.

        Must add any `sub_task_dependencies` that may be needed to complete the new task.
        """
        plan = ctx.deps.plan
        new_id = ctx.deps.next_task_id
        ctx.deps.next_task_id += 1

        task = TaskItem(
            task_id=new_id,
            overall_objective=ctx.deps.objective,
            sub_task_objective=sub_task_objective,
            capability=capability,
            sub_task_dependencies=dependencies or [],
            metadata=metadata or {},
            status=TaskStatus.READY,
        )
        plan[new_id] = task
        return new_id

    async def cancel_task(
        self, ctx: RunContext[RuntimeState], task_id: int, reason: str
    ):
        """
        Tool: Cancel Task
        Description: Use this to remove a task from the plan if it is no longer
        relevant or if a failure in an upstream dependency makes it impossible.
        """
        if task_id in ctx.deps.plan:
            # Instead of deleting, mark as CANCELLED to keep history
            ctx.deps.plan[task_id].status = TaskStatus.CANCELLED
            return f"Task {task_id} cancelled. Reason: {reason}"
        return f"Error: Task {task_id} not found."

    async def patch_task(
        self,
        ctx: RunContext[RuntimeState],
        task_id: int,
        sub_task_objective: Optional[str] = None,
        dependencies: Optional[List[int]] = None,
    ):
        """Update an existing task's objective or its dependency requirements."""
        task = ctx.deps.plan.get(task_id)
        if not task:
            return "Task not found."

        if sub_task_objective:
            task.sub_task_objective = sub_task_objective
        if dependencies is not None:
            task.sub_task_dependencies = dependencies
        return f"Task {task_id} updated successfully."

    @traced()
    async def run(self) -> DeepAgentRunResult:

        runtime_state = self._initialize_runtime_state(
            objective=self.prompt, registry=self.agent_registry
        )
        
        step_count = 0
        stop_execution = False
        while step_count < self._max_steps and not stop_execution:

            logger.info(f"\n--- DeepAgent Cycle {step_count} ---")

            supervisor_response = await self._supervisor_agent.run(
                self._format_supervisor_input_prompt(runtime_state),
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

                qa_response = await self._critic_agent.run(
                    self._format_critic_input_prompt(task_result, runtime_state),
                    deps=runtime_state,
                )
                qa_response = qa_response.output

                logger.info(f"--- QA Response ---")
                logger.info(qa_response.model_dump_json())

                task = runtime_state.plan[task_result.task_id]

                # deterministic transition based on critic
                self.handle_critic_result(task, qa_response)

                # If the task result was produced via tools that return JSON (like write_to_file_system),
                # extract the written files to populate the task's output_paths.
                if hasattr(task_result, "result") and isinstance(
                    task_result.result, str
                ):
                    try:
                        import json

                        tool_output = json.loads(task_result.result)
                        if (
                            isinstance(tool_output, dict)
                            and "written_files" in tool_output
                        ):
                            if task.result:
                                task.result.output_paths.extend(
                                    tool_output["written_files"]
                                )
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                self._checkpoint_state(runtime_state)

                # if qa_response.do
                # add the qa report to the task result for the supervisor to review
                # runtime_state.plan[task_result.task_id].task_feedback = qa_response
            runtime_state.runtime_steps += 1

            # make sure traces flush after each loop
            step_count += 1
            
        return_result = DeepAgentRunResult(
            objective=self.prompt,
            final_result=task.result if "task" in locals() else None,
            plan=runtime_state.plan,
            runtime_state=runtime_state,
        )

        return return_result

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

    @traced(run_type="tool")
    @retry(wait=wait_exponential_jitter(), reraise=True, stop=stop_after_attempt(3))
    async def execute(
        self, sub_agent: Agent, step: TaskItem, runtime_state: RuntimeState
    ) -> TaskItem:
        """Helper to run an agent and capture its output into the step object."""

        _feedback_for_agent = None
        if isinstance(step.parameters, dict):
            _feedback_for_agent = step.parameters.get("supervisor_feedback")

        if step.capability == "producer_agent":
            # Build a synthesis-oriented prompt that summarizes all completed tasks,
            # including their summaries, report_paths, and sources.
            # producer_context = self._build_producer_prompt(runtime_state)

            user_prompt = f"""
            Overall objective:
            {self.prompt}

            You are the final synthesis agent.
            - First, call `list_completed_tasks` to see all completed upstream tasks.
            - For each task that is relevant to the objective (especially research tasks), call `get_task_result(task_id=...)`.
            - THEN, write a single, coherent comparative analysis answering the objective.
            - You MUST explicitly integrate evidence from ALL relevant completed tasks (e.g. Task 1 and Task 2 in this run).
            """

            if _feedback_for_agent:

                user_prompt += f"""

                    Supervisor feedback / additional instructions for this execution:
                    
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

            ONLY act on this sub-task and any feedback. Do not re-plan or change the task.
            """
        # try:
        result = await sub_agent.run(
            user_prompt,
            deps=runtime_state,
            usage_limits=UsageLimits(tool_calls_limit=20),
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

    def handle_critic_result(self, task: TaskItem, review: TaskQAResult):
        """Deterministic transition logic to be used after critic run."""
        if review.passed:
            task.status = TaskStatus.COMPLETED
        else:
            if task.attempt_count < task.max_attempts:
                # THE TRANSITION
                task.status = TaskStatus.READY  # Or a specific RERUN status
                task.attempt_count += 1

                # THE CONTEXT INJECTION (Crucial!)
                # We must force the agent to see the failure so it doesn't repeat it
                task.sub_task_objective += f"\n\n[RETRY {task.attempt_count}] Previous attempt failed review: {review.reasoning}"
            else:
                task.status = TaskStatus.FAILED
                task.error_msg = (
                    f"Max retries reached. Critic feedback: {review.reasoning}"
                )

    async def view_qa_report(self, ctx: RunContext[RuntimeState], task_id: int) -> str:
        """
        Tool Name: View QA Report
        Desription: Tool to view the specific detailed QA report for a given task_id.
        When to use:
            - If you need to review the full QA report from the critic.
        """
        task = ctx.deps.plan.get(task_id)
        logger.info(ctx.deps.plan)
        if task is None:
            return f"No task with id {task_id}."

        fb = getattr(task, "task_feedback", None)
        if fb is None:
            return f"No QA feedback found for task {task_id}."

        # Return either a summary or full JSON depending on your needs
        return fb.model_dump_json(indent=2)
