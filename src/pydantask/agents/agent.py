# from asyncio import tasks
from email import errors
from json import tool
import json
from multiprocessing.connection import wait
import os
import asyncio
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
from collections import Counter

from loguru import logger
from os import system

from enum import Enum
from pydantic_ai import Agent, RunContext
from typing import List, Optional, Literal, Any, Dict, Callable, Union, Type
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from datetime import datetime
import asyncio
from asyncio import TaskGroup
from pydantic_ai import RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.retries import AsyncTenacityTransport
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.usage import UsageLimits
from loguru import logger
from pydantask.agents.spec import (
    BaseAgentSpec,
)
from pydantask.agents import utils
from pathlib import Path
from pydantic_ai.models import Model
from pydantask.prompts.prompts import (
    CRITIC_SYS_PROMPT,
    PRODUCER_SYS_PROMPT,
    RESEARCH_AGENT_SYS_PROMPT,
    SUPERVISOR_INPUT_PROMPT,
    WORKER_AGENT_SYS_PROMPT,
    DYNAMIC_SUPERVISOR_SYS_PROMPT,
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
    DeepAgentRunResult,
    TaskRunDeps,
)

# Default tool wiring is intentionally in-memory focused.
# Filesystem tools still exist in `pydantask.tools.default_tools` but are not enabled by default.
from pydantask.tools.default_tools import (
    append_scratch_note,
    get_current_datetime,
    get_task_result,
    list_completed_tasks,
    read_scratch_notes,
    think_tool,
)

from pydantask.observe.tracing import (
    traced,
    init_tracing_backend,
    autodetect_tracing_backend,
    flush_tracing,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


class DeepAgent:
    """Pydantic AI based DeepAgent that manages sub-agents to achieve complex goals."""

    def __init__(
        self,
        objective: str,
        model: str | Model = "gpt-5.2",
        seed_plan: Plan | None = None,
        planning_mode: Literal["llm", "fixed", "hybrid"] = "llm",
        critic_agent: Optional[Agent] = None,
        supervisor_agent: Optional[Agent] = None,
        researcher_agent: Optional[Agent] = None,
        max_steps: int = 20,
        set_token_budget: Union[int, None] = None,
        sub_agents: Union[None, list[CapabilityDescription]] = None,
        # default output type for the producer agent, can be set to a default type or custom pydantic model for better structure and validation of final output
        output_type: Type = TaskResult,
        # planning_mode: str = "dynamic",  # "static" | "dynamic"
        trace: bool = False,
        checkpoint: bool = False,
        verbose_logging: bool = False,
    ):
        """Initialize a DeepAgent instance.

        Args:
            objective: The overall objective / task the deep agent is working on.
            model: Model identifier or ``pydantic_ai.models.Model`` instance to use
                for all sub-agents. Defaults to ``"gpt-5.2"``.
            seed_plan: Optional pre-defined :class:`~pydantask.models.Plan` to seed
                the initial task DAG. If provided, it is loaded into
                :class:`~pydantask.models.RuntimeState.plan` at the start of
                :meth:`run`.

                Notes:
                * Task IDs are respected and used as keys in ``RuntimeState.plan``.
                * ``RuntimeState.next_task_id`` is set to ``max(task_id) + 1``.
                * Dependencies are validated to ensure they reference existing tasks.
            planning_mode: Controls whether the supervisor is allowed to modify the
                plan at runtime.

                * ``"llm"``: The supervisor may add/patch tasks.
                * ``"hybrid"``: Same as ``"llm"``, but typically used with
                  ``seed_plan`` to provide an initial DAG the supervisor can extend.
                * ``"fixed"``: The supervisor is not given the plan-mutation tools
                  (``add_task``/``patch_task``) and can only execute/transition the
                  existing tasks.
            critic_agent: Optional pre-configured critic ``Agent``. If omitted, a
                default critic agent is created.
            supervisor_agent: Optional supervisor ``Agent`` used to manage the task
                DAG. If omitted, a default dynamic supervisor is created.
            researcher_agent: Optional research ``Agent``. If omitted, a default
                web/doc research agent is created.
            producer_agent: Optional producer ``Agent``. If omitted, a default
                agent is created.
            max_steps: Maximum number of DeepAgent control-loop iterations to run
                before forcing termination.
            set_token_budget: Optional global token budget for the run. Currently
                stored but not strictly enforced.
            sub_agents: Additional ``CapabilityDescription`` objects to register as
                callable sub-agents alongside the built-ins.
            output_type: Pydantic model type used as the default output structure
                for the producer agent.
            trace: If ``True``, auto-configure tracing via the configured backend.
            checkpoint: If ``True``, persist runtime state snapshots to disk after
                each supervisor/critic cycle.
            verbose_logging: If ``True``, log richer debugging information during
                execution.
        """

        if trace:
            init_tracing_backend(autodetect_tracing_backend())

        # `model` can be either:
        #   - a pydantic_ai Model instance (fully custom)
        #   - a bare model name (defaults to OpenAI), e.g. "gpt-4.1-mini"
        #   - a provider-prefixed string, e.g. "openai:gpt-4.1-mini" or "anthropic:claude-sonnet-4-5"
        self.model_name: str = (
            model if isinstance(model, str) else model.__class__.__name__
        )

        if objective is None:
            raise TypeError("DeepAgent requires 'objective' to be provided")

        if planning_mode in {"fixed", "hybrid"} and seed_plan is None:
            raise ValueError(
                "seed_plan must be provided when planning_mode is 'fixed' or 'hybrid'"
            )

        self.objective: str = objective
        self._max_steps: int = max_steps  # Max steps to prevent infinite loops
        self.token_budget: Union[int, None] = set_token_budget
        self.verbose = verbose_logging
        self.output_type = output_type
        self.planning_mode = planning_mode
        self.seed_plan: Union[Plan, None] = seed_plan
        self._retry_client = self._create_retrying_client()

        self.checkpoint = checkpoint

        # Concurrency guardrails:
        # - `_plan_lock` protects plan-level mutations and task claiming (READY->RUNNING).
        #   This prevents accidentally scheduling the same task twice in one cycle.
        self._plan_lock = asyncio.Lock()

        self.checkpoint_path: Path | None = None
        if checkpoint:
            self.checkpoint_path = Path(f"_checkpoint/{uuid.uuid4()}/")
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Build the shared model used by all sub-agents.
        # We inject the retrying httpx client into the provider for durability.
        self._retry_model = self._build_model(model)

        # NOTE: Filesystem tools exist in `pydantask.tools.default_tools`, but are not
        # enabled by default. The harness is currently in-memory focused.
        self._critic_agent = critic_agent or Agent(
            model=self._retry_model,
            name="_default_Critic_Agent",
            system_prompt=CRITIC_SYS_PROMPT,
            output_type=TaskQAResult,
            deps_type=RuntimeState,
            tools=[get_current_datetime, think_tool],
            # end_strategy="exhaustive",
        )

        self._supervisor_agent = supervisor_agent or Agent(
            model=self._retry_model,
            name="_dynamic_Supervisor_Agent",
            system_prompt=DYNAMIC_SUPERVISOR_SYS_PROMPT,
            output_type=SupervisorDecision,
            deps_type=RuntimeState,
            tools=self._supervisor_tools(),
            end_strategy="exhaustive",
        )

        # self._producer_agent = Agent(
        #     model=self._retry_model,
        #     name="_default_Producer_agent",
        #     system_prompt=PRODUCER_SYS_PROMPT,
        #     deps_type=TaskRunDeps,
        #     output_type=TaskResult,
        #     tools=[
        #         # Plan / history inspection
        #         list_completed_tasks,
        #         get_task_result,
        #         # Reflection
        #         think_tool,
        #     ],
        # )

        # TODO: rework some of these tools
        tavily_api_key = os.getenv("TAVILY_API_KEY", None)

        research_tool_set = [
            think_tool,
            append_scratch_note,
            read_scratch_notes,
            get_current_datetime,
        ]

        if not tavily_api_key:
            logger.info(
                "Tavily api key not found. Defaulting to built in Duck Duck Go search tool."
            )
            research_tool_set.append(duckduckgo_search_tool())
        else:
            research_tool_set.append(tavily_search_tool(tavily_api_key))

        self._researcher_agent = researcher_agent or Agent(
            model=self._retry_model,
            name="_default_Research_Agent",  # Use a cheap model for simple tasks
            system_prompt=RESEARCH_AGENT_SYS_PROMPT,
            tools=research_tool_set,
            deps_type=TaskRunDeps,
            output_type=TaskResult,
        )

        self.agent_registry = self._setup_default_sub_agents(
            additonal_capabilities=sub_agents
        )

        # Scheduler/system notes injected into the next supervisor prompt.
        self._last_scheduler_report: str = ""

    async def aclose(self) -> None:
        """Close underlying resources used by this ``DeepAgent`` instance.

        This is primarily responsible for flushing any tracing backends and
        closing the shared async HTTP client used by the model providers.
        Safe to call multiple times.
        """
        try:
            # best-effort; safe to call even if tracing is disabled
            flush_tracing()
        finally:
            if getattr(self, "_retry_client", None) is not None:
                await self._retry_client.aclose()

    async def __aenter__(self) -> "DeepAgent":
        """Enter the async context manager and return this ``DeepAgent``.

        Allows ``async with DeepAgent(...) as agent: ...`` usage.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the async context manager, ensuring resources are cleaned up."""
        await self.aclose()

    def _supervisor_tools(self) -> list[Callable]:
        """Return the set of tools exposed to the supervisor agent.

        The selected toolset depends on :attr:`planning_mode`:

        * ``fixed``: supervisor can update/cancel tasks and view QA reports, but
          cannot add or patch tasks.
        * ``llm`` / ``hybrid``: supervisor can also add and patch tasks.

        Note: This is enforced by tool registration (not just prompting), so in
        ``fixed`` mode the supervisor LLM cannot call ``add_task``/``patch_task``.
        """
        base_tools = [
            self.update_task_status,
            self.cancel_task,
            self.view_qa_report,
            get_current_datetime,
            think_tool,
        ]

        mutating_tools = [
            self.add_task,
            self.patch_task,
        ]

        if self.planning_mode == "fixed":
            return base_tools

        return base_tools + mutating_tools

    def _build_model(self, model: str | Model) -> Model:
        """Construct a ``pydantic_ai`` model, wiring in the shared HTTP client.

        The ``model`` parameter may be either:

        * A bare model name (e.g. ``"gpt-4.1-mini"``) which defaults to the
          OpenAI provider.
        * A provider-prefixed string such as ``"openai:gpt-4.1-mini"`` or
          ``"anthropic:claude-sonnet-4-5"``.
        * An already-instantiated ``pydantic_ai.models.Model`` instance, which is
          returned unchanged.
        """
        if isinstance(model, Model):
            return model

        provider_name: str
        model_name: str
        if ":" in model:
            provider_name, model_name = model.split(":", 1)
            provider_name = provider_name.strip().lower()
            model_name = model_name.strip()
        else:
            provider_name, model_name = "openai", model

        if provider_name in {"openai", "openai_compat", "openrouter"}:
            # NOTE: "openrouter" here assumes OpenAI-compatible API. If you want true
            # OpenRouter defaults (headers/routing), we may want OpenRouterProvider. Dunno
            return OpenAIChatModel(
                model_name, provider=OpenAIProvider(http_client=self._retry_client)
            )

        if provider_name == "anthropic":
            return AnthropicModel(
                model_name,
                provider=AnthropicProvider(http_client=self._retry_client),
            )

        raise ValueError(
            f"Unsupported model provider prefix: {provider_name!r}. "
            "Use e.g. 'openai:...' or 'anthropic:...' or pass a Model instance."
        )

    def _create_retrying_client(self):
        """Create an ``httpx.AsyncClient`` with robust retry behaviour.

        The returned client uses ``AsyncTenacityTransport`` with sensible
        defaults for rate limits and transient network failures. See
        https://ai.pydantic.dev/retries/ for more details.
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
        """Create the default sub-agent capability registry.

        This wires up the built-in producer, researcher, and general worker
        agents, and optionally merges any extra ``CapabilityDescription``
        instances supplied by the caller.

        Args:
            additonal_capabilities: Additional capabilities to register on top of
                the built-in sub-agents.

        Returns:
            Dict[str, CapabilityDescription]: Mapping from capability name to
            its description and callable agent/tool.
        """

        producer_agent = Agent(
            model=self._retry_model,
            name="_default_Producer_agent",
            system_prompt=PRODUCER_SYS_PROMPT,
            deps_type=TaskRunDeps,
            output_type=TaskResult,
            tools=[
                # Plan / history inspection
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
            deps_type=TaskRunDeps,
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

        gen_worker = CapabilityDescription(
            name="worker_agent",
            description=(
                "General-purpose worker for analysis, summarization, document editing, "
                "code or log interpretation, and other non-research tasks that operate on "
                "existing context."
            ),
            tool_func=general_worker_agent,
        )

        _sub_agents_list = [producer, researcher, gen_worker]

        # if additional sub agents been supplied then add those to the registry
        if additonal_capabilities:
            _sub_agents_list.extend(additonal_capabilities)

        _sub_agent_registry = {
            sub_agent.name: sub_agent for sub_agent in _sub_agents_list
        }
        # each agent gets its own unique id
        return _sub_agent_registry

    # def _create_agent_from_spec(
    #     self,
    #     model: Model,
    #     agent_spec: BaseAgentSpec,
    #     name: str = "Agent",
    #     deps_type: Type[RuntimeState] = RuntimeState,
    #     output_type=None,
    #     tools: list[Callable] | None = None,
    #     end_strategy="exhaustive",
    # ) -> Agent:
    #     """Instantiate an ``Agent`` and bind its system prompt from a spec.

    #     The provided ``BaseAgentSpec`` is used to dynamically generate the
    #     system prompt at runtime via ``spec.system_prompt(ctx)``.
    #     """
    #     spec = agent_spec
    #     agent = Agent(
    #         model=model,
    #         name=name,
    #         deps_type=deps_type,
    #         output_type=output_type,
    #         tools=tools or [],
    #         end_strategy=end_strategy,
    #     )

    #     @agent.system_prompt
    #     def _prompt(ctx):
    #         return spec.system_prompt(ctx)

    #     return agent

    def _initialize_runtime_state(self, objective: str, registry: dict) -> RuntimeState:
        """Create the initial :class:`RuntimeState` for a new DeepAgent run.

        This initializes an empty plan. If ``seed_plan`` was provided when the
        DeepAgent was constructed, it is applied at the start of :meth:`run`.

        Args:
            objective: Top-level objective for this DeepAgent execution.
            registry: Mapping of capability names to ``CapabilityDescription``
                instances.

        Returns:
            A freshly initialized ``RuntimeState`` with an empty plan and
            ``next_task_id`` set to ``1``.
        """
        return RuntimeState(
            objective=objective, agent_registry=registry, next_task_id=1
        )

    def _checkpoint_state(self, runtime: RuntimeState):
        """Persist the current runtime state to a JSON checkpoint on disk.

        This is a no-op unless ``checkpoint=True`` was passed at construction.

        The checkpoint directory is unique per DeepAgent instance and is only
        created when checkpointing is enabled on construction.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.checkpoint_path is None:
            return

        new_checkpoint = utils.get_incremented_path(
            f"state_{timestamp}", "json", directory=self.checkpoint_path
        )
        new_checkpoint.write_text(runtime.model_dump_json(indent=4), encoding="utf-8")

    def _format_capabilities(self) -> str:
        """Format all registered capabilities into a planner-friendly string.

        Each line is of the form: ``- <capability_name>: <description>``.
        """
        lines = []
        for name, desc in self.agent_registry.items():
            description = getattr(desc, "description", "")
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def _format_plan(self, plan: Plan):
        """Format a :class:`Plan` instance into a human-readable multi-line string."""
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
        """Build the composite prompt passed to the supervisor agent.

        The prompt includes the overall objective, a summarized status board of
        all tasks in the plan, and a list of available capabilities.
        """
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        capability_display = self._format_capabilities()

        plan_display_lines = []
        for t in ctx.plan.values():
            line = (
                f"- Task ID: {t.task_id} | Status: [{t.status.value}] "
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

        prompt = SUPERVISOR_INPUT_PROMPT.format(
            objective=ctx.objective,
            plan_display=plan_display,
            agent_display=capability_display,
            now=datetime.now(),
            current_year=datetime.now().year,
        )

        if self._last_scheduler_report:
            prompt += (
                "\n\n### SYSTEM SCHEDULER NOTES (deterministic)\n"
                + self._last_scheduler_report.strip()
            )

        return prompt

    def _format_critic_input_prompt(self, task: TaskItem, ctx: RuntimeState) -> str:
        """Construct the evaluation prompt sent to the critic agent.

        The critic receives the overall objective, the ``TaskItem`` definition
        it should be evaluating, the worker's structured ``TaskResult`` (if any),
        and any relevant in-memory documents from the runtime state.

        Note: this harness is currently in-memory focused; do not assume any
        filesystem persistence.
        """
        worker_output = task.result.model_dump_json(indent=2) if task.result else "null"

        _prompt = f"""
            
            Evaluate if the following worker output completed the specified task.

            Overall Objective:
            {ctx.objective}

            Sub Task Definition (TaskItem):
            {task.model_dump_json(indent=2)}

            Worker Output (TaskResult):
            {worker_output}

            In-memory documents / scratchpads:
            {ctx.document_store}
            
            """
        return _prompt

    def _is_context_limit_error(self, exc: Exception) -> bool:
        """Heuristic detection of "context length exceeded" errors.

        Different providers/local gateways surface these differently (OpenAI-style
        400s, Anthropic "prompt too long", llama.cpp "context overflow", etc.).
        """
        msg = str(exc).lower()
        needles = [
            "context length",
            "maximum context",
            "max context",
            "prompt is too long",
            "too many tokens",
            "context overflow",
            "exceeds the context",
            "token limit",
        ]
        if any(n in msg for n in needles):
            return True

        if isinstance(exc, HTTPStatusError):
            # Common for OpenAI-compatible APIs.
            try:
                data = exc.response.json()
            except Exception:
                data = None

            if exc.response.status_code in (400, 413):
                # 413 can happen on some proxies when payload is too large.
                if data and isinstance(data, dict):
                    err = data.get("error") or {}
                    code = (err.get("code") or "").lower()
                    emsg = (err.get("message") or "").lower()
                    if "context" in code or "context" in emsg:
                        return True

        return False

    def _build_resume_prompt(self, step: TaskItem, error: Exception) -> str:
        """Build a minimal resume prompt after a context overflow.

        We intentionally keep this short; the sub-agent should reconstruct its
        progress using task metadata (scratch notes / checkpoints).
        """
        checkpoint = step.metadata.get("scratch_notes", "")
        checkpoint_preview = checkpoint
        if len(checkpoint_preview) > 6_000:
            checkpoint_preview = checkpoint_preview[:6_000] + "\n...[checkpoint truncated]..."

        return f"""
A previous attempt to execute this task failed due to context/window limits.

Task:
- task_id: {step.task_id}
- capability: {step.capability}
- sub_task_objective: {step.sub_task_objective}

Overall objective:
{self.objective}

Checkpoint / scratch notes saved so far (authoritative):
{checkpoint_preview if checkpoint_preview else '<none>'}

Recovery instructions (IMPORTANT):
- Continue the task from the checkpoint above.
- Keep responses concise. Avoid pasting large blobs.
- If you need prior task outputs, call `get_task_result(task_id=..., max_chars=6000)` (or smaller).
- After each major step, call `append_scratch_note(task_id={step.task_id}, note=...)` with a short checkpoint:
  "what I did" + "what I will do next" + "open questions".
- If you feel you're approaching the context limit again, STOP calling tools and output the best possible `TaskResult`.

Error that triggered recovery (for debugging only):
{str(error)}
"""

    async def add_task(
        self,
        ctx: RunContext[RuntimeState],
        sub_task_objective: str,
        capability: str,
        dependencies: list[int] | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Tool: Add Task.

        Note: In ``planning_mode="fixed"`` this tool is not registered on the
        supervisor agent, but it may still be called directly in Python.

        Create and register a new ``TaskItem`` in the current plan/DAG when
        more work is required to achieve the overall objective.

        The supervisor should specify any upstream dependencies so that
        execution order can be enforced.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            sub_task_objective: Natural-language objective for the new task.
            capability: Name of the capability / sub-agent that should execute
                this task.
            dependencies: Optional list of task IDs that must complete
                successfully before this task can run.
            metadata: Optional free-form metadata dictionary attached to the task.

        Returns:
            The integer ``task_id`` assigned to the newly created task.
        """
        async with self._plan_lock:
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
        """Tool: Cancel Task.

        Mark a task as ``CANCELLED`` when it is no longer relevant or when
        a failure in an upstream dependency makes it impossible to complete.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task to cancel.
            reason: Human-readable explanation for the cancellation.
        """
        async with self._plan_lock:
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
        """Tool: Patch Task.

        Note: In ``planning_mode="fixed"`` this tool is not registered on the
        supervisor agent, but it may still be called directly in Python.

        Update an existing task's objective and/or dependency list in-place.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task to modify.
            sub_task_objective: New sub-task objective, if changing.
            dependencies: Updated list of dependency IDs, if changing.
        """
        async with self._plan_lock:
            task = ctx.deps.plan.get(task_id)
            if not task:
                return "Task not found."

            if sub_task_objective:
                task.sub_task_objective = sub_task_objective
            if dependencies is not None:
                task.sub_task_dependencies = dependencies
            return f"Task {task_id} updated successfully."

    def _is_terminal_status(self, status: TaskStatus) -> bool:
        """Return True if a task is in a terminal state.

        Note: ERRORED is intentionally treated as non-terminal; the supervisor
        may still choose to patch the task and rerun it.
        """
        return status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

    async def _scheduler_pass(self, ctx: RuntimeState) -> str:
        """Deterministic scheduler pass.

        This pass performs small, non-LLM state normalization to improve autonomy:

        - Promote PENDING -> READY when all dependencies are COMPLETED.
        - Demote READY -> PENDING if dependencies are not satisfied (keeps the
          status board honest).
        - Mark tasks with unknown capability as ERRORED (unless terminal).

        Returns a human-readable report injected into the next supervisor prompt.
        """
        changes: list[str] = []
        warnings: list[str] = []

        async with self._plan_lock:
            for task_id, task in sorted(ctx.plan.items(), key=lambda kv: kv[0]):
                # Unknown capability detection.
                if task.capability and task.capability not in self.agent_registry:
                    if not self._is_terminal_status(task.status):
                        if task.status != TaskStatus.ERRORED:
                            changes.append(
                                f"- Task {task_id}: {task.status.value} -> errored (unknown capability: {task.capability!r})"
                            )
                        task.status = TaskStatus.ERRORED
                        task.error_msg = f"Unknown capability: {task.capability!r}"
                    continue

                # Dependency-based readiness propagation.
                deps_ok = self._dependencies_satisfied(task, ctx)

                if task.status == TaskStatus.PENDING and deps_ok:
                    task.status = TaskStatus.READY
                    changes.append(f"- Task {task_id}: pending -> ready (deps satisfied)")

                # Keep READY tasks honest if deps are not actually satisfied.
                if task.status == TaskStatus.READY and not deps_ok:
                    task.status = TaskStatus.PENDING
                    changes.append(f"- Task {task_id}: ready -> pending (deps not satisfied)")

        if not changes and not warnings:
            return "No scheduler changes this cycle."

        out: list[str] = []
        if changes:
            out.append("Status normalization:")
            out.extend(changes)
        if warnings:
            out.append("Warnings:")
            out.extend(warnings)
        return "\n".join(out)

    def _build_deadlock_report(
        self, ctx: RuntimeState, decision: SupervisorDecision | None = None
    ) -> str:
        """Explain why no tasks ran in the current cycle."""

        status_counts = Counter(t.status.value for t in ctx.plan.values())
        runnable: list[int] = []
        blocked: list[str] = []

        for task_id, task in sorted(ctx.plan.items(), key=lambda kv: kv[0]):
            if self._is_terminal_status(task.status):
                continue

            if task.status in {TaskStatus.READY, TaskStatus.RERUN} and self._dependencies_satisfied(task, ctx):
                runnable.append(task_id)
                continue

            # Compute a human-readable reason.
            if task.capability not in self.agent_registry:
                blocked.append(
                    f"- Task {task_id} [{task.status.value}]: unknown capability {task.capability!r}"
                )
                continue

            if task.sub_task_dependencies:
                missing = [d for d in task.sub_task_dependencies if d not in ctx.plan]
                if missing:
                    blocked.append(
                        f"- Task {task_id} [{task.status.value}]: missing deps {missing}"
                    )
                    continue

                unmet = [
                    d
                    for d in task.sub_task_dependencies
                    if ctx.plan.get(d) is not None and ctx.plan[d].status != TaskStatus.COMPLETED
                ]
                if unmet:
                    blocked.append(
                        f"- Task {task_id} [{task.status.value}]: waiting on deps {unmet}"
                    )
                    continue

            blocked.append(f"- Task {task_id} [{task.status.value}]: not runnable")

        lines: list[str] = []
        lines.append("Deadlock / no-progress report:")
        lines.append(f"- status_counts: {dict(status_counts)}")
        if decision is not None:
            lines.append(f"- supervisor_requested: {decision.tasks_to_execute or []}")
        lines.append(f"- runnable_now: {runnable}")
        if blocked:
            lines.append("- blocked_examples:")
            # keep this short to avoid prompt bloat
            lines.extend(blocked[:12])

        return "\n".join(lines)

    @traced()
    async def run(self) -> DeepAgentRunResult:
        """Run the full DeepAgent control loop until completion or max steps.

        If a ``seed_plan`` was supplied at construction time, it is loaded into the
        runtime state before the supervisor loop begins.

        This method repeatedly:

        * Invokes the supervisor to decide which tasks to execute next.
        * Executes ready tasks in parallel via their associated sub-agents.
        * Sends results to the critic for QA and status updates.
        * Optionally checkpoints state between iterations.

        Returns:
            A ``DeepAgentRunResult`` containing the final output, the full plan,
            and the final ``RuntimeState``.
        """
        runtime_state = self._initialize_runtime_state(
            objective=self.objective, registry=self.agent_registry
        )
        self._apply_seed_plan(runtime_state)

        errors: list[str] = []
        no_progress_cycles = 0

        step_count = 0
        stop_execution = False
        while step_count < self._max_steps and not stop_execution:

            logger.info(f"\n--- DeepAgent Cycle {step_count} ---")

            # Deterministic scheduler pass to normalize readiness and surface issues.
            self._last_scheduler_report = await self._scheduler_pass(runtime_state)

            supervisor_response = await self._supervisor_agent.run(
                self._format_supervisor_input_prompt(runtime_state),
                deps=runtime_state,
            )
            supervisor_response = supervisor_response.output

            if supervisor_response.all_tasks_completed:
                logger.info("--- Supervisor declared completion. Ending execution loop. ---")
                stop_execution = True
                break

            logger.info("--- Executing Tasks ---")
            # execute tasks that are ready to run and await responses
            task_results = await self._execute_ready_tasks(
                supervisor_response, runtime_state
            )

            # NOTE: `execute(...)` mutates the canonical TaskItem stored in `runtime_state.plan`
            # in-place (it receives the same object reference). Do NOT overwrite
            # `runtime_state.plan[task_id]` with returned TaskItems here; that can clobber
            # concurrent metadata updates (e.g. scratch notes/checkpoints).

            if len(task_results) == 0:
                # No tasks ran this cycle. This is not necessarily terminal in a
                # dynamic planner: we may be blocked on deps, have errored tasks
                # that need patching, or need the supervisor to add new nodes.
                no_progress_cycles += 1
                deadlock = self._build_deadlock_report(runtime_state, supervisor_response)
                self._last_scheduler_report = (self._last_scheduler_report + "\n\n" + deadlock).strip()

                logger.info(
                    f"No executable tasks this cycle (no_progress_cycles={no_progress_cycles}). Continuing."
                )

                # Prevent infinite loops if the supervisor cannot make progress.
                if no_progress_cycles >= 3:
                    msg = (
                        "No progress after 3 consecutive cycles (no tasks executed). "
                        "Stopping to avoid infinite loop. "
                        "See SYSTEM SCHEDULER NOTES in the final cycle for details."
                    )
                    logger.warning(msg)
                    errors.append(msg)
                    stop_execution = True

                runtime_state.runtime_steps += 1
                step_count += 1
                continue

            no_progress_cycles = 0

            logger.info(f"Number of tasks executed: {len(task_results)}")
            # go through responses and evaluate if they have completed the task
            for task_result in task_results or []:
                logger.info(f"--- Evaluating Task Result for {task_result.task_id} ---")

                qa_response = await self._critic_agent.run(
                    self._format_critic_input_prompt(task_result, runtime_state),
                    deps=runtime_state,
                )
                qa_response = qa_response.output
                if self.verbose:
                    logger.info("--- QA Response ---")
                    logger.info(qa_response.model_dump_json(indent=2))

                task = runtime_state.plan[task_result.task_id]

                # deterministic transition based on critic
                self.handle_critic_result(task, qa_response)

                if self.checkpoint:
                    self._checkpoint_state(runtime_state)

            runtime_state.runtime_steps += 1
            step_count += 1

        return_result = DeepAgentRunResult(
            objective=self.objective,
            final_result=task.result if "task" in locals() else None,
            plan=runtime_state.plan,
            runtime_state=runtime_state,
            errors=errors,
        )

        return return_result

    def _apply_seed_plan(self, runtime_state: RuntimeState) -> None:
        """Seed ``runtime_state.plan`` from ``self.seed_plan`` (if provided).

        This is used to support user-specified plans. It validates:

        * Unique task IDs.
        * Dependencies refer to existing tasks.

        It also updates ``runtime_state.next_task_id``.
        """
        if self.seed_plan is None:
            return

        tasks = list(self.seed_plan.tasks or [])
        if not tasks:
            return

        plan_dict: dict[int, TaskItem] = {}
        for t in tasks:
            if t.task_id in plan_dict:
                raise ValueError(f"Duplicate task_id in seed_plan: {t.task_id}")
            # Ensure the overall objective is consistent.
            if not getattr(t, "overall_objective", None):
                t.overall_objective = runtime_state.objective
            plan_dict[t.task_id] = t

        for t in plan_dict.values():
            for dep_id in t.sub_task_dependencies or []:
                if dep_id not in plan_dict:
                    raise ValueError(
                        f"seed_plan task {t.task_id} depends on missing task {dep_id}"
                    )

        runtime_state.plan = plan_dict
        runtime_state.next_task_id = max(plan_dict.keys()) + 1

    def _dependencies_satisfied(self, step: TaskItem, ctx: RuntimeState) -> bool:
        """Return ``True`` if all of a task's dependencies are fully satisfied.

        Currently a dependency is considered satisfied only if the dependent
        task exists and is in the ``COMPLETED`` state.
        """
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

    @traced(capture_input=False)
    async def _execute_ready_tasks(
        self, tasks: SupervisorDecision, ctx: RuntimeState
    ) -> list[TaskItem]:
        """Execute all tasks selected by the supervisor that are ready to run.

        Tasks whose dependencies are satisfied are executed concurrently using
        an ``asyncio.TaskGroup``. The returned list contains the updated
        ``TaskItem`` instances after execution.
        """

        # Dedupe while preserving order (supervisor can occasionally emit duplicates).
        requested_ids: list[int] = list(dict.fromkeys(tasks.tasks_to_execute or []))

        # Be defensive: supervisor might reference missing IDs.
        candidate_steps: list[TaskItem] = [
            ctx.plan[task_id] for task_id in requested_ids if task_id in ctx.plan
        ]

        allowed_statuses = {TaskStatus.READY, TaskStatus.RERUN}

        # Determine which steps are eligible based on status+deps.
        # We'll "claim" them (set RUNNING) under `_plan_lock` below to prevent double-scheduling.
        ready_steps = [
            step
            for step in candidate_steps
            if step.status in allowed_statuses and self._dependencies_satisfied(step, ctx)
        ]
        # if no ready steps return empty list
        if len(ready_steps) == 0:
            return []

        if not ready_steps:
            return []

        # 2. Claim tasks (READY/RERUN -> RUNNING) atomically so we don't schedule the same
        # task twice in parallel.
        claimed_steps: list[TaskItem] = []
        async with self._plan_lock:
            for step in ready_steps:
                # step is a reference to ctx.plan[task_id]
                if step.status not in allowed_statuses:
                    continue
                # deps can change while we awaited the lock (other tasks completing); re-check.
                if not self._dependencies_satisfied(step, ctx):
                    continue
                step.status = TaskStatus.RUNNING
                claimed_steps.append(step)

        if not claimed_steps:
            return []

        # 3. Prepare the concurrent coroutines
        ready_tasks = []
        for step in claimed_steps:
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
                # We wrap the agent run in a small wrapper to update the step status after
                ready_tasks.append(self.execute(worker.tool_func, step, ctx))
            else:
                # No such capability; mark errored so supervisor/QA can see what happened.
                step.status = TaskStatus.ERRORED
                step.error_msg = f"Unknown capability: {step.capability!r}"

        # 4. Execute tasks and return exceptions to notify the supervisor
        logger.info("--- Executing Ready Tasks ---")
        task_results = []
        async with TaskGroup() as tg:
            for task in ready_tasks:
                task_results.append(tg.create_task(task))

        results = [t.result() for t in task_results]
        logger.info("--- All Ready Tasks Completed ---")
        return results

    @traced(run_type="task", capture_input=False)
    @retry(wait=wait_exponential_jitter(), reraise=True, stop=stop_after_attempt(3))
    async def execute(
        self, sub_agent: Agent, step: TaskItem, runtime_state: RuntimeState
    ) -> TaskItem:
        """Execute a sub-agent for a single task and record the result.

        Builds a task-specific prompt (with optional supervisor feedback),
        runs the provided ``sub_agent``, and updates the ``TaskItem`` status
        and result based on success or failure.
        """

        _feedback_for_agent = None
        if isinstance(step.parameters, dict):
            _feedback_for_agent = step.parameters.get("supervisor_feedback")

        if step.capability == "producer_agent":

            user_prompt = f"""
            Overall objective:
            {self.objective}

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
                    - Use ONLY the completed sub-task results from this run.
                    - Combine their findings into a single, coherent final answer.
                    - Follow your system prompt instructions for citations and final TaskResult structure.
                    - Do NOT request new research or create new sub-tasks.
                    """
        else:
            user_prompt = f"""
                You are executing TaskItem:

            {step.model_dump_json(indent=2)}

                Overall objective:
                {self.objective}

                """
            if _feedback_for_agent:
                user_prompt += f"""

                Supervisor feedback / additional instructions for this execution:
                {_feedback_for_agent}
                """

            user_prompt += """

            ONLY act on this sub-task and any feedback. Do not re-plan or change the task.
            """
        task_deps = TaskRunDeps(runtime_state=runtime_state, task=step)

        # Help smaller-context models avoid blowing up in a single long tool-run.
        # This doesn't guarantee safety (tool output can still be large), but combined
        # with truncated tool outputs and scratch checkpoints it greatly improves durability.
        user_prompt += f"""

Context-budget note:
- You may be running on a smaller-context model.
- Prefer small tool outputs. When calling tools that can return large text, request truncation.
- Checkpoint progress frequently via `append_scratch_note(task_id={step.task_id}, note=...)`.
"""

        max_resume_attempts = 2
        last_error: Exception | None = None

        for resume_attempt in range(max_resume_attempts + 1):
            tool_call_limit = 20 if resume_attempt == 0 else 10

            try:
                result = await sub_agent.run(
                    user_prompt,
                    deps=task_deps,
                    usage_limits=UsageLimits(tool_calls_limit=tool_call_limit),
                )
                step.result = result.output
                step.status = TaskStatus.NEEDS_REVIEW
                step.error_msg = None
                return step
            except Exception as e:
                last_error = e
                if self._is_context_limit_error(e) and resume_attempt < max_resume_attempts:
                    # Record the incident and attempt a "fresh run" using scratch checkpoints.
                    step.metadata.setdefault("context_overflow", [])
                    step.metadata["context_overflow"].append(
                        {
                            "at": datetime.now().isoformat(),
                            "attempt": resume_attempt,
                            "error": str(e),
                        }
                    )

                    # Build a minimal prompt to continue from checkpoint notes.
                    user_prompt = self._build_resume_prompt(step, e)
                    continue

                step.status = TaskStatus.ERRORED
                step.error_msg = str(e)
                return step

        # Should be unreachable, but keep a safe fallback.
        step.status = TaskStatus.ERRORED
        step.error_msg = str(last_error) if last_error else "Unknown error"
        return step

    async def update_task_status(
        self, ctx: RunContext[RuntimeState], task_id: int, status: TaskStatus
    ):
        """Tool: Update Task Status.

        Primarily used by the supervisor to transition a task between states
        (e.g. to ``READY`` or ``COMPLETED``) once dependencies are met or QA
        has passed.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task to update.
            status: New :class:`TaskStatus` value for the task.
        """
        async with self._plan_lock:
            if task_id in ctx.deps.plan:
                ctx.deps.plan.get(task_id).status = status
                return f"Status for {task_id} is now {status}."
            return f"Error: No task with {task_id} found in plan. Be sure task_id actually exists."

    def handle_critic_result(self, task: TaskItem, review: TaskQAResult):
        """Apply the critic's QA result to a task.

        Currently this records only the most recent review and increments the
        task's ``attempt_count``. Higher-level logic can then decide whether to
        retry, patch, or cancel the task.
        """
        task.attempt_count += 1
        # IMPORTANT: For now only stores the latest review information, not previous review rounds
        task.task_feedback = review

        if review.passed:
            task.status = TaskStatus.COMPLETED
            task.error_msg = None
            return

        # QA failed
        if task.attempt_count >= task.max_attempts:
            task.status = TaskStatus.FAILED
            task.error_msg = (
                f"Max retries reached ({task.attempt_count}/{task.max_attempts})."
            )
            return

        task.status = TaskStatus.RERUN
        task.error_msg = None
        task.sub_task_objective = f"{task.sub_task_objective}\n\nPrevious attempt failed review; feedback: {review.reasoning}"

    async def view_qa_report(self, ctx: RunContext[RuntimeState], task_id: int) -> str:
        """Tool: View QA Report.

        Return the full serialized QA report for a specific task, if one is
        available. This is typically called by the supervisor when additional
        inspection of the critic's reasoning is required.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task whose QA report should be viewed.

        Returns:
            A JSON-formatted string representation of the stored
            :class:`TaskQAResult`, or a message describing why no report is
            available.
        """
        async with self._plan_lock:
            task = ctx.deps.plan.get(task_id)
            logger.info(f"Checking QA Report for task: {task_id}")
            if task is None:
                return f"No task with id {task_id}."

            fb = getattr(task, "task_feedback", None)
            if fb is None:
                return f"No QA feedback found for task {task_id}."
            task.metadata.setdefault("qa", {})
            task.metadata["qa"]["report_viewed"] = True

            # Return either a summary or full JSON depending on your needs
            return fb.model_dump_json(indent=2)
