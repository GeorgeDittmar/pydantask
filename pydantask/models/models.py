from typing import Literal, List, Union
from enum import Enum
from attr import field
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict, Callable
from datetime import datetime
from pydantic_ai import Agent
from regex import F


class TracingBackend(Enum):
    LANGFUSE = "langfuse"
    LOGFIRE = "logfire"
    LANGSMITH = "langsmith"
    NONE = "none"

class TaskStatus(Enum):
    """Lifecycle state for a :class:`TaskItem` within a DeepAgent plan.

    Values:
        PENDING: Waiting for dependencies to complete. \n
        READY: All dependencies met; eligible to run. \n
        RUNNING: Currently being executed by a sub-agent. \n
        COMPLETED: Successfully finished and accepted. \n
        ERRORED: Execution error occurred (tools, runtime, etc.). \n
        FAILED: Evaluator/critic rejected the result. \n
        NEEDS_REVIEW: Needs evaluator/supervisor review. \n
        RERUN: Marked to be re-executed with revised instructions. \n
        CANCELLED: Task that is no longer needed or relevant to complete the objective. \n
    """

    PENDING = "pending"  # Waiting for dependencies
    READY = "ready"  # Dependencies met, can run now
    RUNNING = "running"  # Currently being executed
    COMPLETED = "completed"
    ERRORED = "errored"  # Execution error occurred
    FAILED = "failed"  # Evaluator rejected it
    NEEDS_REVIEW = "needs_review"  # Needs Evaluator review
    RERUN = "rerun"
    CANCELLED = "cancelled"


class TaskQAResult(BaseModel):
    """Evaluation result produced by the critic/QA agent for a single task.

    Attributes:
        task_id: ID of the :class:`TaskItem` being evaluated. It MUST match the task that is being evaluated.
        reasoning: Detailed explanation of how the result was judged and why you scored the way you did.
        passed: True if the worker output sufficiently meets the task objective.
    """

    task_id: int = Field(
        default=-1,
        description="The task_id for the task being evaluated. Ex. Criticing task has task_id of 1, thus the task_id of this field will also be 1.",
    )
    reasoning: str = Field(
        default="",
        description="Detailed explanation of how the result was judged, why it god the score that it did and feedback for supervisor to attempt a retry.",
    )
    passed: bool = Field(
        default=False,
        description="Whether the task passed qa/critic. True if you found that the worker output sufficiently meets the task objective.",
    )


class KnowledgeRecord(BaseModel):
    """Logical record of a knowledge artifact (file, summary, notes, etc.).

    Stored in :class:`RuntimeState.knowledge_store` to track long-lived documents
    and their relationship to tasks.

    Attributes:
        id: Logical identifier (what tools use as key).
        path: Filesystem path if this record is backed by a file.
        task_ids: IDs of TaskItems this record is associated with.
        summary: Short human-readable description of the content.
        source_task_ids: IDs of tasks that produced or updated this record.
        created_at: Timestamp when the record was created.
        metadata: Arbitrary extra info (tags, type=research/plan/etc.).
    """

    id: str = Field(description="Logical identifier (what tools use as key).")
    path: Optional[str] = Field(
        default=None, description="Filesystem path if this is backed by a file."
    )
    task_ids: list[int] = Field(
        description="which TaskItems this relates to (if any)", default_factory=list
    )
    summary: Optional[str] = Field(
        default=None, description="Short human-readable description of the content."
    )
    source_task_ids: List[int] = Field(
        default_factory=list,
        description="Tasks that produced or updated this document.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this knowledge item was created.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary extra info (tags, type=research/plan/etc).",
    )


class SourceRef(BaseModel):
    """Structured citation / source reference used in ``TaskResult.sources``.

    Each ``id`` is intended to be referenced from text as ``[id]`` (e.g., ``[1]``).

    Attributes:
        id: Short integer identifier used in inline citations.
        kind: Type of source (web page, document, code, data, other).
        title: Human-readable title of the source, if available.
        url: URL for online sources.
        path: Filesystem path or document ID for local artifacts.
        snippet: Short excerpt of key evidence taken from this source.
        accessed_at: When this source was accessed (for date-sensitive content).
        metadata: Extra structured info (author, publisher, etc.).
    """

    id: int = Field(
        description=(
            "Short identifier used in inline citations, e.g. '1', '2'. "
            "The agent should use these IDs inside the text like [1], [2]."
        )
    )
    kind: Literal["web", "document", "code", "data", "other"] = Field(
        description="Type of source (web page, file, code snippet, etc.)."
    )
    title: Optional[str] = Field(
        default=None,
        description="Human-readable title of the source, if available.",
    )
    url: Optional[str] = Field(
        default=None,
        description="URL if this is an online source.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Filesystem path / doc ID if this is a local artifact.",
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Short excerpt of the key evidence used from this source. No more than 2-3 sentences",
    )
    accessed_at: Optional[datetime] = Field(
        default=None,
        description="When this source was accessed (for web/date-sensitive content).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any extra structured info (author, publisher, etc.).",
    )


class TaskResult(BaseModel):
    """Canonical result type for any sub-task executed by DeepAgent.

    TaskResult stores the output from any given task.

    Attributes:
        task_id: ID of the :class:`TaskItem` this result belongs to.
        status: Outcome of this task execution (COMPLETED/ERRORED/FAILED/etc.).
        summary: Human-readable summary of what this task produced or concluded.
        notes: Any notes or scratch references used while completing this task.
        output_paths: Logical or filesystem paths to long-form outputs (reports,
            code files, etc.) produced by this task.
        sources: Structured list of :class:`SourceRef` citations used in this
            result. Inline citations should reference ``SourceRef.id`` values.
        error_msg: Explanation of what went wrong if the task errored or failed.
        metadata: Optional free-form metadata specific to this task execution.
    """

    task_id: int = Field(description="ID of the TaskItem this result belongs to.")

    status: TaskStatus = Field(
        default=TaskStatus.COMPLETED,
        description="Outcome of this specific task execution.",
    )

    summary: str = Field(
        default="",
        description=(
            "Concise, human-readable summary of what this task produced or concluded. "
        ),
    )

    detailed_output: str = Field(
        default="", description="Detailed output for the task if required."
    )

    notes: List[str] = Field(
        default_factory=list,
        description="All notes or scratch file paths that were used to help complete this task.",
    )

    sources: List[SourceRef] = Field(
        default_factory=list,
        description=(
            "Structured list of sources used to produce this result. "
            "Inline citations in summaries should reference SourceRef.id values, "
            "e.g. [1], [2]."
        ),
    )

    error_msg: Optional[str] = Field(
        default=None,
        description=(
            "If status is ERRORED or FAILED, a clear explanation of what went "
            "wrong or what information/tools were missing."
        ),
    )

    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Optional free-form metadata (e.g. timestamps, scoring, extra flags) "
            "specific to this task execution."
        ),
    )


class TaskItem(BaseModel):
    """One sub-task in a DeepAgent plan.

    Created by the planner and then updated over the life of the run as it moves
    through :class:`TaskStatus` states and accumulates results and feedback.

    Attributes:
        task_id: Unique integer task identifier. Integer value.
        overall_objective: The overall objective this task contributes to.
        sub_task_objective: The specific objective for this sub-task.
        status: Current lifecycle state of the task.
        result: The :class:`TaskResult` produced by the assigned capability, if
            the task has been executed.
        capability: Name of the capability (sub-agent) that should execute this
            task (e.g. "research_agent", "producer_agent").
        sub_task_dependencies: IDs of tasks that must be COMPLETED before this
            task is eligible to run.
        task_feedback: Latest :class:`TaskQAResult` from the critic/QA agent.
        error_msg: Any error message associated with this task.
        iteration_history: Optional history of prior attempts/outputs.
        time_scope: Optional temporal scope string ("2026", "last 7 days", etc.).
        parameters: Arbitrary structured parameters (e.g. supervisor feedback).
        attempt_count: How many times this task has been attempted.
        max_attempts: Maximum times this task is allowed to be attempted.
        metadata: Optional free-form metadata for this task.
    """

    task_id: int = Field(description="Unique task id. Should be an integer")
    overall_objective: str = Field(
        description="The overall objective this task is contributing to solving."
    )
    sub_task_objective: str = Field(
        description="The sub task objective that must be solved for."
    )
    status: TaskStatus
    result: Optional[TaskResult] = Field(
        description="Where to put the task result if completed.", default=None
    )
    capability: str = Field(
        description="Which sub agent capability should attempt this task."
    )
    sub_task_dependencies: Optional[List[int]] = Field(
        description="Put task_id dependency IDs here", default_factory=list
    )
    task_feedback: Optional[TaskQAResult] = None  # Store the Eval "critique" here
    error_msg: Optional[str] = Field(
        default=None,
        description="Any errors that happened during the running of this task. Only store the more recent error message.",
    )  # Store any error messages here
    iteration_history: List = Field(
        default_factory=list,
        description="Store any answer history if multiple attempts are made.",
    )  # Store any answer history if multiple attempts are made
    time_scope: Optional[str] = Field(
        default=None, description="2026, 2021-2025, two days ago"
    )  # "2026", "2025-2026", "last 7 days", etc.
    parameters: dict = Field(
        default_factory=dict
    )  # you can stash structured temporal params here
    attempt_count: int = 0
    max_attempts: int = 3
    metadata: dict = Field(
        default_factory=dict, description="Optional metadata for this task."
    )

    @property
    def latest_output(self):
        return self.iteration_history[-1].output if self.iteration_history else None


# Agent/Tool Description Object
class CapabilityDescription(BaseModel):
    """Metadata describing a capability (sub-agent or tool) available to DeepAgent.

    The planner and supervisor see ``name`` and ``description`` when deciding
    which capability to assign to ``TaskItem.capability``. ``tool_func`` holds
    the underlying implementation (often a :class:`pydantic_ai.Agent` or
    callable tool).

    Attributes:
        name: Identifier of the capability (e.g. "research_agent").
        description: Human-readable description of what this capability does.
        tool_func: Concrete implementation (agent instance or callable tool).
        input_schema: Optional Pydantic model class describing structured input
            for this capability.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="Name of the agent/capability, e.g. 'web_search', 'file_writer', etc."
    )
    description: str = Field(
        description="Human-readable description of what this capability does."
    )
    # We keep this very loose to avoid Pydantic trying to introspect complex
    # types like `pydantic_ai.Agent` (which can reference optional imports
    # and cause schema generation issues). At runtime this will typically be
    # either a pydantic_ai Agent instance or a callable tool.
    tool_func: Any = Field(
        description=(
            "Concrete implementation of the capability. Usually a pydantic_ai Agent "
            "instance or an async/sync callable used as a tool."
        )
    )
    # Optional Pydantic model *class* describing structured input, if you
    # want to be explicit about what this capability expects.
    input_schema: Optional[type[BaseModel]] = Field(
        default=None,
        description=(
            "Optional Pydantic model class that defines the expected input schema "
            "for this capability, for better prompting and validation."
        ),
    )


class RuntimeState(BaseModel):
    """Shared mutable state passed between agents during a DeepAgent run.

    Holds the current plan, objective, registry of capabilities, and simple
    in-memory stores for documents and knowledge.

    Attributes:
        plan: Mapping from task_id to :class:`TaskItem` for the current plan.
        objective: The overall user objective being solved.
        agent_registry: Mapping from capability name to its description/
            implementation (excluded from serialization).
        completed_steps: Set of task_ids that have been completed.
        runtime_steps: Number of outer control-loop cycles executed so far.
        tokens_used: Placeholder for token accounting.
        task_queue: Optional queue of additional tasks.
        knowledge_store: Mapping of logical IDs to :class:`KnowledgeRecord`s.
        document_store: Mapping of logical document names to filesystem paths.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Dict[int, TaskItem] = Field(
        description="The plan that is generated to solve the users objective.",
        default_factory=dict,
    )
    objective: str = Field(description="The overall objective to solve for.")
    agent_registry: Dict[str, Any] = Field(
        description="Agents available to perform tasks.",
        default_factory=dict,
        exclude=True,
    )
    completed_steps: set[int] = Field(
        description="Steps from the plan that have been completed.", default_factory=set
    )
    next_task_id: int = Field(
        description="The next valid Task id value that could be added to a running plan"
    )
    runtime_steps: int = 0
    tokens_used: int = 0
    task_queue: List[TaskItem] = Field(default_factory=list)
    knowledge_store: Dict[str, KnowledgeRecord] = Field(
        default_factory=dict,
        description=(
            "Knowledge Store: maps logical IDs to  "
            "(files, summaries, notes, etc.)."
            "Accumulated information is stored here for other agents to use if needed."
        ),
    )  # simple in-memory document store
    document_store: Dict[str, str] = Field(
        description="Documents that are written to the file system if needed for review.",
        default_factory=dict,
    )


class SupervisorDecision(BaseModel):
    """Supervisor's decision for a single control-loop iteration.

    Attributes:
        reasoning: Explanation of why particular tasks were chosen or why the
            run should end.
        tasks_to_execute: IDs of tasks that should be executed next.
        feedback_to_subagents: Optional mapping from task_id to feedback string
            to be injected into sub-agent execution.
        all_tasks_completed: True if the plan is complete or cannot be
            progressed further.
    """

    # status: Literal["DELEGATE", "REPLAN", "COMPLETE", "ERROR"]
    reasoning: str = Field(
        description="Reasoning for why these tasks need to be completed next or the reasoning for when we are done executing."
    )
    tasks_to_execute: List[int] = Field(description="List of task id's to execute.")
    feedback_to_subagents: Optional[Dict[int, str]] = Field(
        default=None,
        description="Any feedback to the sub-agents if additional context or instructions needs to be given to the sub agent for a particular task. Dict is key: task_id, value: feedback for subagent for the given task.",
    )
    all_tasks_completed: bool = Field(
        default=False,
        description="Set this to true if all tasks in the plan have been completed OR there are too many errors that the plan cannot be completed.",
    )


# =========================
# Planner state models
# =========================
class Plan(BaseModel):
    """Planner output: internal reasoning plus the list of tasks.

    ``DeepAgent`` converts ``tasks`` into a ``Dict[int, TaskItem]`` for
    :class:`RuntimeState.plan`.

    Attributes:
        reasoning_steps: Planner's internal reasoning/chain-of-thought.
        tasks: Ordered list of :class:`TaskItem` definitions.
    """

    reasoning_steps: str = Field(
        description="Internal reasoning before finalizing the plan"
    )
    tasks: list[TaskItem]


class SubAgentInstruction(BaseModel):
    """Helper model for passing structured instructions to a sub-agent.

    Attributes:
        reasoning: Optional reasoning or context for the instructions.
        instructions: Concrete instructions the sub-agent should follow.
    """

    reasoning: Optional[str] = None
    instructions: str


class TaskSpec(BaseModel):
    """Specification for a planned task, useful for external planners/tools.

    Attributes:
        task_id: ID of the task.
        task_objective: Human-readable objective for this task.
        capability: High-level capability type (researcher, writer, synthesizer).
        inputs: Structured inputs the task expects.
        success_criteria: How to decide if the task was successful.
        constraints: List of constraints or guidelines for execution.
        overall_objective: The broader objective this task serves.
    """

    task_id: int
    task_objective: str
    capability: Literal["researcher", "writer", "synthesizer"]
    inputs: dict
    success_criteria: str
    constraints: list[str]
    overall_objective: str


class DeepAgentRunResult(BaseModel):
    """High-level summary of a DeepAgent run.

    Wraps the final :class:`TaskResult` (if any) together with the final plan,
    runtime statistics, and high-level status. Suitable as a public return type
    from :meth:`DeepAgent.run`.

    Attributes:
        objective: The original user objective.
        final_result: The final :class:`TaskResult` synthesized by the
            ``producer_agent``, if any.
        status: High-level outcome of the run (success/partial/failed).
        plan: Final mapping from task_id to :class:`TaskItem` after execution.
        runtime_steps: Number of DeepAgent control-loop cycles executed.
        errors: Any top-level errors or important warnings.
    """

    objective: str = Field(..., description="The original user objective.")
    final_result: Optional[TaskResult] = Field(
        default=None,
        description="The final TaskResult synthesized by the producer_agent, if any.",
    )
    # status: Literal["success", "partial", "failed"] = Field(
    #     ..., description="High-level outcome of the run."
    # )
    plan: Dict[int, TaskItem] = Field(
        ...,
        description="The final plan state (all TaskItems after execution).",
    )
    runtime_state: RuntimeState = Field(
        description="Complete runtime state for auditing."
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any top-level errors or important warnings.",
    )
