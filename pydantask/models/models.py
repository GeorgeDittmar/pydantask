from typing import Literal, List, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict, Callable
from datetime import datetime
from pydantic_ai import Agent
from regex import F


class TaskStatus(Enum):
    PENDING = "pending"  # Waiting for dependencies
    READY = "ready"  # Dependencies met, can run now
    RUNNING = "running"  # Currently being executed
    COMPLETED = "completed"
    ERRORED = "errored"  # Execution error occurred
    FAILED = "failed"  # Evaluator rejected it
    NEEDS_REVIEW = "needs_review"  # Needs Evaluator review
    RERUN = "rerun"


class TaskQAResult(BaseModel):
    task_id: int
    reasoning: str
    passed: bool = False


class KnowledgeRecord(BaseModel):
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


class TaskResult(BaseModel):
    """
    Canonical result type for any sub-task.

    Works well for research-style tasks (summary + detailed artifacts + sources),
    but can also be used for other task types that just need a summary.
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
            "For research tasks this should summarize key findings; for other tasks "
            "it should summarize what was done and the result."
        ),
    )

    notes: List[str] = Field(
        default_factory=list,
        description="All notes or scratch file paths that were used.",
    )
    detailed_report_paths: List[str] = Field(
        default_factory=list,
        description=(
            "List of file paths to any detailed reports or long-form outputs "
            "generated during this task (e.g. written via write_to_file_system)."
        ),
    )

    # sources: list[str] = Field(
    #     default_factory=list,
    #     description=(
    #         "List of URLs, document IDs, tool references, or other sources "
    #         "used to produce this result."
    #     ),
    # )

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

    task_id: int = Field(description="Unique task id. Should be an integer value.")
    overall_objective: str = Field(
        description="The overall objective this task is contributing to solving."
    )
    sub_task_objective: str = Field(
        description="The sub task objective that must be solved for."
    )
    status: TaskStatus
    result: Optional[TaskResult] = Field(
        description="Where to put the task result if completed."
    )
    capability: str = Field(
        description="Which sub agent capability should attempt this task."
    )
    sub_task_dependencies: Optional[List[int]] = Field(
        description="Put task dependency IDs here", default_factory=list
    )
    task_feedback: Optional[TaskQAResult] = None  # Store the Eval "critique" here
    error_msg: Optional[str] = None  # Store any error messages here
    iteration_history: List = Field(
        default_factory=list,
        description="Store any answer history if multiple attempts are made.",
    )  # Store any answer history if multiple attempts are made
    time_scope: Optional[str]  # "2026", "2025-2026", "last 7 days", etc.
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Dict[int, TaskItem] = Field(
        description="The plan that was generated to solve the objective."
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
    reasoning_steps: str = Field(
        description="Internal reasoning before finalizing the plan"
    )
    tasks: list[TaskItem]


class SubAgentInstruction(BaseModel):
    reasoning: Optional[str] = None
    instructions: str


class TaskSpec(BaseModel):
    task_id: int
    task_objective: str
    capability: Literal["researcher", "writer", "synthesizer"]
    inputs: dict
    success_criteria: str
    constraints: list[str]
    overall_objective: str
