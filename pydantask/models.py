from typing import Literal, List
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
    NEEDS_REVIEW = "review"  # Needs Evaluator review


class TaskResult(BaseModel):
    task_id: int
    task_status: TaskStatus
    work_summary: str = ""
    output_path: Optional[str] = None
    error_msg: Optional[str] = None


class TaskQAResult(BaseModel):
    task_id: int
    reasoning: str
    passed: bool = False
    task_output: TaskResult


class TaskItem(BaseModel):
    task_id: int = Field(description="Unique task id. Should be an integer value.")
    overall_objective: str = Field(
        description="The overall objective this task is contributing to solving."
    )
    task_objective: str = Field(
        description="Description of the sub task to be executed."
    )
    status: TaskStatus
    result: Any = None  # Store TaskResult here after execution
    capability: str = Field(
        description="Which sub agent capability should attempt this task."
    )
    task_dependencies: Optional[List[int]] = Field(
        description="Put task dependency IDs here", default_factory=list
    )
    task_feedback: Optional[TaskQAResult] = None  # Store the Eval "critique" here
    error_msg: Optional[str] = None  # Store any error messages here
    iteration_history: list = (
        []
    )  # Store any answer history if multiple attempts are made
    attempt_count: int = 0
    max_attempts: int = 3
    metadata: dict = {}

    @property
    def latest_output(self):
        return self.iteration_history[-1].output if self.iteration_history else None


# Tool Desription Object
class ToolDescription(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    description: str
    tool_func: Any


class ResearchResult(BaseModel):
    task_id: int
    summary: str = Field(description="Concise summary of findings.")
    detailed_results_path: list[str] = Field(
        description="path to detailed report files containing in-depth information."
    )


class RuntimeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Dict[int, TaskItem]
    objective: str
    agent_registry: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    completed_steps: set[int] = Field(default_factory=set)
    accumulated_knowledge_store: Dict[str, str] = Field(
        default_factory=dict
    )  # store for accumulated knowledge
    runtime_steps: int = 0
    tokens_used: int = 0
    task_queue: List[TaskItem] = Field(default_factory=list)
    document_store: Dict[str, str] = Field(
        default_factory=dict,
        description="A simple in-memory document store for storing and retrieving documents by ID.",
    )  # simple in-memory document store


class SupervisorDecision(BaseModel):
    # status: Literal["DELEGATE", "REPLAN", "COMPLETE", "ERROR"]
    reasoning: str = Field(
        description="Reasoning for why these tasks need to be completed next."
    )
    tasks_to_execute: List[int] = Field(description="List of task id's to execute.")
    feedback_to_subagent: Optional[str] = Field(
        default=None,
        description="Any feedback to the sub-agent if a task has become blocked, errored, or failed critic review.",
    )
    all_tasks_completed: bool = Field(
        default=False, description="Indicates if all tasks are completed or not."
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


class SelfReflection(BaseModel):
    reflection: str
