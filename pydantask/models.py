from typing import Literal, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict, Callable
from datetime import datetime
from pydantic_ai import Agent


class TaskStatus(Enum):
    PENDING = "pending"  # Waiting for dependencies
    READY = "ready"  # Dependencies met, can run now
    RUNNING = "running"  # Currently being executed
    COMPLETED = "completed"
    ERROR = "error"
    FAILED = "failed"  # Evaluator rejected it
    REVIEW = "review"  # Needs Evaluator review


class TaskQAResult(BaseModel):
    task_id: str
    reasoning: str
    passed: bool = False


class TaskResult(BaseModel):
    task_id: str
    task_status: TaskStatus
    output: Any
    error_msg: Optional[str] = None


class TaskItem(BaseModel):
    id: str
    description: str
    status: TaskStatus
    result: Optional[TaskResult]
    capability: str
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
    summary: str = Field(description="Concise summary of findings.")
    detailed_report_path: list[str] = Field(
        description="path to detailed report files containing in-depth information."
    )


class RuntimeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: Dict[str, TaskItem]
    agent_registry: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    completed_steps: set[int] = Field(default_factory=set)
    accumulated_knowledge_store: Dict[str, str] = Field(
        default_factory=dict
    )  # store for accumulated knowledge
    runtime_steps: int = 0
    tokens_used: int = 0
    goal: str
    task_queue: List[TaskItem] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    # status: Literal["DELEGATE", "REPLAN", "COMPLETE", "ERROR"]
    reasoning: str = Field(
        description="Reasoning for why these tasks need to be completed next."
    )
    tasks_to_execute: List[TaskItem] = Field(description="List of tasks to execute.")
    feedback_for_planner: Optional[str] = Field(
        default=None,
        description="Any feedback to the planner if a task has become blocked.",
    )


class NextAction(BaseModel):
    """Next action to be taken by the supervisor agent."""

    reasoning: str
    action_type: Literal["delegate", "complete"]
    target_agent: Optional[str] = None
    task_spec: Optional[TaskItem] = None


# =========================
# Planner state models
# =========================
class Plan(BaseModel):
    tasks: list[TaskItem]


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


class ToolResult(BaseModel):
    result: str
