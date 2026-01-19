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
    FAILED = "failed"  # Evaluator rejected it

class VerifiedSegment(BaseModel):
    segment_id: str
    content: str
    is_valid: bool
    reason: Optional[str] = None # Why it failed, if it did

class PartialEvaluation(BaseModel):
    all_passed: bool
    segments: List[VerifiedSegment]
    summary_feedback: str # Instructions for the next iteration
    
class TaskIteration(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    output: Any
    evaluation: Optional[PartialEvaluation] = None
    prompt_used: str
    

class TaskItem(BaseModel):
    id: str
    description: str
    status: TaskStatus
    capability: str
    task_dependencies: Optional[List[int]] = Field(description="Put task dependency IDs here",default_factory=list)
    review_feedback: Optional[str] = None  # Store the "critique" here
    iteration_history = list[]
    attempt_count: int = 0
    max_attempts: int = 3

    @property
    def latest_output(self):
        return self.history[-1].output if self.history else None

# Tool Desription Object
class ToolDescription(BaseModel):
    description: str
    tool_func: Callable
    
# Agent Description Object
class AgentDescription(BaseModel):
    description: str
    agent_func: Agent

class ResearchResult(BaseModel):
    summary: str
    information: list[str]
    sources: list[str]

class RuntimeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    plan: Dict[str, TaskItem]
    capability_registry: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    completed_steps: set[int] = Field(default_factory=set)
    research_results: Dict[int, ResearchResult] = Field(default_factory=dict)
    accumulated_knowledge_store: Dict[str, str] = Field(
        default_factory=dict
    )  # store for accumulated knowledge
    runtime_steps: int = 0
    tokens_used: int = 0
    goal: str 

class SupervisorDecision(BaseModel):
    status: Literal["DELEGATE", "REPLAN", "COMPLETE", "ERROR"]
    tasks: List[TaskItem]
    reasoning: str
    feedback_for_planner: Optional[str] = None
    
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