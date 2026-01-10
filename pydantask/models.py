from typing import Literal, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict


class TaskItem(BaseModel):
    id: str
    description: str
    status: str = "pending"
    capability: Literal[
        "researcher",
        "writer",
        "synthesiser",
        # "external_interaction",
        # "creative_problem_solving",
        # "collaboration",
    ] = "research"
    task_dependencies: Optional[List[int]] = Field(description="Put task dependency IDs here",default_factory=list)
    review_feedback: Optional[str] = None  # Store the "critique" here
    attempt_count: int = 0
    max_attempts: int = 3

class ResearchResult(BaseModel):
    summary: str
    information: list[str]
    sources: list[str]

class RuntimeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    plan: list[TaskItem]
    agent_registry: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    completed_steps: set[int] = Field(default_factory=set)
    research_results: Dict[int, ResearchResult] = Field(default_factory=dict)
    accumulated_knowledge_store: Dict[str, str] = Field(
        default_factory=dict
    )  # store for accumulated knowledge
    runtime_steps: int = 0
    tokens_used: int = 0
    tool_available: Literal["research_agent", "writer_agent", "web_search"] = (
        "web_search"
    )
    goal: str = Field(
        default="Perform a websearch for microsoft and tell me what you know."
    )

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