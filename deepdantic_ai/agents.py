from typing import List, Optional, Literal, Dict, Callable
from pydantic import BaseModel

# from pydantic_graph import GraphNode  # Uncomment when ready to use Pydantic Graph
from pydantic_ai.agent import Agent  # Stub for PydanticAI

# ---------------------------
# Core State Models (Graph-Ready)
# ---------------------------


class ToolCall(BaseModel):
    name: str
    args: Dict


class ToolResult(BaseModel):
    name: str
    output: Dict
    success: bool = True
    error: Optional[str] = None


# Graph-ready step (can later inherit from GraphNode)
class ResearchStep(BaseModel):
    id: Optional[str] = None  # unique identifier for graph
    action: Literal[
        "decompose_question",
        "generate_hypothesis",
        "search_evidence",
        "evaluate_claim",
        "synthesize",
        "self_critique",
    ]
    rationale: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    confidence: Optional[float] = None
    # Later: add dependencies in graph
    dependencies: Optional[List[str]] = []


# Graph-ready state (nodes can be added later)
class AgentState(BaseModel):
    question: str
    subquestions: List[str] = []
    hypotheses: List[str] = []
    evidence: List[Dict] = []
    claims: List[str] = []
    confidence: float = 0.0
    open_gaps: List[str] = []
    steps: List[ResearchStep] = []


# ---------------------------
# Tool Registry
# ---------------------------


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self.tools[name] = func

    def execute(self, tool_call: ToolCall) -> ToolResult:
        func = self.tools.get(tool_call.name)
        if not func:
            return ToolResult(
                name=tool_call.name, output={}, success=False, error="Tool not found"
            )
        try:
            output = func(**tool_call.args)
            return ToolResult(name=tool_call.name, output=output)
        except Exception as e:
            return ToolResult(
                name=tool_call.name, output={}, success=False, error=str(e)
            )


# ---------------------------
# Graph-Ready Deep Agent Framework
# ---------------------------


class DeepAgent:
    def __init__(
        self, name: str, ai_model: Agent, tools: Optional[ToolRegistry] = None
    ):
        self.name = name
        self.ai_model = ai_model
        self.tool_registry = tools or ToolRegistry()
        self.state: Optional[AgentState] = None
        self.step_counter = 0

    def initialize(self, question: str):
        self.state = AgentState(question=question)
        self.step_counter = 0

    def plan_next_step(self) -> ResearchStep:
        """LLM generates a typed ResearchStep (graph-ready)."""
        # For graph readiness, assign a unique ID to each step
        step_id = f"step_{self.step_counter}"
        step = self.ai_model.run(ResearchStep, input=self.state.dict())
        step.id = step_id
        # Optional: specify dependencies later
        step.dependencies = []  # populate with IDs of steps this depends on
        self.step_counter += 1
        return step

    def execute_step(self, step: ResearchStep) -> ToolResult:
        if step.tool_call:
            result = self.tool_registry.execute(step.tool_call)
            step.tool_result = result
            return result
        step.tool_result = ToolResult(name="none", output={}, success=True)
        return step.tool_result

    def update_state(self, step: ResearchStep):
        """Update agent state based on step."""
        self.state.steps.append(step)
        if step.action == "generate_hypothesis":
            self.state.hypotheses.append(step.rationale)
        elif step.action == "search_evidence" and step.tool_result:
            self.state.evidence.append(step.tool_result.output)
        elif step.action == "evaluate_claim":
            self.state.confidence = step.confidence or self.state.confidence
        elif step.action == "synthesize":
            self.state.claims.append(step.rationale)
        elif step.action == "self_critique":
            self.state.confidence = step.confidence or self.state.confidence

    def run(self, max_steps: int = 10):
        for _ in range(max_steps):
            step = self.plan_next_step()
            self.execute_step(step)
            self.update_state(step)
            if step.action == "synthesize":
                break
        return self.state


# ---------------------------
# Example Tool
# ---------------------------


def dummy_search(query: str) -> dict:
    return {"results": [f"Evidence for {query}"]}


# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    ai_model = Agent(model="gpt-5")  # PydanticAI stub
    tools = ToolRegistry()
    tools.register("search", dummy_search)

    agent = DeepAgent(name="DeepDanticGraphAgent", ai_model=ai_model, tools=tools)
    agent.initialize("What causes urban heat islands?")
    final_state = agent.run(max_steps=5)
    print(final_state.json(indent=2))
