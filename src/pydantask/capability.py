from pydantic import BaseModel
from pydantic_ai import Agent


class Capability(BaseModel):
    description: str
    capability_id: str
    capability: Agent
