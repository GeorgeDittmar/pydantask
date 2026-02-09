# Agent Concepts

Agents are the core entities in Pydantask. They:
- Accept a prompt or task
- Access a set of tools (simple functions or sub-agents)
- Can operate with access to persistent shared runtime state

## Built-In Agents

- **Planner**: Breaks down the main objective into discrete tasks
- **Critic**: Evaluates outputs of tasks/subtasks for quality
- **Supervisor**: Oversees execution, advancing task states
- **Researcher**: Searches for new information to fulfill plan steps

Agents may invoke tools or sub-agents recursively to accomplish their objectives.
