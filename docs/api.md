# API Reference

_This section is for detailed usage and code documentation._

This page aggregates the auto-generated API documentation for the main classes and functions in Pydantask. The content is rendered from the docstrings in your code via the configured documentation tool (e.g. mkdocstrings).

---

## Orchestrator

### DeepAgent

High-level orchestrator that coordinates planning, supervision, execution, and QA across sub‑agents.

```python
from pydantask.agents import DeepAgent
```

::: pydantask.agents.DeepAgent

---

## Core Models

These models define the task/plan structure, runtime state, and capability descriptions used by `DeepAgent` and sub‑agents.

### Task and Plan Models

## TaskStatus

::: pydantask.models.TaskStatus

---

## TaskItem

::: pydantask.models.TaskItem

---

::: pydantask.models.TaskResult

---

::: pydantask.models.TaskQAResult

---

::: pydantask.models.Plan

---

::: pydantask.models.SupervisorDecision

---

### Runtime and Capability Models

::: pydantask.models.RuntimeState

---

::: pydantask.models.CapabilityDescription

---

::: pydantask.models.KnowledgeRecord

---

See the source code for any additional helpers and the most up‑to‑date details.

