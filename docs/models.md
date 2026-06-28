# Models

This page documents the core PydanTask models as defined in `pydantask.models.models`.

These models define:

- Task lifecycle and evaluation (`TaskStatus`, `TaskItem`, `TaskResult`, `TaskQAResult`)
- Knowledge and citations (`KnowledgeRecord`, `SourceRef`)
- Capabilities and runtime state (`CapabilityDescription`, `RuntimeState`)
- Supervisor and planner structures (`SupervisorDecision`, `Plan`, `SubAgentInstruction`, `TaskSpec`)
- Run/tracing utilities (`DeepAgentRunResult`, `TracingBackend`)

All content below is rendered from the code via mkdocstrings.

---

## Task Lifecycle

### `TaskStatus`

::: pydantask.models.TaskStatus

---

### `TaskQAResult`

::: pydantask.models.TaskQAResult

---

### `TaskResult`

::: pydantask.models.TaskResult

---

### `TaskItem`

::: pydantask.models.TaskItem

---

## Knowledge and Sources

### `KnowledgeRecord`

::: pydantask.models.KnowledgeRecord

---

### `SourceRef`

::: pydantask.models.SourceRef

---

## Capabilities and Runtime State

### `CapabilityDescription`

::: pydantask.models.CapabilityDescription

---

### `RuntimeState`

::: pydantask.models.RuntimeState

---

## Supervisor and Planner Models

### `SupervisorDecision`

::: pydantask.models.SupervisorDecision

---

### `Plan`

::: pydantask.models.Plan

---

## Workflow YAML Config Models

These models define the strict YAML contract used by `import_yaml_workflow(...)`.

### `WorkflowYamlConfig`

::: pydantask.models.WorkflowYamlConfig

---

### `WorkflowTaskConfig`

::: pydantask.models.WorkflowTaskConfig

---

### `SubAgentInstruction`

::: pydantask.models.SubAgentInstruction

---

### `TaskSpec`

::: pydantask.models.TaskSpec

---

## Run and Tracing

### `DeepAgentRunResult`

::: pydantask.models.DeepAgentRunResult

---

### `TracingBackend`

::: pydantask.models.TracingBackend

---
