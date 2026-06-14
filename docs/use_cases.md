# Use Cases

Pydantask is meant to be a generalizable deep agent harness that can handle a variety of use cases. This page contains some example use cases for you to see how Pydantask can help in your workflow.

## Deep Research

Deep research problems are where Pydantask shines: multi-step investigations that require web research, cross-checking sources, critique, and final synthesis into a coherent report.

This section walks through a complete Deep Research workflow using `DeepAgent`.

Note: the public API is stable, but some internal attribute names (like the capability registry) are implementation details.

---

## Prerequisites

Before you start, make sure you have:

- Python installed (3.12+).
- Dependencies for Pydantask installed (for example: `pip install -e .` in this repo).
- Environment variables set:
  - `OPENAI_API_KEY` – for the language model.
  - `TAVILY_API_KEY` – for the research agent’s web search tool.
- (Optional) tracing configured (auto-detected) if you want run traces:
  - Langfuse: `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`
  - Logfire: `LOGFIRE_API_KEY` (example)
  - LangSmith: `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY`

You can use a `.env` file and `python-dotenv` to load these automatically.

---

## Conceptual Overview

At a high level, a deep research run with `DeepAgent` follows a simple loop:

1. **You provide an objective.** A single natural-language objective string that describes the research question or report you want.
2. **A dynamic supervisor builds the task graph.** Rather than relying on a separate upfront planner step, the supervisor agent incrementally creates `TaskItem`s at runtime using the `add_task` tool.
3. **Capabilities execute tasks.** The supervisor schedules runnable tasks, and `DeepAgent` executes them using the capability named in `TaskItem.capability` (typically `research_agent` for web research, `worker_agent` for general analysis, and `producer_agent` for synthesis).
4. **A critic reviews results.** The critic (`TaskQAResult`) checks each task output and provides structured QA feedback (pass/fail + reasoning). The harness then applies deterministic status transitions:
   - if QA passes → the task becomes `COMPLETED`
   - if QA fails but retries remain → the task becomes `RERUN` (and the feedback is appended to the task objective)
   - if QA fails and retries are exhausted → the task becomes `FAILED`
5. **Repeat until done.** The loop continues until the supervisor sets `all_tasks_completed=True` (or `max_steps` is reached).
6. **You get a final report.** `DeepAgent.run()` returns a `DeepAgentRunResult` with:
   - `final_result` (a `TaskResult`, when a producer task ran)
   - the final `plan`
   - the full `runtime_state`


---

## Step-by-Step Tutorial: Running a Deep Research Job

In this tutorial, you will:

1. Define a deep research objective.
2. Instantiate `DeepAgent` for that objective.
3. Run the deep agent.
4. Save and inspect the outputs.

Create a file named `example.py`:

```python
import asyncio
import json
from pprint import pprint

from dotenv import load_dotenv, find_dotenv
from pydantask.agents import DeepAgent

# Load environment variables from a .env file (if present):
# - OPENAI_API_KEY (for the language model)
# - TAVILY_API_KEY (for the research_agent's web search tool)
load_dotenv(find_dotenv())

async def main():
    # 1. Define your high-level research objective
    objective = (
        "I need help researching how to scale a machine learning system to millions "
        "of users an hour. I want concrete system design examples, explanations, "
        "and some sample code to show how to build scalable ML systems. "
        "This can cover LLM use cases or classical ML use cases.\n\n"
        "Output all information as a markdown-formatted document and clearly cite "
        "all sources at the end of the report."
    )

    # 2. Create a DeepAgent instance configured for deep research
    da = DeepAgent(
        objective=objective,
        model="gpt-4.1-mini",  # or any supported OpenAI-compatible model
        trace=True,            # enable tracing (auto-detected) if configured via env vars
    )

    # 3. Run the deep research workflow
    run_result = await da.run()

    # 4. Inspect the final result object
    final_result = run_result.final_result

    # Save full structured run output as JSON (for debugging / auditing)
    with open("deep_research_run.json", "w", encoding="utf-8") as f:
        f.write(run_result.model_dump_json(indent=2))

    # Save the final markdown report
    if final_result is not None:
        with open("deep_research_report.md", "w", encoding="utf-8") as f:
            f.write(final_result.detailed_output)

        print("\n=== Final Report (truncated) ===\n")
        print(final_result.detailed_output[:2000])  # show first 2k chars
    else:
        print("No final_result produced. Check the run_result.plan and errors.")
        pprint(run_result.errors)
        # Optionally inspect run_result.plan here for debugging

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Example

From the root of your project, run:

```bash
python example.py
```

After it finishes, you should see:

- `deep_research_run.json` – a structured dump of the entire `DeepAgentRunResult`.
- `deep_research_report.md` – the final markdown report generated by the producer agent.

You can open `deep_research_report.md` in any markdown viewer or editor to read the full research output, complete with sources and structure.

---

## What Happens Under the Hood

When you run the script above:

- `DeepAgent` initializes a shared `RuntimeState` (objective, empty plan, capability registry).

- `DeepAgent` constructs and wires together:

  - A **supervisor agent** (system prompt: `DYNAMIC_SUPERVISOR_SYS_PROMPT`) that:
    - inspects the current task DAG (“mission control board”)
    - decides which tasks to run next (`SupervisorDecision.tasks_to_execute`)
    - can mutate the plan at runtime using tools:
      - `add_task`, `patch_task`, `cancel_task`, `update_task_status`
      - `view_qa_report` (inspect critic feedback if it was stored)

  - A **critic agent** (system prompt: `CRITIC_SYS_PROMPT`) that:
    - evaluates each executed task’s `TaskResult`
    - returns a `TaskQAResult(passed=..., reasoning=...)`

  - Deterministic QA transitions applied by `handle_critic_result(...)`:
    - if `passed=True` → `TaskItem.status = COMPLETED`
    - else if `attempt_count >= max_attempts` → `TaskItem.status = FAILED`
    - otherwise → `TaskItem.status = RERUN` and the critic feedback is appended to the task objective

  - A capability registry (`DeepAgent._capability_registry`) that contains (by default):
    - `research_agent`: uses `tavily_search_tool` when `TAVILY_API_KEY` is set,
      otherwise a DuckDuckGo-based search tool, to gather and cite sources.
    - `producer_agent`: synthesizes across completed tasks into a final
      `TaskResult`.
    - `worker_agent`: a general-purpose worker used for analysis,
      summarization, and other non-research tasks operating on existing
      context.

- Execution is dependency-aware and parallelized:
  - The supervisor may schedule multiple tasks.
  - `DeepAgent` only runs tasks whose dependencies are all `COMPLETED`, and runs eligible tasks concurrently.

This gives you:

- A high-level final answer in `final_result`.
- A fully inspectable execution trace in `plan` and `runtime_state`.
- A set of intermediate artifacts (files, notes, summaries) you can reuse in future runs.

---

## Adapting the Pattern

You can adapt this pattern to any deep research question by:

- Changing the `objective` to match your topic.
- Optionally adding custom capabilities via the `sub_agents` argument to `DeepAgent`.
- Adjusting `max_steps` or `set_token_budget` for longer or shorter runs.
- Inspecting `run_result.plan` and `run_result.runtime_state` for debugging, analytics, or UI visualization.

Pydantask handles the orchestration — dynamic task-graph growth, dependency-aware execution, QA-driven retries, and synthesis — so you can focus on defining the research problem and consuming the final structured output.