"""Report generation helper — renders reports/lab_report.md from a MetricsReport."""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from .metrics import MetricsReport

STUDENT_NAME = "Bùi Minh Long"
STUDENT_ID = "2A202601462"
REPO_URL = "https://github.com/ZotAZaw3/phase2-k3-4-track3-day8-langgraph-agent"


def _current_commit() -> str:
    """Resolve the commit this report describes; blank outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "(not a git checkout)"
    return result.stdout.strip()

ARCHITECTURE = """START → intake → classify → [route_after_classify]
  simple       → answer → finalize → END
  tool         → tool → evaluate → [route_after_evaluate]
                                     success     → answer → finalize → END
                                     needs_retry → retry → [route_after_retry]
                                                             attempt < max → tool (loop)
                                                             else          → dead_letter → finalize
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → [route_after_approval]
                                             approved → tool → evaluate → ...
                                             rejected → clarify → finalize → END
  error        → retry → [route_after_retry] → ..."""

STATE_FIELDS = [
    ("route", "overwrite", "only the current classification matters"),
    ("risk_level", "overwrite", "current risk assessment"),
    ("attempt", "overwrite", "counter, bounded against max_attempts"),
    ("evaluation_result", "overwrite", "gate read by route_after_evaluate"),
    ("pending_question", "overwrite", "latest clarification asked back to the user"),
    ("proposed_action", "overwrite", "single action awaiting approval"),
    ("approval", "overwrite", "latest HITL decision, read by route_after_approval"),
    ("final_answer", "overwrite", "one answer per run"),
    ("messages", "append", "conversation trace"),
    ("tool_results", "append", "every tool call kept for grounding + evaluation"),
    ("errors", "append", "failure history must not be lost across retries"),
    ("events", "append", "audit trail; nodes_visited and retry counts are derived from it"),
]


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    failed = [item for item in metrics.scenario_metrics if not item.success]
    retried = [item for item in metrics.scenario_metrics if item.retry_count]
    approved = [item for item in metrics.scenario_metrics if item.approval_observed]

    rows = "\n".join(
        f"| {m.scenario_id} | {m.expected_route} | {m.actual_route} | "
        f"{'PASS' if m.success else 'FAIL'} | {m.retry_count} | {m.interrupt_count} | "
        f"{m.nodes_visited} | {m.latency_ms} |"
        for m in metrics.scenario_metrics
    )
    state_rows = "\n".join(f"| {name} | {reducer} | {why} |" for name, reducer, why in STATE_FIELDS)
    failures = (
        "\n".join(
            f"- `{m.scenario_id}`: expected `{m.expected_route}`, got `{m.actual_route}`"
            + (f" — errors: {'; '.join(m.errors)}" if m.errors else "")
            for m in failed
        )
        or "- None: every scenario matched its expected route and produced an answer."
    )

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: {STUDENT_NAME} (MSSV {STUDENT_ID})
- Repo/commit: {REPO_URL} @ `{_current_commit()}`
- Date: {date.today().isoformat()}

## 2. Architecture

The graph is an eleven-node `StateGraph` over a `TypedDict` state. Classification decides the
route; every branch converges on `finalize → END`, so no path can hang.

```text
{ARCHITECTURE}
```

`classify_node` and `answer_node` call a real LLM (structured output for classification,
grounded generation for the answer). `evaluate_node` uses a deterministic hard-failure check
plus an LLM-as-judge for soft quality. Nothing branches on scenario ids or exact query text.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
{state_rows}

## 4. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Nodes | Latency (ms) |
|---|---|---|---|---:|---:|---:|---:|
{rows}

- Total scenarios: **{metrics.total_scenarios}**
- Success rate: **{metrics.success_rate:.0%}**
- Avg nodes visited: **{metrics.avg_nodes_visited:.1f}**
- Total retries: **{metrics.total_retries}**
- Total interrupts (approval events): **{metrics.total_interrupts}**
- Checkpoint resume verified: **{metrics.resume_success}**

Scenarios that exercised the retry loop: {", ".join(m.scenario_id for m in retried) or "none"}.
Scenarios that hit the approval gate: {", ".join(m.scenario_id for m in approved) or "none"}.

## 5. Failure analysis

1. **Transient tool failure / unbounded retry.** `tool_node` fails on the error route while
   `attempt < 2`. `evaluate_node` marks the result `needs_retry`, `retry_or_fallback_node`
   increments `attempt`, and `route_after_retry` compares it against `max_attempts`. Without
   that bound the graph would loop until LangGraph's recursion limit; with it, an exhausted
   scenario lands in `dead_letter_node`, which still produces a `final_answer` and an escalation
   error entry instead of an empty response.
2. **Risky action executed without approval.** `risky` never reaches `tool` directly — it must
   pass `risky_action → approval`. A rejection routes to `clarify`, so a rejected refund or
   deletion ends as a question to the customer, never as a side effect. The approval payload is
   part of checkpointed state, so the decision is auditable after the fact.
3. **LLM misclassification.** The classifier is prompted with an explicit priority order
   (risky > tool > missing_info > error > simple) and constrained by structured output, so it
   can only return a known route; unknown values still fall back to `answer` in
   `route_after_classify` rather than dead-ending the graph.

Observed failures this run:

{failures}

## 6. Persistence / recovery evidence

Every scenario runs with its own `thread_id` (`thread-<scenario_id>`) against the configured
checkpointer (`configs/lab.yaml`). After the run the CLI re-reads the last thread's checkpoint
with `graph.get_state()` and counts `graph.get_state_history()`; `resume_success` above is that
readback, not a hardcoded flag. Switching `checkpointer: sqlite` in the config persists
checkpoints to `outputs/checkpoints.sqlite` (WAL mode), which survives process restart.

## 7. Extension work

- LLM-as-judge in `evaluate_node` (with deterministic fallback when the judge is unavailable).
- SQLite checkpointer with WAL, plus a Postgres branch in `persistence.py`.
- Real HITL: `LANGGRAPH_INTERRUPT=true` makes `approval_node` call `interrupt()` and wait for a
  human decision instead of mock-approving.
- State-history readback as machine-checked persistence evidence.

## 8. Improvement plan

With one more day: replace the mock tool with a real API client behind a timeout + circuit
breaker, cache classification per query hash to cut LLM cost, add per-node latency to the event
payloads for a proper trace, and push the approval step into a queue with a real reviewer UI so
`interrupt()` resumes from an external decision rather than a mock.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
