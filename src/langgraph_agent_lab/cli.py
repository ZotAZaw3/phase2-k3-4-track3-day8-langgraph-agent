"""CLI for the lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

app = typer.Typer(no_args_is_help=True)


def _checkpoint_readback(graph: CompiledStateGraph, run_config: RunnableConfig | None) -> bool:
    """Re-read the last thread from the checkpointer: proof the run is resumable."""
    if not run_config:
        return False
    try:
        snapshot = graph.get_state(run_config)
        history = list(graph.get_state_history(run_config))
    except Exception as exc:  # no checkpointer configured, or backend unavailable
        typer.echo(f"checkpoint readback unavailable: {exc}")
        return False
    resumable = bool(snapshot.values.get("final_answer") or snapshot.values.get("pending_question"))
    typer.echo(
        f"checkpoint readback: thread={(run_config.get('configurable') or {}).get('thread_id')} "
        f"checkpoints={len(history)} resumable={resumable}"
    )
    return resumable and len(history) > 1


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    run_config: RunnableConfig | None = None
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        started = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        metric = metric_from_state(
            final_state, scenario.expected_route.value, scenario.requires_approval
        )
        metric.latency_ms = int((time.perf_counter() - started) * 1000)
        metrics.append(metric)
        status = "ok" if metric.success else "FAIL"
        typer.echo(f"{metric.scenario_id}: {metric.actual_route} {status}")
    report = summarize_metrics(metrics)
    report.resume_success = _checkpoint_readback(graph, run_config)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("export-trace")
def export_trace(
    thread_id: Annotated[str, typer.Option("--thread-id")],
    output: Annotated[Path, typer.Option("--output")],
    checkpointer: Annotated[str, typer.Option("--checkpointer")] = "sqlite",
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
) -> None:
    """Export the checkpointed state history of one thread as a readable trace."""
    graph = build_graph(checkpointer=build_checkpointer(checkpointer, database_url))
    run_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshots = list(graph.get_state_history(run_config))
    if not snapshots:
        raise typer.BadParameter(f"No checkpoints found for thread {thread_id}")

    steps = []
    for index, snapshot in enumerate(reversed(snapshots)):  # oldest checkpoint first
        values = snapshot.values
        answer = values.get("final_answer")
        steps.append(
            {
                "step": index,
                "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                "next_nodes": list(snapshot.next),
                "nodes_so_far": [event.get("node") for event in values.get("events", [])],
                "route": values.get("route"),
                "attempt": values.get("attempt"),
                "evaluation_result": values.get("evaluation_result"),
                "approval": values.get("approval"),
                "tool_results": values.get("tool_results", []),
                "final_answer": (answer[:160] if answer else None),
            }
        )

    trace = {"thread_id": thread_id, "checkpoint_count": len(snapshots), "steps": steps}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Wrote {len(snapshots)} checkpoints for {thread_id} to {output}")


if __name__ == "__main__":
    app()
