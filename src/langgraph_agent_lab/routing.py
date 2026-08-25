"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState, Route

ROUTE_TO_NODE = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node. Unknown routes fall back to a direct answer."""
    return ROUTE_TO_NODE.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Retry-loop gate: unsatisfactory tool result goes back through retry, else answer."""
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
    """Bounded retry: attempt is incremented by retry_or_fallback_node before this runs."""
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Approved risky actions execute the tool; rejected ones go back to the user."""
    approval = state.get("approval") or {}
    return "tool" if approval.get("approved") else "clarify"
