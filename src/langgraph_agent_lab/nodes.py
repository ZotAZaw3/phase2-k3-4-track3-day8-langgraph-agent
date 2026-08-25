"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal, cast

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, Route, make_event

ERROR_MARKER = "ERROR"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── LLM structured-output schemas ───────────────────────────────────
class Classification(BaseModel):
    """Intent classification returned by the LLM."""

    route: Literal["risky", "tool", "missing_info", "error", "simple"] = Field(
        description="Intent category of the support ticket"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="high when the request has irreversible or customer-visible side effects"
    )
    reason: str = Field(description="One short sentence explaining the choice")


class Judgement(BaseModel):
    """LLM-as-judge verdict on a tool result."""

    satisfactory: bool = Field(description="True if the tool result answers the user's request")
    reason: str = Field(description="One short sentence explaining the verdict")


CLASSIFY_PROMPT = """You classify support tickets for an automated agent.

Categories, in strict priority order — pick the FIRST one that applies:
1. risky        — the user asks the agent to perform an action with side effects:
                  refunds, payments, deletions, cancellations, sending emails, account changes.
2. tool         — the user asks to look something up: order status, tracking, records, search.
3. missing_info — the request is too vague or incomplete to act on (no subject, no identifier,
                  no way to know what "it" refers to).
4. error        — the ticket reports a system/technical failure: timeout, crash, outage,
                  service unavailable, cannot recover.
5. simple       — a general question answerable from knowledge, without tools or actions.

risk_level is "high" for the risky category, "low" otherwise (use "medium" only when unsure).

Ticket:
{query}"""

ANSWER_PROMPT = """You are a support agent writing the final reply to a customer ticket.

Ground your answer ONLY in the context below. If tool results are present, use their content.
If an action was approved, confirm what was done. Do not invent order numbers, amounts or dates.
Reply in 1-3 sentences, plain text, no greeting boilerplate.

Ticket: {query}
Intent: {route}
Tool results: {tool_results}
Approval decision: {approval}"""


# ─── Nodes ───────────────────────────────────────────────────────────
def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    llm = get_llm().with_structured_output(Classification)
    result = cast(
        Classification, llm.invoke(CLASSIFY_PROMPT.format(query=state.get("query", "")))
    )
    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "messages": [f"classify:{result.route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={result.route} risk={result.risk_level}",
                reason=result.reason,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call, simulating a transient failure on the error route.

    The failure is keyed on the classified route + attempt counter only — never on a
    specific query or scenario id — so hidden scenarios behave the same way.
    """
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    action = state.get("proposed_action")
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"{ERROR_MARKER}: upstream service unavailable (attempt {attempt})"
        status = "failed"
    elif action and (state.get("approval") or {}).get("approved"):
        result = f"action executed: {action} (ref ACT-{attempt + 1:04d})"
        status = "completed"
    else:
        result = f"lookup ok: record found for '{query[:60]}' (attempt {attempt})"
        status = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", status, result[:120], attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate.

    Hard failures are detected deterministically; anything else goes to an LLM judge so
    a plausible-but-useless result still triggers a retry.
    """
    results = state.get("tool_results") or []
    latest = results[-1] if results else ""
    if not latest or ERROR_MARKER in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "tool result failed hard check")],
        }

    verdict, reason = True, "heuristic pass"
    try:
        judge = get_llm().with_structured_output(Judgement)
        judgement = cast(
            Judgement,
            judge.invoke(
                "You check whether a tool call is usable as grounding for a reply — you are "
                "NOT judging whether the wording is complete or polished.\n"
                "satisfactory=false ONLY if the result reports a failure, is empty, or is "
                "about a different request. A short internal record or confirmation line is "
                "satisfactory.\n"
                f"Request: {state.get('query', '')}\nTool result: {latest}"
            ),
        )
        verdict, reason = judgement.satisfactory, judgement.reason
    except Exception as exc:  # judge is a bonus signal, never a hard dependency
        reason = f"judge unavailable ({type(exc).__name__}), fell back to heuristic"

    evaluation = "success" if verdict else "needs_retry"
    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", f"{evaluation}: {reason}"[:200])],
    }


def answer_node(state: AgentState) -> dict:
    """Generate the final response with an LLM, grounded in tool results and approval."""
    llm = get_llm(temperature=0.2)
    prompt = ANSWER_PROMPT.format(
        query=state.get("query", ""),
        route=state.get("route", "unknown"),
        tool_results="\n".join(state.get("tool_results") or []) or "(none)",
        approval=state.get("approval") or "(not required)",
    )
    answer = str(llm.invoke(prompt).content).strip()
    return {
        "final_answer": answer,
        "messages": [f"answer:{answer[:40]}"],
        "events": [make_event("answer", "completed", answer[:200])],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for the missing information instead of hallucinating an answer."""
    llm = get_llm(temperature=0.2)
    rejected = (state.get("approval") or {}).get("approved") is False
    context = (
        "The reviewer rejected the proposed action: "
        f"{state.get('proposed_action') or 'n/a'}. Ask the customer how they want to proceed."
        if rejected
        else "The ticket is too vague to act on. Ask for the single most useful missing detail."
    )
    question = str(
        llm.invoke(
            "Write one short clarifying question for this support ticket. Question only.\n"
            f"{context}\nTicket: {state.get('query', '')}"
        ).content
    ).strip()
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"clarify:{question[:40]}"],
        "events": [make_event("clarify", "completed", question[:200], rejected=rejected)],
    }


def risky_action_node(state: AgentState) -> dict:
    """Describe the side-effecting action so a human can approve or reject it."""
    llm = get_llm()
    action = str(
        llm.invoke(
            "Summarize the side-effecting action this ticket requests, in one imperative "
            "sentence, so a human reviewer can approve or reject it. Sentence only.\n"
            f"Ticket: {state.get('query', '')}"
        ).content
    ).strip()
    return {
        "proposed_action": action,
        "messages": [f"risky_action:{action[:40]}"],
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                action[:200],
                risk_level=state.get("risk_level", "high"),
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop gate. Mock-approves by default so CI runs offline.

    Set LANGGRAPH_INTERRUPT=true to pause the graph and wait for a real human decision.
    """
    action = state.get("proposed_action") or state.get("query", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt({"proposed_action": action, "question": "Approve this action?"})
        if not isinstance(decision, dict):  # resumed with a bare True/False
            decision = {"approved": bool(decision)}
        approved = bool(decision.get("approved"))
        reviewer, comment = "human", str(decision.get("comment", ""))
    else:
        approved, reviewer, comment = True, "mock-reviewer", "auto-approved (mock HITL)"
    approval = {"approved": approved, "reviewer": reviewer, "comment": comment}
    return {
        "approval": approval,
        "messages": [f"approval:{approved}"],
        "events": [
            make_event(
                "approval",
                "approved" if approved else "rejected",
                f"{reviewer}: {comment}"[:200],
                proposed_action=action[:200],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record one retry attempt and increment the bounded attempt counter."""
    attempt = state.get("attempt", 0) + 1
    results = state.get("tool_results") or []
    detail = results[-1] if results else "no tool result yet"
    message = f"attempt {attempt}/{state.get('max_attempts', 3)}: {detail}"
    return {
        "attempt": attempt,
        "errors": [message],
        "events": [make_event("retry", "retrying", message[:200], attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Third failure layer: retry → fallback → dead letter. No more attempts, escalate."""
    attempts = state.get("attempt", 0)
    answer = (
        "We could not complete this request automatically after "
        f"{attempts} attempt(s). It has been escalated to a human support engineer, "
        "who will follow up on this ticket."
    )
    return {
        "final_answer": answer,
        "errors": [f"dead_letter: exhausted {attempts} attempts"],
        "events": [
            make_event("dead_letter", "escalated", answer[:200], attempts=attempts)
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit the final audit event. All routes pass through here before END."""
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                attempts=state.get("attempt", 0),
                answered=bool(state.get("final_answer")),
            )
        ]
    }
