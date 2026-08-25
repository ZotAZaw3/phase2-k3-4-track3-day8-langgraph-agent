# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date: 2026-08-25

## 2. Architecture

The graph is an eleven-node `StateGraph` over a `TypedDict` state. Classification decides the
route; every branch converges on `finalize → END`, so no path can hang.

```text
START → intake → classify → [route_after_classify]
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
  error        → retry → [route_after_retry] → ...
```

`classify_node` and `answer_node` call a real LLM (structured output for classification,
grounded generation for the answer). `evaluate_node` uses a deterministic hard-failure check
plus an LLM-as-judge for soft quality. Nothing branches on scenario ids or exact query text.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| route | overwrite | only the current classification matters |
| risk_level | overwrite | current risk assessment |
| attempt | overwrite | counter, bounded against max_attempts |
| evaluation_result | overwrite | gate read by route_after_evaluate |
| pending_question | overwrite | latest clarification asked back to the user |
| proposed_action | overwrite | single action awaiting approval |
| approval | overwrite | latest HITL decision, read by route_after_approval |
| final_answer | overwrite | one answer per run |
| messages | append | conversation trace |
| tool_results | append | every tool call kept for grounding + evaluation |
| errors | append | failure history must not be lost across retries |
| events | append | audit trail; nodes_visited and retry counts are derived from it |

## 4. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Nodes | Latency (ms) |
|---|---|---|---|---:|---:|---:|---:|
| S01_simple | simple | simple | PASS | 0 | 0 | 4 | 4949 |
| S02_tool | tool | tool | PASS | 0 | 0 | 6 | 2949 |
| S03_missing | missing_info | missing_info | PASS | 0 | 0 | 4 | 1956 |
| S04_risky | risky | risky | PASS | 0 | 1 | 8 | 3865 |
| S05_error | error | error | PASS | 2 | 0 | 10 | 2579 |
| S06_delete | risky | risky | PASS | 0 | 1 | 8 | 4251 |
| S07_dead_letter | error | error | PASS | 1 | 0 | 5 | 969 |

- Total scenarios: **7**
- Success rate: **100%**
- Avg nodes visited: **6.4**
- Total retries: **3**
- Total interrupts (approval events): **2**
- Checkpoint resume verified: **True**

Scenarios that exercised the retry loop: S05_error, S07_dead_letter.
Scenarios that hit the approval gate: S04_risky, S06_delete.

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

- None: every scenario matched its expected route and produced an answer.

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
