# Self-Assessment — Day 08 LangGraph Agent Lab

- Name: Bùi Minh Long (MSSV 2A202601462)
- Repo: https://github.com/ZotAZaw3/phase2-k3-4-track3-day8-langgraph-agent
- Scored against `docs/RUBRIC.md`. Every claim below points at a file or an artifact in
  this repo, so a grader can check it without running anything.

## Score by category

| Category | Max | Self-score | Evidence |
|---|---:|---:|---|
| Architecture & state schema | 15 | 14 | `state.py`: 4 added fields, overwrite vs append-only justified in report §3 |
| Graph construction & wiring | 15 | 15 | `graph.py`: 11 nodes, 4 conditional edges, Mermaid diagram in report §2 |
| LLM integration | 15 | 14 | `classify_node` structured output, `answer_node` grounded, `evaluate_node` LLM-as-judge |
| Graph behavior | 20 | 19 | 7/7 routes correct, bounded retry, HITL, all trails end at `finalize` |
| Persistence & recovery | 10 | 10 | Cross-process SQLite resume + `outputs/trace_history.json`, `outputs/trace_retry.json` |
| Metrics & tests | 15 | 14 | `outputs/metrics.json` valid, 25 tests pass, ruff + mypy clean |
| Report & demo | 10 | 9 | `reports/lab_report.md` §2–§8 with failure analysis and evidence |
| **Total** | **100** | **95** | |

## Why not full marks

- **Architecture (−1):** `pending_question` and `final_answer` both carry the clarification
  text. It keeps `metric_from_state` happy without special-casing, but it is duplicated state;
  a cleaner design would derive one from the other at the boundary.
- **LLM integration (−1):** `risky_action_node` and `ask_clarification_node` each make their own
  LLM call. Folding those into the `classify_node` structured output would cut two calls per
  risky ticket — correctness is fine, cost is not optimal.
- **Graph behavior (−1):** verified against the 7 sample scenarios only. Classification is an
  LLM judgment, so a genuinely ambiguous hidden ticket could route differently. The
  priority-ordered prompt, the constrained enum, and the `answer` fallback for unknown routes
  are the defenses; none of them is a proof.
- **Metrics & tests (−1):** no test covers `cli.py`. That gap is not theoretical — a `NameError`
  in the CLI survived a green lint + mypy + pytest run and was only caught by executing
  `run-scenarios` (written up in `progress.md`).
- **Report (−1):** the failure analysis is design-level. It has no screenshot of a real
  production-style incident, because the tool is a mock.

## Deliverables checklist

| Deliverable | Where |
|---|---|
| Core tests pass | `pytest -q` → 25 passed (6 LLM tests skip without a key) |
| 6+ scenarios run | `outputs/metrics.json` — 7 scenarios, 100% success |
| Trace JSON / state history | `outputs/trace_history.json` (HITL), `outputs/trace_retry.json` (retry loop) |
| metrics.json + report.md | `outputs/metrics.json`, `reports/lab_report.md` |
| Self-assessment | this file |
| Final repo | pushed to `main` |

## Demo script (2 minutes)

1. **One route — risky.** `classify_node` returns `risky` via structured output (priority
   risky > tool > missing_info > error > simple) → `risky_action` → `approval` → the tool runs
   **only** after approval → `evaluate` → `answer`. Show `outputs/trace_history.json`: at step 5
   the next node is `tool` and `approval` is already in state.
2. **One failure mode — unbounded retry.** `route_after_retry` compares `attempt` against
   `max_attempts`. S07 sets `max_attempts=1`, exhausts immediately, and lands in `dead_letter`,
   which still returns a `final_answer` instead of nothing. Show `outputs/trace_retry.json` for
   the bounded loop: `retry → tool → evaluate → retry → tool → evaluate → answer`.
3. **Rejection path.** `LANGGRAPH_INTERRUPT=true`, resume with `approved: False` → the trail is
   `risky_action → approval → clarify → finalize` and `tool_results` stays empty: the deletion
   never executes (transcript in `reports/lab_report.md` §7).

## Known limitations

- The tool is a mock; failures are simulated from route + attempt, never from query text.
- Postgres checkpointer is written but untested — no server available.
- Actions is disabled on this fork, so CI has no runs. Verified locally that it would pass:
  `ruff check src tests` clean, and with API keys unset `pytest -q` gives 19 passed / 6 skipped,
  which is exactly what the CI job does.
