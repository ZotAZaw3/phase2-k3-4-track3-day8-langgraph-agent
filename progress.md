# Progress — Day 08 LangGraph Agent Lab

Status: **all `TODO(student)` sections implemented; 7/7 sample scenarios pass; 25/25 tests pass.**

Environment: Python 3.12, langgraph 1.2.11, langchain-openai (`OPENAI_API_KEY` from `.env`,
default model `gpt-4o-mini`), `langgraph-checkpoint-sqlite` installed during this work.

---

## What was implemented

### `state.py` — state schema
Added the four missing fields, all **overwrite** (no reducer) because only the current value
matters and history is already captured in `events`/`errors`:

| Field | Why it exists |
|---|---|
| `evaluation_result` | gate read by `route_after_evaluate` (`success` / `needs_retry`) |
| `pending_question` | clarification asked back to the user |
| `proposed_action` | the side-effecting action awaiting approval |
| `approval` | HITL decision dict, read by `route_after_approval` |

Append-only fields left as-is: `messages`, `tool_results`, `errors`, `events` — retry history
must survive the loop. `initial_state()` now seeds the new fields so state stays serializable
and complete from the first checkpoint.

### `nodes.py` — all 10 nodes
- `classify_node` — **real LLM**, `.with_structured_output(Classification)` returning
  `route` / `risk_level` / `reason`. Prompt encodes the strict priority order
  risky > tool > missing_info > error > simple. Structured output means the model can only
  return a known route.
- `tool_node` — mock tool. Fails while `route == "error" and attempt < 2`; returns
  `action executed: …` when an approved `proposed_action` exists, otherwise a lookup record.
  Branching is on **route + attempt only** — never on query text or scenario id.
- `evaluate_node` — deterministic hard check (`ERROR` marker / empty) first, then
  **LLM-as-judge** (bonus) for soft quality, wrapped in `try/except` so the judge is never a
  hard dependency of the retry loop.
- `answer_node` — **real LLM**, grounded in `tool_results` + approval decision, told not to
  invent identifiers.
- `ask_clarification_node` — LLM-generated question; distinguishes "too vague" from
  "reviewer rejected the action" and asks accordingly. Sets both `pending_question` and
  `final_answer`.
- `risky_action_node` — LLM one-sentence summary of the side-effecting action for the reviewer.
- `approval_node` — mock-approves by default (CI runs offline); with `LANGGRAPH_INTERRUPT=true`
  calls `langgraph.types.interrupt()` and accepts either a dict or a bare bool on resume.
- `retry_or_fallback_node` — increments `attempt`, appends the failure to `errors`.
- `dead_letter_node` — sets a `final_answer` explaining escalation, so an exhausted run still
  returns something to the user.
- `finalize_node` — audit event carrying route, attempts, answered flag.

### `routing.py` — 4 routing functions
Dict mapping for `route_after_classify` (unknown → `answer`), retry gate, **bounded**
`route_after_retry` (`attempt < max_attempts` → `tool`, else `dead_letter`), approval routing
(rejected → `clarify`).

### `graph.py` — 11 nodes wired
`START → intake → classify →` conditional fan-out; `tool → evaluate →` conditional retry loop;
`risky_action → approval →` conditional; `answer` / `clarify` / `dead_letter → finalize → END`.
Compiled with the injected checkpointer. Verified with `get_graph().draw_mermaid()`.

### `persistence.py` — checkpointers
`none` / `memory` / `sqlite` (WAL, `check_same_thread=False`) / `postgres`. SQLite path defaults
to `outputs/checkpoints.sqlite`, accepts a `sqlite:///` URL.

### `report.py` — `render_report()`
Renders the full `reports/lab_report_template.md` structure from `MetricsReport`: summary
counters, per-scenario table, state-schema table, architecture diagram, failure analysis
(with the actually-observed failures injected), persistence evidence, extensions, improvement
plan. `make run-scenarios` writes it to `reports/lab_report.md`.

### Supporting fixes (not in the TODO list, but the lab could not run without them)
- **Nothing loaded `.env`.** Added `load_dotenv()` in `llm.py` (covers the CLI) and
  `tests/conftest.py` (runs before the smoke tests evaluate their API-key skip marks —
  otherwise every LLM test silently skipped). Added `python-dotenv` to `pyproject.toml`.
- **`latency_ms` was always 0.** `cli.py` now times each `graph.invoke`.
- **`resume_success` was hardcoded `False`.** `cli.py` re-reads the last thread with
  `get_state()` / `get_state_history()` and reports the actual readback.
- **`.gitignore` did not cover the SQLite checkpoint** at the lab's own default path
  (it listed `checkpoints.db*`, the code writes `outputs/checkpoints.sqlite`). Added
  `outputs/*.sqlite*` so a persistence run cannot drag a 76 KB binary into a submission.

---

## Verification — final gate (all green)

```
ruff check src tests           All checks passed!
mypy src                       Success: no issues found in 11 source files
pytest -q                      25 passed  (incl. 6 live-LLM graph smoke tests)
run-scenarios                  7/7 ok  → outputs/metrics.json + reports/lab_report.md
validate-metrics               Metrics valid. success_rate=100.00%
git diff --check               clean (no whitespace errors)
```

The starter shipped with 45 ruff findings and 4 mypy errors. Both gates are now green via real
fixes, not rule suppression: missing annotations added, imports sorted, long lines wrapped, one
unused import removed, `TYPE_CHECKING` imports so `graph.py`/`persistence.py` stay import-safe
while still being typed, `cast()` around `with_structured_output` (LangChain types it as
`dict | BaseModel`), and `types-PyYAML` installed. Test files were touched for annotations and
line wrapping **only** — no assertion, fixture value, config or scenario file was changed to
make a gate pass. The one skip-condition rewrite in `test_graph_smoke.py`
(`not A and not B and not C` → `not any(...)`) is logically identical.

### Sample run (`outputs/metrics.json`)

`total_scenarios=7 · success_rate=1.0 · avg_nodes_visited=6.43 · total_retries=3 ·
total_interrupts=2 · resume_success=true`

| Scenario | Route | Nodes | Retries | Approval | Path walked |
|---|---|---:|---:|---|---|
| S01_simple | simple | 4 | 0 | – | intake→classify→answer→finalize |
| S02_tool | tool | 6 | 0 | – | …→tool→evaluate→answer→finalize |
| S03_missing | missing_info | 4 | 0 | – | …→clarify→finalize |
| S04_risky | risky | 8 | 0 | yes | …→risky_action→approval→tool→evaluate→answer→finalize |
| S05_error | error | 10 | 2 | – | …→retry→tool→evaluate→retry→tool→evaluate→answer→finalize |
| S06_delete | risky | 8 | 0 | yes | same as S04 |
| S07_dead_letter | error | 5 | 1 | – | …→retry→dead_letter→finalize (max_attempts=1) |

Every trail ends at `finalize`. Approval is observed **before** `tool` on both risky scenarios.

### Persistence evidence (cross-process, real)

Process A ran the risky scenario with `checkpointer: sqlite`; a **fresh interpreter** (process B)
opened only the DB file and recovered the thread:

```
checkpoints recovered: 10
recovered route: risky | approval: {'approved': True, 'reviewer': 'mock-reviewer', ...}
recovered answer: The refund has been processed for your account, and a confirmation email …
```

---

## Course corrections during the build

**The LLM judge was too strict (caught on the first full run).** First run: all 7 routes correct,
but risky scenarios burned 3 retries / 15 nodes and S05 needed 3 retries — the judge was asked
"does this satisfy the request?" and correctly answered *no*, because a mock string like
`lookup ok: record found for 'Refund this customer'` does not look like a refund. Both risky
runs and S05 therefore fell through to `dead_letter`; they still scored `success` only because
`dead_letter_node` sets a `final_answer`, which masked the wrong path. Two fixes:
1. `tool_node` now returns `action executed: <approved action> (ref ACT-nnnn)` on the approved
   risky path, so the mock result actually matches the request.
2. The judge's remit was narrowed to "is this usable as grounding?" — `satisfactory=false` only
   for a reported failure, empty result, or a result about a different request.

After the fix the node counts match the target topology exactly (8 / 10 / 5 above).

**The final gate caught a bug I introduced while making mypy green.** Typing `run_config` as
`RunnableConfig` led me to construct it as `RunnableConfig(configurable=...)`, but that symbol is
imported only under `TYPE_CHECKING` — so `run-scenarios` died with
`NameError: name 'RunnableConfig' is not defined` even though lint, typecheck and pytest were all
green (no test covers the CLI). Fixed by going back to a plain dict literal, which the annotation
still types. Worth noting that `validate-metrics` passed *during that failure*, because it was
reading the previous run's `metrics.json`; I deleted `outputs/metrics.json` and
`reports/lab_report.md` and regenerated both from scratch before re-running the gate.

**Rejected-approval path, verified end-to-end** (`LANGGRAPH_INTERRUPT=true`, resumed with
`Command(resume={"approved": False, ...})`):

```
PAUSED at interrupt: True
proposed_action: Delete the customer account following support verification.
nodes: ['intake', 'classify', 'risky_action', 'approval', 'clarify', 'finalize']
approval: {'approved': False, 'reviewer': 'human', 'comment': 'not verified by a human yet'}
tool called? False
pending_question: What specific action would you like us to take regarding the customer account?
```

The rejection never reaches `tool` — the destructive action is not executed.

---

## Not done / deliberate limits

- `configs/grading.yaml` **not run** — hidden scenarios are not in the repo, as instructed.
- `reports/lab_report.md` is generated with the student name/commit fields left blank —
  fill those in before submitting.
- Postgres checkpointer is written but untested (no server here); SQLite is tested.
- No Streamlit UI and no `Send()` fan-out — the completed extensions are LLM-as-judge, SQLite
  persistence with cross-process resume, `interrupt()`-based HITL, and the Mermaid diagram.
