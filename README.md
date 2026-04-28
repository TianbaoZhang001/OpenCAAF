# OpenCAAF: Convergent AI Agent Framework

Reference implementation for the paper **"Harness as an Asset: Enforcing Determinism via the Convergent AI Agent Framework (CAAF)"**.

CAAF transitions LLM agents from probabilistic generators to deterministic, control-theoretic systems for safety-critical engineering domains (autonomous driving, pharmaceutical reactor design, regulated industrial workflows) where 95% correct is undeployable.

---

## The Three Pillars

1. **Recursive Atomic Decomposition (RAD)** — partitions high-entropy requests into a topological DAG. Each node executes in an isolated sub-agent context, preventing Context Rot.
2. **Harness as an Asset (HaaA)** — formalizes domain invariants as YAML rules + Python assertions. The deterministic Unified Assertion Interface (UAI) gives the same compliance floor regardless of which LLM is plugged in.
3. **Structured Semantic Gradients with State Locking** — the Reviewer emits structured directional corrections; satisfied dimensions are frozen across iterations, forcing monotonic non-regression and preventing the stochastic-oscillation trap that traps naive reflection loops.

See `ARCHITECTURE.md` for the design and `paper.pdf` for full empirical evidence.

---

## Headline empirical results

Both benchmarks evaluate the architecturally hardest case: **physically irreconcilable** constraint sets where the correct action is to detect and halt rather than emit a plausible-but-violating answer.

| Benchmark | Monolithic GPT-4o | CAAF-all-mini |
|---|---|---|
| L3 AD Degradation Paradox (Table 1, n=30) | **0%** paradox detection | **100%** |
| Pharma Flow Reactor Paradox (Table 3, n=20) | **0%** | **100%** |

**Cost (Finding 10).** Three configurations all reach 100% on the pharma 7-constraint paradox: CAAF-all-mini at **$0.0044/trial**, CAAF-all-Haiku-4.5 at $0.20/trial, and Mono+UAI true-tool-call on Opus 4 thinking at $0.499/trial. CAAF on commodity models is **~114× cheaper** than the frontier-reasoning + true-tool-call path at identical reliability, and **415× cheaper** with the Cohere Command-R7B executor.

**Open-weight replication (Finding 5).** Both Cohere Command-R7B (7B, structured-output fine-tuned) and Google Gemma-3-12B-IT (12B) achieve 100% on AD PASS-path and pharma 3-way paradox (80/80 trials, $0.061 total) — establishing that CAAF reliability is architectural, not model-scale-dependent, and a fully on-prem deployment is feasible for regulated industries.

**Total reproduction cost.** Every benchmark reported in the paper runs for **under $2.20 USD** in total API spend.

---

## Quick start

```bash
cp .env.example .env          # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
pip install -r requirements.txt
```

Run the headline AD experiment (Table 1, 7 conditions, n=30):

```bash
python -m OpenCAAF.demos.benchmark_full_experiment
```

Each run writes `OpenCAAF/demos/logs/<experiment>_<timestamp>/` containing `results.json` (aggregate metrics), `*_runs.jsonl` (per-trial records), and, for CAAF runs, a `*_traces/run_NN/` subtree of per-iteration strategy plans, expert outputs, UAI feedback, and final artifacts.

Sample size is overridable via `N_TRIALS` (default 20; 30 for `benchmark_full_experiment.py`):

```bash
N_TRIALS=5 python -m OpenCAAF.demos.benchmark_pharma_reactor    # quick smoke
```

---

## Reproducing the paper

| Paper artifact | Script |
|---|---|
| Table 1 (AD 7-condition, n=30, Findings 1–5) | `demos/benchmark_full_experiment.py` |
| Table 2 (open-weight replication) | `demos/benchmark_ad_pass_path.py` and `demos/benchmark_pharma_reactor.py` with `CAAF_BACKEND=openrouter` and `CAAF_EXECUTOR_MODEL=cohere/command-r7b-12-2024` (or `google/gemma-3-12b-it`) |
| Table 3 cells 1–4 (Pharma GPT-4o-mini, F6, F7, F8-i) | `demos/benchmark_pharma_reactor.py` |
| Table 3 cells 6–7 (Mono+UAI true tool-call, F8 core) | `demos/reasoning_uai_react.py pharma --n 20` |
| Table 3 cell 5 (Mono Opus 4 thinking, no UAI, F9) | `demos/smoke_v2.py B` |
| Table 3 cell 8 (CAAF-all-Haiku-4.5) | `demos/smoke_v2.py C` |
| Table 4 (AD PASS-path convergence) | `demos/benchmark_ad_pass_path.py` |
| Table on baselines (debate + sequential, F11) | `demos/benchmark_debate_baseline.py` + `demos/benchmark_sequential_baseline.py` |
| Context Rot benchmark | `demos/benchmark_context_rot_v2.py` |
| Fig 2 (oscillation) | `demos/benchmark_oscillation_v2.py` |
| Cost / cross-table rollup | `demos/aggregate_v2_results.py` |
| Regenerate paper figures | `demos/generate_paper_figures_v2.py` |

A reviewer-friendly workflow with timing and expected cost is documented in `demos/README.md`.

---

## Repository layout

- `engine/` — Orchestrator, RAD, Semantic Reviewer, Context Resolver
- `harness/` — Harness Registry (YAML + Python assertion engine)
  - `harness/data/ad_degradation.yaml` — L3 AD paradox variant
  - `harness/data/ad_degradation_pass.yaml` — L3 AD PASS-path variant
  - `harness/data/pharma_flow_reactor.yaml` — Pharma paradox variant
  - `harness/data/pharma_flow_reactor_pass.yaml` — Pharma PASS-path variant
- `adapters/` — LLM provider adapters (OpenAI / Anthropic / OpenRouter)
- `schemas/` — JSON schemas (AtomicTask, DecompositionTree, gradient triple, state locking)
- `demos/` — One benchmark script per paper experiment cell
- `utils/` — Logging, cost accounting, parse-retry policy

---

## Model access

| Condition | Provider | Env var |
|---|---|---|
| GPT-4o / GPT-4o-mini monolithic + CAAF reference | OpenAI | `OPENAI_API_KEY` |
| Claude Opus 4 thinking / Haiku 4.5 monolithic + tool-call | Anthropic | `ANTHROPIC_API_KEY` |
| Cohere Command-R7B / Google Gemma-3-12B-IT open-weight | OpenRouter | `OPENROUTER_API_KEY` |

---

## Minimal usage example

```python
from OpenCAAF.engine.orchestrator import OpenCAAFOrchestrator
from OpenCAAF.adapters.openai_adapter import OpenAIAdapter

executor = OpenAIAdapter(model="gpt-4o-mini")
reviewer = OpenAIAdapter(model="gpt-4o-mini")

orchestrator = OpenCAAFOrchestrator(executor, reviewer)

tree = orchestrator.run_full_pipeline(
    request="Design an L3 autonomous driving fallback for highway cruise at 120 km/h with 30m perception range.",
    domain_id="ad_degradation",   # loads harness/data/ad_degradation.yaml
)
```

---

## License

Apache-2.0.
