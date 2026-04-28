# CAAF Demos — One Script per Paper Experiment

Every script in this directory reproduces a specific table, figure, or finding
from the paper. Scripts are organized by benchmark domain; each writes a
timestamped directory under `demos/logs/` with aggregate metrics
(`results.json`), per-trial records (`*_runs.jsonl`), and, for CAAF runs, a
`*_traces/run_NN/` tree of per-iteration strategy plans, expert outputs, UAI
feedback, and final artifacts.

## Setup

```bash
cp ../.env.example ../.env          # fill in OPENAI / ANTHROPIC / OPENROUTER keys
pip install -r ../requirements.txt
```

## L3 Autonomous Driving benchmark

| Script                                      | What it reproduces |
|---------------------------------------------|--------------------|
| `benchmark_full_experiment.py`              | Table 1 (7-condition AD batch, $n{=}30$); Findings F1--F5 |
| `benchmark_ad_pass_path.py`                 | AD PASS-path convergence on the satisfiable variant (perception 30m → 90m); Appendix F |
| `benchmark_context_rot_v2.py`               | Context Rot variant (clean-only + noise-injected); Appendix G |
| `benchmark_oscillation_v2.py`               | Fig 2 — naive reflection vs CAAF stochastic oscillation ($n{=}20$ × 5 iters) |

## Pharmaceutical flow-reactor benchmark

| Script                                      | What it reproduces |
|---------------------------------------------|--------------------|
| `benchmark_pharma_reactor.py`               | Table 3 cells 1--4 (GPT-4o-mini family) — F6, F7, and F8-(i) prompt-simulated UAI |
| `reasoning_uai_react.py`                    | Table 3 cells 6--7 (Mono+UAI **true tool-call** on Opus 4 thinking and Haiku 4.5); core evidence for F8 |
| `smoke_v2.py`                               | Table 3 cell 5 (Mono Opus 4 thinking, no UAI, F9 dual-attractor) and cell 8 (CAAF-all-Haiku-4.5) |
| `benchmark_pharma_pass_path.py`             | Companion PASS-path variant ($\tau_{\max}{=}180$s); optional harder-setting benchmark, not reported in main tables |

## Multi-agent baselines

| Script                                      | What it reproduces |
|---------------------------------------------|--------------------|
| `benchmark_debate_baseline.py`              | Multi-agent debate baseline (2 agents × 3 rounds) on AD + Pharma; Appendix I |
| `benchmark_sequential_baseline.py`          | Sequential-checker baseline (primary + LLM checker + 3 retries) on AD + Pharma; Appendix I |

## Reproducibility helpers

| Script                                      | Purpose |
|---------------------------------------------|---------|
| `aggregate_v2_results.py`                   | Scans `demos/logs/` and emits a unified markdown results matrix and per-trial cost roll-up |
| `generate_paper_figures_v2.py`              | Regenerates paper figures from logged results |

## Typical reviewer workflow

```bash
# 1. Two core benchmark batches (~30 min, ~$1.50 total)
python -m demos.benchmark_full_experiment          # AD, Table 1
python -m demos.benchmark_pharma_reactor           # Pharma cells 1--4

# 2. Frontier-reasoning cells (~15 min, ~$0.50)
python -m demos.reasoning_uai_react pharma --n 20  # Pharma cells 6--7
python -m demos.smoke_v2 B                         # Pharma cell 5
python -m demos.smoke_v2 C                         # Pharma cell 8 (CAAF-Haiku)

# 3. Supporting studies
python -m demos.benchmark_oscillation_v2           # Fig 2
python -m demos.benchmark_context_rot_v2           # Appendix G
python -m demos.benchmark_debate_baseline          # Appendix I
python -m demos.benchmark_sequential_baseline      # Appendix I
python -m demos.benchmark_ad_pass_path             # Appendix F PASS-path

# 4. Roll up and regenerate figures
python -m demos.aggregate_v2_results
python -m demos.generate_paper_figures_v2
```

Total API cost of the full reproduction is **under \$2.20 USD** (see the
paper's Reproducibility appendix for the per-experiment breakdown).

## Model access

| Condition | Provider | Env var |
|-----------|----------|---------|
| GPT-4o / GPT-4o-mini monolithic + CAAF reference          | OpenAI     | `OPENAI_API_KEY`     |
| Claude Opus 4 thinking / Haiku 4.5 monolithic + tool-call | Anthropic  | `ANTHROPIC_API_KEY`  |
| Cohere Command-R7B / Google Gemma-3-12B-IT open-weight    | OpenRouter | `OPENROUTER_API_KEY` |

## Selecting trial count

Every script honors the `N_TRIALS` environment variable (default `20`,
except `benchmark_full_experiment.py` which defaults to `30`):

```bash
N_TRIALS=5 python -m demos.benchmark_pharma_reactor    # quick smoke
```

## What each log directory contains

```
demos/logs/<experiment>_<timestamp>/
├── results.json                  # aggregate metrics (Correct%, cost, timings)
├── <cond>_runs.jsonl             # one JSON line per trial
└── <cond>_traces/run_NN/         # CAAF-only: per-iteration intermediate state
    ├── decomposition.json
    ├── executor_<node>.json
    ├── uai_<node>.json
    ├── reviewer_<node>.json
    └── final_artifact.json
```
