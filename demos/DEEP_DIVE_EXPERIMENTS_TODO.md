# CAAF Deep Dive Experiments: Action Plan & TODOs

This document outlines the systematic implementation plan for the three core ablation studies designed to empirically validate the Convergent AI Agent Framework (CAAF) against Monolithic Large Language Models.

All experiments will be centered around the L3 Autonomous Driving "Kessler Syndrome" degradation scenario to provide a deep, rigorous vertical analysis.

---

## Experiment 1: The "Context Rot" Scalability Test (Robustness to Complexity)

**Hypothesis:** Monolithic LLMs suffer from "Context Rot" (attention degradation) as the number of interconnected engineering constraints scales. As the context window fills with complex, competing sub-systems, the LLM prioritizes satisfying linguistic constraints over maintaining strict physical invariants. CAAF maintains 100% interception due to its $O(1)$ deterministic UAI assertion engine, regardless of context size.

- [x] **1.1 Create Sub-system Generator Script:** 
  - Write a utility to dynamically inject $N$ (e.g., 5, 20, 50) **realistic, coupled sub-system constraints** into the AD degradation prompt and the Harness YAML. Instead of trivial noise (like wiper speed), these will be complex interdependent rules (e.g., thermal management limits for the radar, battery discharge rate caps during heavy braking, network bandwidth throttling for V2X telemetry, redundant power steering current limits).
- [x] **1.2 Develop Benchmark Runner (`benchmark_context_rot.py`):**
  - Run the Monolithic GPT-4o baseline against prompts of increasing sub-system complexity.
  - Run the CAAF orchestrator against the same prompts.
  - Record the pass/fail rate specifically for the critical physical paradoxes (Stopping Distance vs. Perception Range and Deceleration Limits).
- [x] **1.3 Data Extraction & Visualization:**
  - Generate a JSON dataset tracking `Coupled Sub-systems Count` vs. `Critical Failure Rate`.
  - (Optional) Plot a line graph showing the monolithic model's safety compliance decaying as system complexity $N$ increases, contrasting with CAAF's flat 100% interception line.

---

## Experiment 2: Stochastic Oscillation vs. Monotonic Convergence (State Locking)

**Hypothesis:** Without architectural "State Locking," multi-agent reflection loops fall into unbounded "Stochastic Oscillation" when resolving coupled physical paradoxes (fixing A breaks B). CAAF's topological scoping guarantees monotonic convergence to a mathematical boundary.

- [x] **2.1 Implement Naive Reflection Baseline (`case_01_naive_reflection.py`):**
  - Create a loop where an LLM generates the JSON parameters, the UAI evaluates them, and the raw text error (e.g., "Stopping distance 70m > 30m perception") is fed back to the LLM to "try again" *without* locking any previously correct states.
- [x] **2.2 Execute Parallel Traces:**
  - Run the Naive baseline for 10 iterations on the Kessler paradox.
  - Run the CAAF Interactive Mode (with dynamic relaxation) to resolve the same paradox.
- [x] **2.3 Data Extraction & Visualization:**
  - Extract the specific parameter values (`vehicle_speed_at_t5`) and the UAI passing scores across iterations.
  - Generate a JSON dataset showing the oscillation of the naive model versus the step-wise monotonic convergence of CAAF.

---

## Experiment 3: Compute-for-Risk Arbitrage (Economic Viability)

**Hypothesis:** Trading inexpensive LLM inference tokens (compute) for automated physical verification (risk mitigation) drastically reduces the Total Cost of Ownership (TCO) compared to relying on Human-in-the-Loop (HITL) debugging for "compliant hallucinations."

- [x] **3.1 Aggregation of Empirical Token Usage:**
  - Parse the existing `trace.json` files from the successful CAAF convergence runs (from Exp 2).
  - Calculate the exact API cost based on August 2024 OpenAI pricing (Prompt tokens vs. Completion tokens).
- [x] **3.2 TCO Modeling Script (`generate_economics_data.py`):**
  - Model the monolithic approach: Cost of 1 API call + Probability of Failure $\times$ \$50 (expert debugging time).
  - Model the CAAF approach: Cost of $N$ API calls across the DAG + \$0 debugging cost (since the artifact is mathematically proven safe upon convergence).
- [x] **3.3 Final Synthesis:**
  - Output a conclusive economic summary table/JSON proving that CAAF flattens the exponential cost curve of complex system engineering.

---

*Status: Plan Approved. Awaiting execution of Experiment 1.*
