# OpenCAAF Empirical Demonstrations

This directory contains the Minimum Viable Instances (MVIs) used to empirically validate the core pillars of the **Convergent AI Agent Framework (CAAF)**. These experiments are designed to reproduce the failure modes of contemporary Large Language Models (LLMs) in safety-critical industrial engineering and demonstrate how CAAF's cybernetic control loops enforce fail-safe determinism.

---

## 1. End-to-End Lifecycle (`demo_l3_ad_kessler_lifecycle.py`)

**Paper Reference:** 
- Section 4: Empirical Evaluation: The Kessler Syndrome
- Pillar 2: Harness as an Asset (HaaA) / UAI Hard Interception

**What it does:**
This script simulates the "Kessler Syndrome" scenario: an L3 autonomous vehicle cruising at 120km/h suddenly loses primary LiDAR in extreme rain, capping perception at 30m. The system is tasked with designing a degradation state machine that satisfies both rear-collision prevention (max deceleration) and forward-collision prevention (stopping distance).

**How it proves the paper's claims:**
1. **Monolithic Failure ("Compliant Hallucination"):** The control group (GPT-4o zero-shot) successfully outputs a perfectly structured JSON, but it hallucinated physical parameters that violate the laws of physics (e.g., claiming a stopping distance of 82m is safe within a 30m perception range), leading to a catastrophic "blind collision".
2. **CAAF Deterministic Interception:** OpenCAAF uses the Unified Assertion Interface (UAI) to evaluate the generated artifact against the YAML Harness. It catches the physics paradox and throws a `[HARD FAILURE]`.
3. **Strategic Negotiation:** Instead of blindly guessing a fix, CAAF identifies the deadlock, pauses execution, and presents the human operator with a Strategic Resolution Menu to negotiate business trade-offs (e.g., relax passenger comfort/deceleration to ensure safety).
4. **Dynamic Override & Convergence:** Once the human authorizes a relaxation, CAAF modifies the Harness on the fly and monotonically converges on a mathematically safe state.

---

## 2. Combating Information Noise (`benchmark_context_rot.py`)

**Paper Reference:**
- Section 5: Empirical Deep Dive I: Context Rot and Semantic Compromise
- Pillar 1: Recursive Atomic Decomposition (RAD) & Topological Scoping

**What it does:**
This benchmark stresses the Orchestrator with a highly complex Product Requirements Document (PRD) injected with "Information Noise" (e.g., irrelevant constraints about VIP passenger sleep mode jerk limits and 5G video bandwidth). 

**How it proves the paper's claims:**
1. **Monolithic "Die by Compliance":** When fed the complex PRD, the monolithic baseline suffers from *Context Rot*. The LLM's attention mechanism gets overwhelmed by the noisy semantic constraints (e.g., ensuring a smooth ride for the VIP). In striving to satisfy the semantic text, it "silently forgets" the core physical constraint (braking distance), resulting in a smooth, comfortable, but ultimately fatal crash.
2. **CAAF's Context Firewall:** CAAF utilizes RAD to physically decouple the context. The node responsible for calculating the kinematics is *only* given kinematics data. By stripping away the VIP noise, the isolated Executor operates in a pristine cognitive environment. When assembled, the UAI strictly enforces the physics, proving that RAD acts as a structural defense against semantic compromise.

---

## 3. Ending Infinite Loops (`benchmark_oscillation.py`)

**Paper Reference:**
- Section 6: Empirical Deep Dive II: Stochastic Oscillation vs. Monotonic Convergence
- Pillar 3: Structured Semantic Gradients & Monotonic Convergence

**What it does:**
This benchmark places the LLMs into an unresolvable physical dilemma where the forward safety constraint and rear safety constraint are mutually exclusive based on the initial prompt constraints.

**How it proves the paper's claims:**
1. **Stochastic Oscillation (The Seesaw Effect):** A standard "AutoGPT-style" multi-agent or reflection loop relies on *Blind Gradients* (e.g., feeding the text "you failed the forward collision check" back to the LLM). The LLM drops the speed, which then fails the rear collision check. Told to fix that, it raises the speed back up. The LLM gets trapped in an infinite, oscillating loop, wasting compute without ever finding a solution.
2. **Monotonic Convergence via UAI Exact Boundaries:** CAAF detects the deadlock and uses the UAI's mathematical solver to calculate the *Exact Boundary* required to bridge the paradox (the Structured Semantic Gradient). By projecting this precise mathematical magnitude into the LLM's semantic space, CAAF skips the heuristic trial-and-error phase. It acts like a second-order optimization (Newton's Method), pointing the system directly to the required boundary, achieving mathematically verified convergence in a single step without oscillation.