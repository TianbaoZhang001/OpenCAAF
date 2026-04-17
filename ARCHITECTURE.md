# OpenCAAF: Core Architecture

OpenCAAF transitions AI agents from "chat-based interfaces" to "deterministic control components." This document mirrors the architecture described in the CAAF paper (Sections 3.1–3.5) and is the entry point for understanding the codebase.

---

## 1. Three Pillars

### Pillar 1: Recursive Atomic Decomposition (RAD)
- **Problem**: Long contexts cause attention decay and "Context Rot" — safety constraints introduced early get deprioritized in long prompts.
- **Solution**: Break complex tasks into a topological **DAG of atomic nodes**. Each node is executed in an isolated context scope by a dedicated Sub-Agent.
- **Mechanism**: The Orchestrator prunes irrelevant global state, injecting only the variables required for the current node. Cross-node conflicts are surfaced deterministically at integration time, not within any single Executor's reasoning loop.
- **Code**: `engine/orchestrator.py`, `engine/rad.py`, `engine/context_resolver.py`

### Pillar 2: Harness as an Asset (HaaA)
- **Problem**: Embedding safety logic in prompts couples it to a specific LLM and a specific phrasing — neither survives model upgrades or paraphrased requests.
- **Solution**: Formalize domain constraints as **Harness Assets** (YAML rules + Python assertions) enforced by the **Unified Assertion Interface (UAI)**. The Harness is the proprietary "compliance moat"; the LLM is replaceable.
- **Mechanism**: At UAI evaluation time, the Harness produces a deterministic PASS/FAIL with the exact mathematical boundary required for compliance — not a qualitative gradient.
- **Code**: `harness/engine.py`, `harness/data/*.yaml`

### Pillar 3: Structured Semantic Gradients + State Locking
- **Problem**: Naive reflection ("try again") leads to **stochastic oscillation** — fixing constraint A while breaking constraint B.
- **Solution**: The Reviewer functions as a **semantic sensor**, converting deterministic UAI failures into **structured gradients** (dimension + direction + boundary value). Validated dimensions are then **state-locked**: read-only in subsequent iterations, so the Executor can only mutate failing dimensions.
- **Mechanism**: This guarantees that the set of verified constraints grows monotonically across iterations (V_t ⊆ V_{t+1}), terminating in either SUCCESS or `FAILED_PARADOX`.
- **Code**: `engine/reviewer.py`, `engine/orchestrator.py` (State Locking is enforced via the orchestrator's per-iteration JSON-schema rewriting on top of the Reviewer's PASS markers).

---

## 2. Operational Lifecycle

1. **DECOMPOSE** — Root request + Domain Harness → DAG of atomic nodes. The plan itself is audited against the Harness before execution begins.
2. **DISPATCH** — Orchestrator schedules nodes by dependency (serial / parallel / hybrid).
3. **EXECUTE** — Each Executor receives `Task + Scoped Context + Harness Clause`. Output is a verifiable claim against a contract, not free-form prose.
4. **REVIEW** — Reviewer evaluates the aggregated artifact against the same Harness clauses. UAI runs deterministic Python assertions; the Reviewer wraps the failure trace into a structured gradient.
5. **CONVERGE** — On UAI failure, Executors mutate only unlocked dimensions. On global infeasibility, the Reviewer emits `FAILED_PARADOX` and triggers Strategic Negotiation (see Pillar 4 below).

---

## 3. Termination and Strategic Negotiation

CAAF has two architectural exits:

- **`SUCCESS`** — Every Harness clause passes on the aggregated artifact.
- **`FAILED_PARADOX`** — The Reviewer's root-cause analysis over deterministic UAI failures determines the constraint set is irreconcilable. This is *not* a runtime error: the system surfaces the minimal unsatisfiable constraint subset and a quantified resolution menu (e.g., "relax constraint X by Δ to obtain feasibility"), then waits for human authorization. No silent constraint relaxation is possible.

---

## 4. Skills vs. Harness: The Enterprise Asset Layer

| Dimension | Skills (Expert Brain) | Harness (Compliance Sensor) |
| :--- | :--- | :--- |
| **Nature** | Heuristic | Deterministic |
| **Asset Type** | Commodity Knowledge | Proprietary Asset |
| **Role** | Guides Generation (Executor) | Computes Gradients (Reviewer) |
| **Logic** | Probabilistic | Hard-coded / Verifiable |

The **Harness Registry** allows organizations to swap the underlying LLM (open-weight or frontier) while retaining the same compliance floor. The architectural reliability behavior is API-independent.

---

## 5. Key Files

| Concern | Files |
| :--- | :--- |
| Orchestration | `engine/orchestrator.py`, `engine/rad.py`, `engine/context_resolver.py` |
| Harness Registry | `harness/engine.py`, `harness/data/*.yaml` |
| UAI / assertions | `harness/engine.py` (`AssertionEngine`) |
| Reviewer / gradients | `engine/reviewer.py` |
| State Locking | enforced in `engine/orchestrator.py` (JSON-schema rewriting between iterations) |
| LLM adapters | `adapters/openai_adapter.py`, `adapters/base.py` |
| Paper experiments | `demos/benchmark_*.py` |

See the CAAF paper for the formal convergence statement (Eq. 2 in §3.3) and the empirical validation across the L3 AD and pharmaceutical flow reactor benchmarks.
