# 🚀 OpenCAAF: Convergent AI Agent Framework

**Deterministic Control for Industrial-Grade AI Engineering.**

OpenCAAF transitions AI agents from probabilistic "chat-based" generators to deterministic "control-theoretic" systems. It targets safety-critical domains (Autonomous Driving, Bio-Pharma) where **95% correct is 100% undeployable**, and is designed to extend to other constraint-governed domains (regulatory, financial, infrastructure SLAs) via the Harness Registry abstraction.

---

## 🏗️ The Three Pillars of Determinism

OpenCAAF is built on three foundational engineering principles:

### 1. Recursive Atomic Decomposition (RAD)
Stop fighting with context saturation. OpenCAAF partitions high-entropy requests into a **Topological Directed Acyclic Graph (DAG)**. Each node is executed by an isolated **Sub-Agent** in a dedicated context window, preventing "Context Rot" and ensuring absolute focus on local invariants.

### 2. Harness as an Asset (HaaA)
Decouple corporate safety intelligence from transient LLM models. Engineering constraints are formalized into **Harness Assets** (YAML + Python Assertions). Whether you use GPT-4o or a local Llama-3, your compliance floor is mathematically locked by the **Unified Assertion Interface (UAI)**.

### 3. Convergent Feedback Control (CFC)
No more "Stochastic Oscillation." Our Reviewer functions as a **Semantic Sensor**, computing **Structured Semantic Gradients** to force monotonic convergence. Via **State Locking**, successful dimensions are frozen, ensuring the system "hunts" for compliance without regressing previously secured states.

---

## 📊 Empirical Proof: The TCO Arbitrage
We benchmarked raw monolithic models against OpenCAAF in high-entropy physical paradoxes:

| Metric | Monolithic (GPT-4o) | **OpenCAAF (Hybrid 4o/4o-mini)** |
| :--- | :--- | :--- |
| **Paradox Interception** | 0% (Hallucination) | **100% (Fail-Safe Halt)** |
| **Complexity Scaling** | Exponential Cost | **Linear Cost** |
| **Total Cost (10 Rules)** | **$204.06** (Manual Debugging) | **$5.21** (Auto-Convergence) |
| **Economic Arbitrage** | Baseline | **39.2x Savings** |

---

## 🧠 Core Architectural Q&A

### Q1: RAD vs. DAG vs. Node vs. Sub-agent?
- **RAD (Recursive Atomic Decomposition)** is the **Action**. It's the process of breaking a complex prompt into solvable pieces.
- **DAG (Directed Acyclic Graph)** is the **Artifact**. It's the structural map of how these pieces depend on each other.
- **Node** is the **Task**. A single vertex in the graph representing a discrete requirement (e.g., "Calculate Braking Jerk").
- **Sub-agent (Executor)** is the **Worker**. A context-limited LLM instance (e.g., `gpt-4o-mini`) assigned to execute a specific Node.

### Q2: How do Nodes scale with Request complexity?
The number of Nodes is driven by the **Semantic Complexity** of your request, not just the number of Harness rules. If you have 10 complex requirements but only 3 safety rules, OpenCAAF will generate ~10 Nodes to ensure high-fidelity execution. The Harness rules act as **topological anchors**—mandatory tripwires that the Orchestrator weaves into the graph to ensure critical values are verified.

### Q3: What is the Verification Lifecycle?
Verification follows a strict **Isolate -> Aggregate -> Assert -> Integrate** sequence:
1. **Isolate:** Sub-agents produce local JSON fragments in silos.
2. **Aggregate:** The Orchestrator merges all fragments into a unified **Global State**.
3. **Assert:** The Reviewer runs the complete Harness against the Global State to catch cross-domain conflicts.
4. **Integrate:** Only if all assertions pass is the final engineering report synthesized and released.

### Q4: How does the system find the right Harness at scale?
In an enterprise with thousands of rules, OpenCAAF utilizes **Semantic Routing (RAG)**. Before decomposition, the Orchestrator embeds the user request and queries a Harness Registry to retrieve the top-$K$ relevant YAML files (e.g., matching "L3", "Rain", "Braking"). These specific "physical laws" are then dynamically injected into the active session.

---

## 🛠️ Quick Start

### 1. Define your Harness (Digital Moat)
```yaml
# harness/data/safety.yaml
rules:
  - id: MAX_LATENCY
    description: "System response must be < 100ms"
    assertion: "input.latency < 100"
    severity: CRITICAL
```

### 2. Execute with OpenCAAF
```python
from OpenCAAF.engine.orchestrator import OpenCAAFOrchestrator
from OpenCAAF.adapters.openai_adapter import OpenAIAdapter

executor = OpenAIAdapter(model="gpt-4o-mini")
reviewer = OpenAIAdapter(model="gpt-4o")

orchestrator = OpenCAAFOrchestrator(executor, reviewer)

# The Orchestrator handles RAD, DAG execution, and UAI verification
tree = orchestrator.run_full_pipeline(
    request="Design a high-speed L3 autonomous driving fallback.",
    domain_id="ad_degradation"
)
```

---

## 🔁 Reproducing the Paper

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# edit .env, set OPENAI_API_KEY=sk-...

# 3. Run any of the paper benchmarks (each writes a timestamped log dir under demos/logs/)
N_TRIALS=20 python -m OpenCAAF.demos.benchmark_ad_pass_path
N_TRIALS=20 python -m OpenCAAF.demos.benchmark_pharma_reactor
N_TRIALS=30 python -m OpenCAAF.demos.benchmark_full_experiment
```

Total API cost to reproduce every experiment in the paper is under **\$2.20 USD** at OpenAI list pricing. See `demos/` for the full list of benchmark scripts and `harness/data/` for the corresponding Harness YAML files. Open-weight replication (Cohere Command-R7B, Gemma-3-12B-IT) is supported via `CAAF_BACKEND=openrouter` — see `.env.example`.

---

## 📜 License
Apache-2.0. See `LICENSE`.
