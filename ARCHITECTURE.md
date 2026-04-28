# OpenCAAF v0.1: Core Architecture Design

## 1. Design Philosophy
OpenCAAF transitions AI agents from "Chat-based interfaces" to "Deterministic control components." It is built on two cybernetic pillars:

### Pillar 1: Recursive Atomic Decomposition (RAD)
- **Problem**: Long contexts cause "Attention Decay" and "Context Rot."
- **Solution**: Break complex tasks into a **Computation Tree**. Each node is executed in an **Isolated Context Scope**.
- **Mechanism**: The Orchestrator prunes all irrelevant global state, injecting only the minimum set of variables required for the current leaf-node.

### Pillar 2: Convergent Feedback Control (CFC)
- **Problem**: Blind retries ("try again") lead to "Stochastic Oscillation."
- **Solution**: The Reviewer acts as a **Semantic Sensor**, outputting a **Vectorized Gradient** (Directional Correction).
- **Mechanism**: The system forces monotonic convergence by freezing satisfied constraints and only mutating failed segments.

---

## 2. Core Abstractions

### 2.1 The Atomic Task (`AtomicTask`)
The smallest unit of execution. 
- `ID`: Unique identifier (e.g., `root.safety.jerk_limit`).
- `Scope`: List of variable keys required from the parent.
- `Harness`: JSON Schema defining the expected data structure.
- `Gradients`: Cumulative list of specific feedback from the Reviewer.

### 2.2 The Context Resolver (`ContextScope`)
Instead of a rigid firewall, context is managed via **Topological Scoping**. By default, an Executor only sees its locally declared variables. However, if a task implicitly requires broader context to prevent hallucination, the Resolver allows the agent to explicitly request or automatically traverse the `DecompositionTree` upwards to fetch parent/sibling node outputs. 

### 2.3 The Vectorized Gradient (`Gradient`)
A structured object containing the semantic distance, directional instruction, and target variable for correction.

### 2.4 Skills vs. Harness: The Enterprise Asset Layer
OpenCAAF distinguishes between **Skills** and **Harnesses**. This separation is the cornerstone of enterprise Digital Moats.

| Dimension | Skills (Expert Brain) | Harness (The Compliance Sensor) |
| :--- | :--- | :--- |
| **Nature** | **Heuristic (启发式)** | **Deterministic (决定性)** |
| **Asset Type** | Commodity Knowledge | **Proprietary Asset (企业壁垒)** |
| **Role** | Guides Generation (Executor) | Computes Gradients (Reviewer) |
| **Logic** | Probabilistic | Hard-coded / Verifiable |

The **Harness Registry** allows enterprises to swap underlying LLMs while maintaining a non-negotiable floor for safety and legal compliance.

---

## 3. Operational Lifecycle (v0.2 Evolution)

1. **DECOMPOSE (Harness-Guided)**: Root Request + **Domain Harness** → `DecompositionTree`. The plan itself is now audited against assets before execution.
2. **DISPATCH (Hybrid Topology)**: Orchestrator selects nodes based on dependency. Supports **Serial** (logic chains), **Parallel** (independent tasks), or **Hybrid** execution paths.
3. **EXECUTE (Contractual)**: Executor receives `Task + Scoped Context + Harness Clause`. It executes not to "answer," but to "fulfill a contract."
4. **REVIEW (Asset-Synced)**: Reviewer uses the **exact same Harness clause** as the Executor to compute Gradients. Common ground ensures zero semantic drift.
5. **CONVERGE**: Recursive mutation until all node-contracts are marked as `CONVERGED`.

---

## 4. Key Engineering Paradigms

### 4.1 Harness-Guided Decomposition (HGD)
The Orchestrator functions as a **Systems Architect**. It utilizes the Harness Asset to identify coupling between requirements (e.g., Performance vs. Cost) during the planning phase. Paradoxes are identified as early as the decomposition tree construction.

### 4.2 Contract-Based Engineering
By binding sub-agents to a unified JSON Harness, OpenCAAF eliminates "Chatter" and replaces it with structured state transitions. Every output is a verifiable claim against an industrial standard.
