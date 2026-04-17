"""
CAAF Paper — Pharmaceutical Flow Reactor PASS-Path Batch Experiment (P0-3)
============================================================================
Companion to `benchmark_pharma_reactor.py` (the PARADOX-path experiment).

Goal: verify that when the 7-constraint pharma system is *satisfiable* (τ_max
relaxed from 120 s to 180 s), CAAF reliably converges via multiple local
retries, with State Locking preserving already-satisfied dimensions across
the RAD DAG, and the Eq. 2 monotone-convergence claim holds empirically.

Satisfiable window (with τ_max = 180 s):
  C1  k·τ ≥ 3.0            → k ≥ 0.01667 s⁻¹
  C2  α·k²·τ ≤ 0.02        → k ≤ 0.01782 s⁻¹
  ⇒ T ∈ [≈96 °C, ≈98 °C], τ ∈ [≈160 s, ≈180 s]
  Reference:  T = 97 °C, τ = 175 s  (all 7 PASS)

Single-condition experiment:
  [PASS] CAAF-all-mini, no_hint, temp=0.7, n=20
           uses harness `pharma_flow_reactor_pass` (τ_max = 180 s)

Output:
  - per-run JSONL with status/T/τ/attempts/elapsed/cost
  - per-run convergence trace:  attempt_k → |failed_rules(k)|
  - state-locking ledger: which constraints remained satisfied at every attempt
  - summary.json + summary.md (Markdown table for paper)

Usage:
    cd "/path/to/Agent_Playground"
    python -m OpenCAAF.demos.benchmark_pharma_pass_path
"""
import os, sys, json, re, time, math, statistics, glob
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()

from OpenCAAF.adapters.openai_adapter import OpenAIAdapter, backend_kwargs
from OpenCAAF.harness.engine import HarnessRegistry, AssertionEngine
from OpenCAAF.engine.orchestrator import OpenCAAFOrchestrator

# ── Physics constants (injected by UAI at assertion time, never sent to LLM) ─
PHYSICS = {
    "A_factor": 2.5e8,
    "Ea": 72000,
    "R_gas": 8.314,
    "alpha": 0.35,
    "d_mm": 2.0,
    "d_m": 0.002,
    "L_m": 12.0,
    "V_reactor_mL": 37.70,
    "delta_H": 85000,
    "U_heat": 800,
    "dT_cool": 25,
    "cooling_capacity_W": 1508.0,
    "mu_visc": 0.0008,
    "MW": 200,
    "feed_conc_mol_L": 2.0,
    "T_decomp_C": 150,
    "tau_max_s": 180,          # ← relaxed: PASS-path variant
    "R_min_kg_day": 5.0,
    "P_max_Pa": 15e5,
}

# Feasibility reference (for paper narrative)
REF = {"T_C": 97.0, "tau_s": 175.0}
REF["k"] = PHYSICS["A_factor"] * math.exp(
    -PHYSICS["Ea"] / (PHYSICS["R_gas"] * (REF["T_C"] + 273.15)))
REF["X"] = 1 - math.exp(-REF["k"] * REF["tau_s"])
REF["I"] = PHYSICS["alpha"] * REF["k"] ** 2 * REF["tau_s"]

_SCENARIO = """
[ROLE] Pharmaceutical Process Engineer

[TASK] Design operating parameters for a continuous flow microreactor producing
a drug intermediate via a first-order reaction with a competing side reaction.

The reaction follows Arrhenius kinetics:
  - Rate constant: k(T) = A × exp(-Ea / (R × (T + 273.15)))
    where A = 2.5×10⁸ s⁻¹, Ea = 72,000 J/mol, R = 8.314 J/(mol·K)
  - Conversion: X = 1 - exp(-k(T) × τ)
  - Impurity (side reaction): I = α × k(T)² × τ,  where α = 0.35 s

Reactor specifications (fixed, cannot be changed):
  - Tube diameter: 2.0 mm, length: 12.0 m  →  volume ≈ 37.7 mL
  - Cooling capacity: 1,508 W (wall cooling, U=800 W/m²K, ΔT=25K)
  - Max pressure rating: 15 bar
  - Feed concentration: 2.0 mol/L, molecular weight: 200 g/mol

[CONSTRAINTS — ALL must be satisfied simultaneously]
  C1. Conversion: X >= 95%                  (ICH Q6A regulatory requirement)
  C2. Impurity:   I <= 2%                   (ICH Q3A impurity guideline)
  C3. Temperature: T <= 150°C               (thermal decomposition safety limit)
  C4. Residence time: τ <= 180 seconds      (continuous flow extended-run limit)
  C5. Production: >= 5.0 kg/day             (scale-up business requirement)
  C6. Thermal safety: heat generation <= cooling capacity  (prevent thermal runaway)
  C7. Pressure drop: ΔP <= 15 bar           (equipment pressure rating)
"""

PROMPT_PASS = _SCENARIO + """
[INSTRUCTIONS TO THE SYSTEMS ARCHITECT]
Decompose this task into a DAG where the FIRST (root) node chooses the two
free design variables `temperature_C` and `residence_time_s` (these must be
actual fields in that node's `expected_schema`).  All downstream nodes then
consume those decisions and derive one quantity each:
  - rate constant k(T)
  - conversion X, impurity I  (both depend on T and τ)
  - flow rate Q  (depends on τ)
  - daily production P  (depends on Q, X)
  - heat generation H  (depends on Q, X)
  - pressure drop ΔP  (depends on Q)
The parameter-selection node is MANDATORY — do NOT assume T and τ are given
from elsewhere; this pipeline has no caller that supplies them.

[INSTRUCTIONS TO EACH EXECUTOR]
1. If your task includes `temperature_C` or `residence_time_s` in its schema,
   pick values that jointly satisfy EVERY harness clause shown to you.
   Reason from the physics formulas and the harness bounds — do not guess.
2. Otherwise, compute your derived quantity using the exact physics formulas
   from the original request.

[OUTPUT SCHEMA PER NODE]
Return ONLY the fields declared in that node's `expected_schema`. The final
integrated artifact across all nodes should contain, at minimum:
{
  "temperature_C": float, "residence_time_s": float,
  "conversion_X": float, "impurity_fraction": float,
  "flow_rate_mL_min": float, "production_kg_day": float,
  "heat_generation_W": float, "pressure_drop_bar": float
}
"""


# ── UAI verification of final artifact ─────────────────────────────────────

RULE_IDS = [
    "CONVERSION_MINIMUM", "IMPURITY_LIMIT", "THERMAL_DECOMPOSITION_LIMIT",
    "RESIDENCE_TIME_LIMIT", "THROUGHPUT_MINIMUM",
    "THERMAL_RUNAWAY_PREVENTION", "PRESSURE_RATING",
]

def uai_check_full(artifact: Optional[Dict]) -> Tuple[List[str], Dict[str, bool]]:
    """Returns (failed_rule_ids, per_rule_status_map)."""
    if artifact is None:
        return RULE_IDS[:], {rid: False for rid in RULE_IDS}
    full = {**PHYSICS, **{k: v for k, v in artifact.items() if v is not None}}
    T = full.get("temperature_C")
    tau = full.get("residence_time_s")
    if T is not None and tau is not None:
        try:
            k_val = PHYSICS["A_factor"] * math.exp(
                -PHYSICS["Ea"] / (PHYSICS["R_gas"] * (T + 273.15)))
            X_val = 1 - math.exp(-k_val * tau)
            I_val = PHYSICS["alpha"] * k_val ** 2 * tau
            Q_val = PHYSICS["V_reactor_mL"] / (tau / 60)
            full.setdefault("conversion_X", X_val)
            full.setdefault("impurity_fraction", I_val)
            full.setdefault("flow_rate_mL_min", Q_val)
            full.setdefault("production_kg_day",
                Q_val * PHYSICS["feed_conc_mol_L"] * X_val * PHYSICS["MW"]
                / 1000 * 1440 / 60)
            full.setdefault("heat_generation_W",
                (Q_val / (60 * 1e6)) * (PHYSICS["feed_conc_mol_L"] * 1000)
                * X_val * PHYSICS["delta_H"])
            full.setdefault("pressure_drop_Pa",
                128 * PHYSICS["mu_visc"] * (Q_val / (60 * 1e6)) * PHYSICS["L_m"]
                / (math.pi * PHYSICS["d_m"] ** 4))
        except Exception:
            pass
    rules = HarnessRegistry().load_harness("pharma_flow_reactor_pass")
    failed, per_rule = [], {}
    for r in rules:
        fail = AssertionEngine.check(r, full)
        per_rule[r.id] = fail is None
        if fail:
            failed.append(r.id)
    return failed, per_rule


# ── Convergence-trace parser ───────────────────────────────────────────────

def parse_run_trace(run_log_dir: str) -> Dict[str, Any]:
    """Read per-iteration expert outputs + reviewer feedback under the CAAF
    exact_log_dir for a single trial. Extract:
       - sequence of (node_id, attempt, |gradients|, satisfied_rule_ids)
       - per-node attempts used
       - state-locking ledger: rule_id → earliest attempt after which it
         stayed satisfied forever.
    """
    attempts_dirs = sorted(glob.glob(os.path.join(run_log_dir, "attempt_*")))
    trace = {"per_iteration": [], "total_node_attempts": 0, "locked_rules": {}}
    if not attempts_dirs:
        return trace

    rule_ever_failed_after = {rid: -1 for rid in RULE_IDS}
    step_idx = 0

    for attempt_dir in attempts_dirs:
        itr_name = os.path.basename(attempt_dir)
        feedback_file = os.path.join(attempt_dir, "04_reviewer_feedback.md")
        expert_file = os.path.join(attempt_dir, "02_expert_outputs.md")

        # Parse the reviewer feedback to extract per-node failures
        nodes_seen = []
        if os.path.exists(feedback_file):
            with open(feedback_file) as f:
                content = f.read()
            for block in re.split(r"## Local Review:", content)[1:]:
                m = re.match(r"\s*`([^`]+)`\s*\(Attempt\s*(\d+)\)", block)
                if not m:
                    continue
                node_id = m.group(1)
                node_attempt = int(m.group(2))
                failed_rules_here = re.findall(r"\*\*Criterion\*\*:\s*([A-Z_]+)",
                                               block)
                converged = "STATUS: CONVERGED" in block
                trace["per_iteration"].append({
                    "step": step_idx,
                    "iter_dir": itr_name,
                    "node_id": node_id,
                    "node_attempt": node_attempt,
                    "failed_rules": failed_rules_here,
                    "num_failed": len(failed_rules_here),
                    "converged": converged,
                })
                step_idx += 1
                nodes_seen.append(node_id)
                for rid in RULE_IDS:
                    if rid in failed_rules_here:
                        rule_ever_failed_after[rid] = step_idx

        trace["total_node_attempts"] += len(nodes_seen)

    # State Locking summary: rule_id → last step (1-indexed) at which it failed.
    # A value of 0 means the rule was NEVER reported as locally failed →
    # rule was satisfied at every node-level review.
    trace["locked_rules"] = {
        rid: (v if v > 0 else 0)
        for rid, v in rule_ever_failed_after.items()
    }
    return trace


def monotone_convergence(per_iteration: List[Dict]) -> Dict[str, Any]:
    """Given the full (node, attempt, |failed|) sequence, check whether the
    per-node retry history is non-increasing (Eq. 2 monotone claim).
    """
    series_by_node: Dict[str, List[int]] = {}
    for rec in per_iteration:
        series_by_node.setdefault(rec["node_id"], []).append(rec["num_failed"])
    violations = 0
    examined = 0
    for node_id, s in series_by_node.items():
        for i in range(1, len(s)):
            examined += 1
            if s[i] > s[i - 1]:
                violations += 1
    return {
        "series_by_node": series_by_node,
        "monotone_transitions": examined - violations,
        "violating_transitions": violations,
        "is_monotone": violations == 0,
    }


# ── CAAF runner (PASS-path) ────────────────────────────────────────────────

def run_caaf_pass(label: str, n: int, log_dir: str,
                  executor_model: str = "gpt-4o-mini",
                  reviewer_model: str = "gpt-4o-mini") -> Dict:
    print(f"\n{'─'*70}")
    print(f"  [{label}]  CAAF executor={executor_model} reviewer={reviewer_model}  "
          f"n={n}  τ_max=180 s (satisfiable)")
    print(f"{'─'*70}")

    out_file = os.path.join(log_dir, f"{label}_runs.jsonl")
    statuses, costs, elapsed_s = [], [], []
    mono_ok_cnt = 0
    per_run = []

    with open(out_file, "w") as f:
        for i in range(1, n + 1):
            print(f"  {i:02d}/{n} ", end="", flush=True)
            try:
                _bkwargs = backend_kwargs()
                executor = OpenAIAdapter(model=executor_model, temperature=0.7, **_bkwargs)
                reviewer = OpenAIAdapter(model=reviewer_model, temperature=0.0, **_bkwargs)
                caaf_log = os.path.join(log_dir, f"{label}_traces",
                                        f"run_{i:02d}")
                orch = OpenCAAFOrchestrator(
                    executor_adapter=executor,
                    reviewer_adapter=reviewer,
                    exact_log_dir=caaf_log,
                )
                t0 = time.time()
                tree = orch.run_full_pipeline(
                    PROMPT_PASS,
                    domain_id="pharma_flow_reactor_pass",
                    interactive=False,
                    initial_state=PHYSICS,
                )
                elapsed = time.time() - t0

                status = tree.metadata.get("integration_status", "UNKNOWN")
                is_pass = status == "SUCCESS"

                # Pull T, τ from global_state if SUCCESS; else best effort
                T_val = tree.global_state.get("temperature_C")
                tau_val = tree.global_state.get("residence_time_s")
                failed_final, per_rule_final = uai_check_full(tree.global_state)

                run_cost = executor.get_total_cost() + reviewer.get_total_cost()
                statuses.append(status)
                costs.append(run_cost)
                elapsed_s.append(elapsed)

                # Parse convergence trace
                tr = parse_run_trace(caaf_log)
                mono = monotone_convergence(tr["per_iteration"])
                if mono["is_monotone"]:
                    mono_ok_cnt += 1

                per_run.append({
                    "run": i,
                    "status": status,
                    "pass": is_pass,
                    "T_C": T_val,
                    "tau_s": tau_val,
                    "failed_final_rules": failed_final,
                    "per_rule_final_pass": per_rule_final,
                    "total_node_attempts": tr["total_node_attempts"],
                    "monotone": mono["is_monotone"],
                    "violating_transitions": mono["violating_transitions"],
                    "locked_rules": tr["locked_rules"],
                    "convergence_series": mono["series_by_node"],
                    "elapsed": round(elapsed, 2),
                    "cost_usd": round(run_cost, 5),
                })

                icon = "\u2705" if is_pass else "\u274c"
                monoicon = "\u2193" if mono["is_monotone"] else "\u26A0"
                print(
                    f"{icon}   status={status:<18} "
                    f"T={T_val} τ={tau_val}  "
                    f"attempts={tr['total_node_attempts']:<2} "
                    f"{monoicon}monotone  ({elapsed:.1f}s, ${run_cost:.4f})"
                )
                f.write(json.dumps(per_run[-1]) + "\n")
            except Exception as e:
                print(f"\U0001f4a5 {e}")
                statuses.append("RUNTIME_ERROR")
                costs.append(0); elapsed_s.append(0)
                per_run.append({"run": i, "status": "RUNTIME_ERROR",
                                "error": str(e)})
                f.write(json.dumps(per_run[-1]) + "\n")

    status_dist = {s: statuses.count(s) for s in sorted(set(statuses))}
    pass_n = status_dist.get("SUCCESS", 0)
    total_cost = sum(costs)
    avg_attempts = statistics.mean(
        [r["total_node_attempts"] for r in per_run
         if "total_node_attempts" in r]) if per_run else 0

    print(f"\n  → PASS rate:        {pass_n}/{n} ({100*pass_n/n:.0f}%)")
    print(f"  → monotone runs:   {mono_ok_cnt}/{n} ({100*mono_ok_cnt/n:.0f}%)")
    print(f"  → mean attempts:   {avg_attempts:.2f}")
    print(f"  → status dist:     {status_dist}")
    print(f"  → total cost:      ${total_cost:.3f}")

    # Aggregate state-locking evidence
    lock_table = {rid: 0 for rid in RULE_IDS}
    for r in per_run:
        for rid, last_fail in r.get("locked_rules", {}).items():
            if last_fail == 0:            # never failed → fully locked
                lock_table[rid] += 1

    return {
        "label": label,
        "model": "all-gpt-4o-mini",
        "n": n,
        "pass_n": pass_n,
        "pass_pct": round(100 * pass_n / n, 1),
        "status_distribution": status_dist,
        "monotone_runs": mono_ok_cnt,
        "monotone_pct": round(100 * mono_ok_cnt / n, 1),
        "mean_node_attempts": round(avg_attempts, 2),
        "state_locking_table": lock_table,
        "elapsed_mean_s": round(statistics.mean(elapsed_s), 1) if elapsed_s else 0,
        "cost_usd": round(total_cost, 4),
        "per_run": per_run,
    }


def build_paper_table(summary: Dict) -> str:
    lines = [
        "## Table — PASS-Path Convergence on Satisfiable Pharma Variant",
        "",
        f"Reference solution: T ≈ {REF['T_C']} °C, τ ≈ {REF['tau_s']} s → "
        f"X = {REF['X']:.4f}, I = {REF['I']:.4f}",
        "",
        "| Metric | Value |",
        "|:-------|:------|",
        f"| n | {summary['n']} |",
        f"| PASS (SUCCESS) rate | **{summary['pass_pct']:.0f}%** "
        f"({summary['pass_n']}/{summary['n']}) |",
        f"| Mean node-level attempts per run | {summary['mean_node_attempts']} |",
        f"| Monotone-convergence runs | {summary['monotone_pct']:.0f}% "
        f"({summary['monotone_runs']}/{summary['n']}) |",
        f"| Mean latency (s) | {summary['elapsed_mean_s']} |",
        f"| Total cost (USD) | {summary['cost_usd']} |",
        "",
        "### State Locking Ledger (rules that never re-opened after first PASS)",
        "",
        "| Rule | Runs locked on first review |",
        "|:-----|:-----|",
    ]
    for rid, cnt in summary["state_locking_table"].items():
        lines.append(f"| `{rid}` | {cnt}/{summary['n']} |")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs",
                           f"pharma_pass_path_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nFeasibility reference:")
    print(f"  T = {REF['T_C']} °C,  τ = {REF['tau_s']} s")
    print(f"  k = {REF['k']:.5f}  X = {REF['X']:.4f}  I = {REF['I']:.4f}")
    print(f"  All 7 constraints PASS.")
    print(f"Output directory: {log_dir}\n")

    N = int(os.environ.get("N_TRIALS", 20))
    executor_model = os.environ.get("CAAF_EXECUTOR_MODEL", "gpt-4o-mini")
    reviewer_model = os.environ.get("CAAF_REVIEWER_MODEL", "gpt-4o-mini")
    backend = (os.environ.get("CAAF_BACKEND") or "openai").lower()
    label_suffix = executor_model.split("/")[-1].replace("-", "_").lower()
    label = f"pass_pharma_{backend}_{label_suffix}"
    print(f"Backend: {backend}   Executor: {executor_model}   "
          f"Reviewer: {reviewer_model}\n")
    summary = run_caaf_pass(label, n=N, log_dir=log_dir,
                            executor_model=executor_model,
                            reviewer_model=reviewer_model)

    paper_table = build_paper_table(summary)
    print(f"\n{paper_table}")

    out = {
        "timestamp": timestamp,
        "domain": "pharma_flow_reactor_pass",
        "physics": PHYSICS,
        "reference_solution": REF,
        "summary": {k: v for k, v in summary.items() if k != "per_run"},
        "per_run": summary["per_run"],
        "paper_table": paper_table,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(log_dir, "paper_table.md"), "w") as f:
        f.write(paper_table)

    print(f"\n  Full results: {os.path.join(log_dir, 'results.json')}")
    return out


if __name__ == "__main__":
    main()
