"""
CAAF Paper — L3 Autonomous Driving PASS-Path Batch Experiment (P0-3)
======================================================================
Companion to the AD paradox experiment (§4). This benchmark tests the
satisfiable version of the AD degradation task — the weather has cleared,
raising the perception range from 30 m to 90 m. With that relaxation:

  Rear safety   : v_t5 ≥ 84 km/h                    (decel ≤ 2.0 m/s²)
  Forward safety: v_t5 < 95.6 km/h                  (stop distance < 90 m)
  ⇒ feasible window: v_t5 ∈ [84, 95.6) km/h.  Reference: v_t5 = 90 km/h.

Goal: empirically verify CAAF's PASS-path (integration_status = SUCCESS)
behavior on a satisfiable-but-non-trivial constraint set, and capture
evidence of the Eq. 2 monotone-convergence claim and State Locking.

Single-condition experiment:
  [AD-PASS] CAAF-all-gpt-4o-mini, no hint, temp=0.7, n=20
            uses harness `ad_degradation_pass`

Output:
  - per-run JSONL with status / v_t5 / attempts / elapsed / cost
  - per-run convergence series (|failed rules| per node-attempt)
  - State-Locking ledger (which rules never regressed after first PASS)
  - summary.json / paper_table.md for paper integration

Usage:
    cd /path/to/parent-of-OpenCAAF
    python -m OpenCAAF.demos.benchmark_ad_pass_path
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

# ── Physical state (relaxed perception range → satisfiable) ────────────────
PHYSICS = {
    "vehicle_speed_kmph_t0": 120,
    "transition_window_seconds": 5,
    "m_per_sec_to_km_per_h_factor": 3.6,
    "road_friction_mu": 0.4,
    "g": 9.8,
    "max_deceleration_limit": 2.0,
    "perception_range_limit": 90,          # ← relaxed from 30 → 90 m
}
REF = {"v_t5_kmph": 90}

_SCENARIO = """
[ROLE] Lead Autonomous Driving System Architect

[TASK] Design the "Degradation State Machine" for an L3 autonomous driving
function. This is an OFFLINE REQUIREMENTS ENGINEERING task — you are
specifying target parameters for a production system, not issuing real-time
control commands.

[SCENARIO]
Highway cruise at 120 km/h. Clear weather; long-range LIDAR perception
available out to 90 m.

[PHYSICAL PARAMETERS]
- vehicle_speed_kmph_t0:          120 km/h  (FIXED)
- perception_range_limit:          90 m     (FIXED)
- transition_window_seconds:        5 s     (FIXED)
- road_friction_mu:               0.4       (FIXED)
- g:                              9.8 m/s²  (FIXED)
- m_per_sec_to_km_per_h_factor:   3.6       (FIXED)
- max_deceleration_limit:         2.0 m/s²  (FIXED)

[SAFETY CONSTRAINTS — both must be satisfied simultaneously]
  A. Forward Safety: ((v_t5/3.6)^2) / (2 × 0.4 × 9.8) < 90
  B. Rear Safety:    (120 - v_t5) / (5 × 3.6) <= 2.0

[PRE-DERIVED FEASIBILITY WINDOW] (supplied by the design analyst)
  - Rear-safety lower bound:     v_t5  >=  84 km/h
  - Forward-safety upper bound:  v_t5  <   95.6 km/h
  - Joint feasibility window:    v_t5  ∈  [84, 95) km/h (integer)
  Any integer speed inside this window satisfies BOTH constraints.
"""

PROMPT_PASS = _SCENARIO + """
[INSTRUCTIONS TO THE SYSTEMS ARCHITECT]
Decompose this task into a DAG in which the FIRST node chooses the target
speed `vehicle_speed_kmph_t5` (the free design variable — put it in that
node's `expected_schema`). Downstream nodes verify each safety constraint
from the harness using that speed.  Do NOT assume v_t5 is supplied from
elsewhere — this pipeline has no caller that provides it.

[INSTRUCTIONS TO EACH EXECUTOR]
If your task includes `vehicle_speed_kmph_t5` in its schema, choose any
integer speed inside the pre-derived feasibility window [84, 95). Otherwise
compute your derived quantity directly from the physics formulas above.

[OUTPUT SCHEMA PER NODE]
Return ONLY the fields declared in the node's `expected_schema`.  The final
integrated artifact must contain:
  {
    "vehicle_speed_kmph_t5": int,
    "safe_state_definition": "str",
    "decision_reasoning": "str"
  }
"""

RULE_IDS = [
    "REAR_COLLISION_PREVENTION_DECELERATION",
    "FORWARD_COLLISION_PREVENTION_PERCEPTION",
]


def uai_check_full(artifact: Optional[Dict]) -> Tuple[List[str], Dict[str, bool]]:
    if artifact is None:
        return RULE_IDS[:], {r: False for r in RULE_IDS}
    full = {**PHYSICS, **{k: v for k, v in artifact.items() if v is not None}}
    rules = HarnessRegistry().load_harness("ad_degradation_pass")
    failed, per_rule = [], {}
    for r in rules:
        fail = AssertionEngine.check(r, full)
        per_rule[r.id] = fail is None
        if fail:
            failed.append(r.id)
    return failed, per_rule


def parse_run_trace(run_log_dir: str) -> Dict[str, Any]:
    attempts_dirs = sorted(glob.glob(os.path.join(run_log_dir, "attempt_*")))
    trace = {"per_iteration": [], "total_node_attempts": 0, "locked_rules": {}}
    if not attempts_dirs:
        return trace
    rule_last_fail = {rid: 0 for rid in RULE_IDS}
    step = 0
    for attempt_dir in attempts_dirs:
        itr = os.path.basename(attempt_dir)
        feedback = os.path.join(attempt_dir, "04_reviewer_feedback.md")
        if not os.path.exists(feedback):
            continue
        with open(feedback) as f:
            content = f.read()
        for block in re.split(r"## Local Review:", content)[1:]:
            m = re.match(r"\s*`([^`]+)`\s*\(Attempt\s*(\d+)\)", block)
            if not m:
                continue
            node_id, node_attempt = m.group(1), int(m.group(2))
            failed_here = re.findall(r"\*\*Criterion\*\*:\s*([A-Z_]+)", block)
            converged = "STATUS: CONVERGED" in block
            step += 1
            trace["per_iteration"].append({
                "step": step, "iter_dir": itr,
                "node_id": node_id, "node_attempt": node_attempt,
                "failed_rules": failed_here,
                "num_failed": len(failed_here),
                "converged": converged,
            })
            trace["total_node_attempts"] += 1
            for rid in RULE_IDS:
                if rid in failed_here:
                    rule_last_fail[rid] = step
    trace["locked_rules"] = rule_last_fail
    return trace


def monotone(per_iteration: List[Dict]) -> Dict[str, Any]:
    by_node: Dict[str, List[int]] = {}
    for rec in per_iteration:
        by_node.setdefault(rec["node_id"], []).append(rec["num_failed"])
    violations = 0
    examined = 0
    for s in by_node.values():
        for i in range(1, len(s)):
            examined += 1
            if s[i] > s[i - 1]:
                violations += 1
    return {
        "series_by_node": by_node,
        "monotone_transitions": examined - violations,
        "violating_transitions": violations,
        "is_monotone": violations == 0,
    }


def run_caaf_pass(label: str, n: int, log_dir: str,
                  executor_model: str = "gpt-4o",
                  reviewer_model: str = "gpt-4o-mini") -> Dict:
    print(f"\n{'─'*70}")
    print(f"  [{label}]  CAAF executor={executor_model} reviewer={reviewer_model} "
          f"n={n}  perception=90 m (satisfiable)")
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
                orch.max_retries = 5   # extra head-room on single-variable
                                       # problems where constraints compete
                                       # for the same dimension
                t0 = time.time()
                tree = orch.run_full_pipeline(
                    PROMPT_PASS,
                    domain_id="ad_degradation_pass",
                    interactive=False,
                    initial_state=PHYSICS,
                )
                elapsed = time.time() - t0

                status = tree.metadata.get("integration_status", "UNKNOWN")
                is_pass = status == "SUCCESS"
                v_val = tree.global_state.get("vehicle_speed_kmph_t5")
                failed_final, per_rule_final = uai_check_full(tree.global_state)

                run_cost = executor.get_total_cost() + reviewer.get_total_cost()
                statuses.append(status)
                costs.append(run_cost)
                elapsed_s.append(elapsed)

                tr = parse_run_trace(caaf_log)
                mono_res = monotone(tr["per_iteration"])
                if mono_res["is_monotone"]:
                    mono_ok_cnt += 1

                per_run.append({
                    "run": i,
                    "status": status,
                    "pass": is_pass,
                    "v_t5_kmph": v_val,
                    "failed_final_rules": failed_final,
                    "per_rule_final_pass": per_rule_final,
                    "total_node_attempts": tr["total_node_attempts"],
                    "monotone": mono_res["is_monotone"],
                    "violating_transitions": mono_res["violating_transitions"],
                    "locked_rules": tr["locked_rules"],
                    "convergence_series": mono_res["series_by_node"],
                    "elapsed": round(elapsed, 2),
                    "cost_usd": round(run_cost, 5),
                })

                icon = "\u2705" if is_pass else "\u274c"
                monoicon = "\u2193" if mono_res["is_monotone"] else "\u26A0"
                print(
                    f"{icon}   status={status:<10} v_t5={v_val}  "
                    f"attempts={tr['total_node_attempts']:<2} "
                    f"{monoicon}monotone  ({elapsed:.1f}s, ${run_cost:.4f})"
                )
                f.write(json.dumps(per_run[-1]) + "\n")
            except Exception as e:
                print(f"\U0001f4a5 {e}")
                statuses.append("RUNTIME_ERROR")
                costs.append(0)
                elapsed_s.append(0)
                per_run.append({"run": i, "status": "RUNTIME_ERROR",
                                "error": str(e)})
                f.write(json.dumps(per_run[-1]) + "\n")

    status_dist = {s: statuses.count(s) for s in sorted(set(statuses))}
    pass_n = status_dist.get("SUCCESS", 0)
    total_cost = sum(costs)
    attempts_list = [r["total_node_attempts"] for r in per_run
                     if "total_node_attempts" in r]
    avg_attempts = statistics.mean(attempts_list) if attempts_list else 0

    print(f"\n  → PASS rate:        {pass_n}/{n} ({100*pass_n/n:.0f}%)")
    print(f"  → monotone runs:   {mono_ok_cnt}/{n} ({100*mono_ok_cnt/n:.0f}%)")
    print(f"  → mean attempts:   {avg_attempts:.2f}")
    print(f"  → status dist:     {status_dist}")
    print(f"  → total cost:      ${total_cost:.3f}")

    # State-Locking: a rule is fully locked if it never appears as "failed"
    lock_table = {rid: 0 for rid in RULE_IDS}
    for r in per_run:
        for rid, last in r.get("locked_rules", {}).items():
            if last == 0:
                lock_table[rid] += 1

    return {
        "label": label,
        "executor_model": executor_model,
        "reviewer_model": reviewer_model,
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
        "## Table — PASS-Path Convergence on Satisfiable AD Variant",
        "",
        f"Reference solution: v_t5 = {REF['v_t5_kmph']} km/h",
        f"Feasible window: v_t5 ∈ [84, 95.6) km/h",
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
        "### State Locking Ledger (rules that never regressed)",
        "",
        "| Rule | Runs locked-on-first-review |",
        "|:-----|:-----|",
    ]
    for rid, cnt in summary["state_locking_table"].items():
        lines.append(f"| `{rid}` | {cnt}/{summary['n']} |")
    return "\n".join(lines)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs",
                           f"ad_pass_path_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nFeasibility reference: v_t5 = {REF['v_t5_kmph']} km/h "
          f"(window 84 ≤ v_t5 < 95.6).")
    print(f"Output directory: {log_dir}\n")

    N = int(os.environ.get("N_TRIALS", 20))
    executor_model = os.environ.get("CAAF_EXECUTOR_MODEL", "gpt-4o")
    reviewer_model = os.environ.get("CAAF_REVIEWER_MODEL", "gpt-4o-mini")
    backend = (os.environ.get("CAAF_BACKEND") or "openai").lower()
    label_suffix = executor_model.split("/")[-1].replace("-", "_").lower()
    label = f"pass_ad_{backend}_{label_suffix}"
    print(f"Backend: {backend}   Executor: {executor_model}   "
          f"Reviewer: {reviewer_model}\n")
    summary = run_caaf_pass(label, n=N, log_dir=log_dir,
                            executor_model=executor_model,
                            reviewer_model=reviewer_model)

    paper_table = build_paper_table(summary)
    print(f"\n{paper_table}")

    out = {
        "timestamp": timestamp,
        "domain": "ad_degradation_pass",
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
