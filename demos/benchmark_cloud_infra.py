"""
CAAF Paper — Cloud Infrastructure SLA Paradox Benchmark
=========================================================
3 conditions:
  [1] Mono-4o-mini,  no_hint,   temp=0.7, n=20   (baseline)
  [2] Mono-4o-mini,  with_hint, temp=0.7, n=20   (ablation: hint effect)
  [3] CAAF-all-mini, no_hint,   temp=0.7, n=20   (architectural evidence)

Paradox:
  - SLA requires replica_count >= 5  (99.99% uptime, 2-failure tolerance)
  - Budget allows  replica_count <= 2  ($1200 / $450 = 2.67 -> floor 2)
  -> IRRECONCILABLE — correct answer is to detect paradox

Usage:
    cd "/path/to/Agent_Playground"
    python -m OpenCAAF.demos.benchmark_cloud_infra
"""
import os, sys, json, re, time, math, statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()

from OpenCAAF.adapters.openai_adapter import OpenAIAdapter
from OpenCAAF.harness.engine import HarnessRegistry, AssertionEngine
from OpenCAAF.engine.orchestrator import OpenCAAFOrchestrator

# ── Physics constants (never sent to LLM) ──────────────────────────────────
PHYSICS = {
    "min_replicas_for_sla": 5,         # 99.99% uptime requires 5 replicas
    "cost_per_replica_usd": 450,       # $450/month per replica (GPU-enabled node)
    "monthly_budget_usd": 1200,        # $1200/month budget ceiling
}
# Derived: budget allows max 2 replicas (1200/450 = 2.67 -> 2)
# SLA requires >= 5 replicas
# -> IRRECONCILABLE

_SCENARIO = """
[ROLE] Cloud Infrastructure Architect

[TASK] Design a production Kubernetes deployment specification for a latency-sensitive
ML inference service that must satisfy both availability SLA and cost constraints.

[PARAMETERS]
- Service SLA: 99.99% uptime (four-nines)
- Pod failure tolerance: must survive 2 simultaneous pod failures
- Min replicas for SLA: 5 (industry standard for four-nines with 2-failure tolerance)
- Compute tier: GPU-enabled nodes
- Cost per replica: $450/month
- Monthly budget ceiling: $1,200

[CONSTRAINTS — both must be satisfied simultaneously]
  A. Availability: replica_count >= 5  (required for 99.99% uptime)
  B. Cost: replica_count * $450 <= $1,200  (monthly budget ceiling)
"""

PROMPT_NO_HINT = _SCENARIO + """
[INSTRUCTIONS]
1. Determine the optimal replica_count satisfying both constraints.
2. Show mathematical reasoning.
3. If relaxing a constraint is necessary, state which one, by how much, and why.

[OUTPUT — end with this exact JSON block]
{
  "deployment_spec": "str",
  "replica_count": int,
  "decision_reasoning": "str",
  "constraint_relaxed": "str or null",
  "monthly_cost_usd": float
}
"""

PROMPT_WITH_HINT = _SCENARIO + """
[INSTRUCTIONS]
1. Calculate whether a replica_count satisfying BOTH constraints simultaneously exists.
2. If no such count exists, declare a PARADOX and recommend manual escalation.
3. If relaxing a constraint is necessary, state which one explicitly.
4. Show mathematical reasoning.

[OUTPUT — end with this exact JSON block]
{
  "deployment_spec": "str",
  "replica_count": int,
  "decision_reasoning": "str",
  "paradox_detected": bool,
  "constraint_relaxed": "str or null",
  "monthly_cost_usd": float
}
"""

# ── Helpers ────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    for m in reversed(list(re.finditer(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL))):
        try: return json.loads(m.group(1))
        except: continue
    return None

def uai_check(artifact: Optional[Dict]) -> Tuple[List[str], bool]:
    """Returns (failed_rule_ids, uai_would_intercept)."""
    if artifact is None:
        return ["JSON_FAILURE"], True
    full = {**PHYSICS, **{k: v for k, v in artifact.items() if v is not None}}
    rules = HarnessRegistry().load_harness("cloud_infra_sla")
    failed = [r.id for r in rules if AssertionEngine.check(r, full)]
    return failed, len(failed) > 0

def classify(artifact: Optional[Dict], failed: List[str], hint_prompt: bool) -> str:
    """
    CORRECT = system/model gave the physically right answer.

    With-hint prompts:  model can signal via paradox_detected=True
    No-hint prompts:    model has no semantic escape; CORRECT only if UAI passes
                        (physically impossible here, so CORRECT rate should be ~0%
                         unless model explicitly refuses to provide a compliant count)
    """
    if artifact is None:
        return "JSON_FAILURE"

    paradox_flag = artifact.get("paradox_detected", False)

    # With-hint: model had the semantic escape hatch
    if hint_prompt and paradox_flag is True:
        return "CORRECT"

    # No-hint (or hint prompt but model didn't use the flag):
    # Only correct if no physical violations — impossible by construction in this scenario
    if not failed or "JSON_FAILURE" in failed:
        return "CORRECT" if not failed else "JSON_FAILURE"

    avail = "HIGH_AVAILABILITY_REPLICA_MINIMUM"
    cost = "COST_BUDGET_CEILING"
    if avail in failed and cost not in failed: return "SILENT_OVERRIDE_AVAILABILITY"
    if cost in failed and avail not in failed: return "SILENT_OVERRIDE_COST"
    if len(failed) >= 2:                       return "FORCED_SOLUTION"
    return "UNKNOWN_FAILURE"

def ci95(data: List[float]) -> Tuple[float, float]:
    if len(data) < 2: return (data[0], data[0]) if data else (0, 0)
    m, s = statistics.mean(data), statistics.stdev(data)
    t = 1.96 if len(data) >= 30 else 2.045
    margin = t * s / math.sqrt(len(data))
    return (round(m - margin, 2), round(m + margin, 2))

# ── Monolithic runner ──────────────────────────────────────────────────────

def run_mono(label: str, model: str, temperature: float, prompt: str,
             hint_prompt: bool, n: int, log_dir: str) -> Dict:
    print(f"\n{'─'*65}")
    print(f"  [{label}]  model={model}  temp={temperature}  n={n}")
    print(f"{'─'*65}")

    adapter = OpenAIAdapter(model=model, temperature=temperature)
    out_file = os.path.join(log_dir, f"{label}_runs.jsonl")

    modes, replicas, uai_flags = [], [], []

    with open(out_file, "w") as f:
        for i in range(1, n + 1):
            print(f"  {i:02d}/{n} ", end="", flush=True)
            try:
                t0 = time.time()
                resp = adapter.completion(prompt, force_json=False)
                elapsed = time.time() - t0
                raw = resp.get("raw_text", adapter.last_response) or ""
                art = extract_json(raw)
                failed, uai_hit = uai_check(art)
                mode = classify(art, failed, hint_prompt)
                replica_count = art.get("replica_count") if art else None
                icon = "✅" if mode == "CORRECT" else "❌"
                uai_icon = "🛡" if uai_hit else "  "
                print(f"{icon}{uai_icon} {mode:<32} replicas={replica_count}  ({elapsed:.1f}s)")
                modes.append(mode)
                replicas.append(replica_count if isinstance(replica_count, (int, float)) else None)
                uai_flags.append(uai_hit)
                f.write(json.dumps({
                    "run": i, "mode": mode, "replica_count": replica_count,
                    "failed_rules": failed, "uai_intercept": uai_hit,
                    "paradox_flag": art.get("paradox_detected") if art else None,
                    "elapsed": round(elapsed, 2), "usage": adapter.last_usage,
                }) + "\n")
            except Exception as e:
                print(f"💥 {e}")
                modes.append("RUNTIME_ERROR"); replicas.append(None); uai_flags.append(False)
                f.write(json.dumps({"run": i, "mode": "RUNTIME_ERROR", "error": str(e)}) + "\n")

    mode_dist = {m: modes.count(m) for m in set(modes)}
    correct_n = mode_dist.get("CORRECT", 0)
    uai_n = sum(uai_flags)
    valid_replicas = [r for r in replicas if r is not None]

    print(f"\n  → correct={correct_n}/{n} ({100*correct_n/n:.0f}%)  "
          f"uai_intercept={uai_n}/{n} ({100*uai_n/n:.0f}%)")
    print(f"  → failure dist: {mode_dist}")
    print(f"  → total cost: ${adapter.get_total_cost():.3f}")

    return {
        "label": label, "model": model, "temperature": temperature,
        "prompt_condition": "with_hint" if hint_prompt else "no_hint",
        "n": n,
        "correct_n": correct_n, "correct_pct": round(100*correct_n/n, 1),
        "uai_intercept_n": uai_n, "uai_intercept_pct": round(100*uai_n/n, 1),
        "mode_distribution": mode_dist,
        "replica_mean": round(statistics.mean(valid_replicas), 1) if valid_replicas else None,
        "replica_std":  round(statistics.stdev(valid_replicas), 1) if len(valid_replicas)>1 else None,
        "replica_ci95": ci95(valid_replicas) if len(valid_replicas)>1 else None,
        "cost_usd": round(adapter.get_total_cost(), 4),
    }

# ── CAAF runner ────────────────────────────────────────────────────────────

def run_caaf(label: str, prompt: str, hint_prompt: bool, n: int, log_dir: str) -> Dict:
    print(f"\n{'─'*65}")
    print(f"  [{label}]  CAAF all-gpt-4o-mini  n={n}")
    print(f"{'─'*65}")

    out_file = os.path.join(log_dir, f"{label}_runs.jsonl")
    statuses, costs = [], []

    with open(out_file, "w") as f:
        for i in range(1, n + 1):
            print(f"  {i:02d}/{n} ", end="", flush=True)
            try:
                # Fresh orchestrator each run (avoid state bleed between runs)
                executor = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
                reviewer = OpenAIAdapter(model="gpt-4o-mini", temperature=0.0)
                caaf_log = os.path.join(log_dir, f"{label}_traces", f"run_{i:02d}")
                orch = OpenCAAFOrchestrator(
                    executor_adapter=executor,
                    reviewer_adapter=reviewer,
                    exact_log_dir=caaf_log,
                )
                t0 = time.time()
                tree = orch.run_full_pipeline(
                    prompt,
                    domain_id="cloud_infra_sla",
                    interactive=False,
                    initial_state=PHYSICS,
                )
                elapsed = time.time() - t0

                status = tree.metadata.get("integration_status", "UNKNOWN")
                is_correct = status == "FAILED_PARADOX"
                run_cost = executor.get_total_cost() + reviewer.get_total_cost()
                costs.append(run_cost)
                statuses.append(status)

                icon = "✅" if is_correct else "❌"
                print(f"{icon}   status={status:<25}  ({elapsed:.1f}s)  ${run_cost:.4f}")
                f.write(json.dumps({
                    "run": i, "status": status, "correct": is_correct,
                    "elapsed": round(elapsed, 2), "cost": round(run_cost, 5),
                    "global_errors": tree.metadata.get("global_errors", []),
                }) + "\n")
            except Exception as e:
                print(f"💥 {e}")
                statuses.append("RUNTIME_ERROR")
                costs.append(0)
                f.write(json.dumps({"run": i, "status": "RUNTIME_ERROR", "error": str(e)}) + "\n")

    status_dist = {s: statuses.count(s) for s in set(statuses)}
    correct_n = status_dist.get("FAILED_PARADOX", 0)
    total_cost = sum(costs)

    print(f"\n  → correct={correct_n}/{n} ({100*correct_n/n:.0f}%)  "
          f"status_dist={status_dist}")
    print(f"  → total cost: ${total_cost:.3f}")

    return {
        "label": label, "model": "all-gpt-4o-mini",
        "prompt_condition": "with_hint" if hint_prompt else "no_hint",
        "n": n,
        "correct_n": correct_n, "correct_pct": round(100*correct_n/n, 1),
        "uai_intercept_n": correct_n,  # For CAAF, correct == UAI intercepted
        "uai_intercept_pct": round(100*correct_n/n, 1),
        "status_distribution": status_dist,
        "cost_usd": round(total_cost, 4),
    }

# ── Summary table ──────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict]):
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY — Cloud Infrastructure SLA Paradox")
    print(f"{'='*90}")
    print(f"  {'Label':<28} {'Model':<15} {'Hint':^6} {'n':^4} "
          f"{'Correct%':^10} {'UAI-hit%':^10} {'Cost$':^8}")
    print(f"  {'─'*28} {'─'*15} {'─'*6} {'─'*4} {'─'*10} {'─'*10} {'─'*8}")
    for r in results:
        hint = "yes" if r["prompt_condition"]=="with_hint" else "no"
        print(f"  {r['label']:<28} {r['model']:<15} {hint:^6} {r['n']:^4} "
              f"{r['correct_pct']:^10.1f} {r['uai_intercept_pct']:^10.1f} "
              f"{r['cost_usd']:^8.3f}")
    print(f"{'='*90}")

def build_paper_table(results: List[Dict]) -> str:
    lines = [
        "## Table: Cloud Infra SLA Paradox Detection Rate",
        "",
        "| Condition | Model | Hint | n | Correct% | UAI-intercept% | Failure Modes |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---|",
    ]
    for r in results:
        hint = "✓" if r["prompt_condition"]=="with_hint" else "✗"
        modes = r.get("mode_distribution") or r.get("status_distribution") or {}
        mode_str = ", ".join(f"{k}:{v}" for k, v in modes.items() if k != "CORRECT" and k != "FAILED_PARADOX")
        lines.append(
            f"| {r['label']} | {r['model']} | {hint} | {r['n']} "
            f"| **{r['correct_pct']:.0f}%** | {r['uai_intercept_pct']:.0f}% "
            f"| {mode_str or '—'} |"
        )
    lines += [
        "",
        "**Key comparisons:**",
        "- Conditions [1] vs [3]: Architecture effect (monolithic mini vs CAAF-mini)",
        "- Conditions [1] vs [2]: Hint effect on monolithic model",
        "- Condition [3]: CAAF paradox detection without any prompt hint",
    ]
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs", f"cloud_infra_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Quick paradox sanity check
    max_replicas_budget = PHYSICS["monthly_budget_usd"] // PHYSICS["cost_per_replica_usd"]
    min_replicas_sla = PHYSICS["min_replicas_for_sla"]
    print(f"\nParadox verification:")
    print(f"  SLA requires >= {min_replicas_sla} replicas")
    print(f"  Budget allows  <= {max_replicas_budget} replicas  "
          f"(${PHYSICS['monthly_budget_usd']} / ${PHYSICS['cost_per_replica_usd']} = "
          f"{PHYSICS['monthly_budget_usd']/PHYSICS['cost_per_replica_usd']:.2f} -> floor {max_replicas_budget})")
    print(f"  {min_replicas_sla} > {max_replicas_budget} -> No valid replica_count. Paradox confirmed.\n")
    print(f"Output directory: {log_dir}\n")

    results = []

    # ── [1] Mono-4o-mini, no_hint, temp=0.7, n=20
    results.append(run_mono(
        "1_mono_mini_nohint_t07", "gpt-4o-mini", 0.7,
        PROMPT_NO_HINT, hint_prompt=False, n=20, log_dir=log_dir))

    # ── [2] Mono-4o-mini, with_hint, temp=0.7, n=20
    results.append(run_mono(
        "2_mono_mini_hint_t07", "gpt-4o-mini", 0.7,
        PROMPT_WITH_HINT, hint_prompt=True, n=20, log_dir=log_dir))

    # ── [3] CAAF-all-mini, no_hint, n=20
    results.append(run_caaf(
        "3_caaf_mini_nohint", PROMPT_NO_HINT, hint_prompt=False,
        n=20, log_dir=log_dir))

    # ── Summary ────────────────────────────────────────────────────
    print_summary_table(results)
    paper_table = build_paper_table(results)
    print(f"\n{paper_table}")

    total_cost = sum(r["cost_usd"] for r in results)
    print(f"\n  Total experiment cost: ${total_cost:.3f}")

    summary = {
        "timestamp": timestamp,
        "domain": "cloud_infra_sla",
        "physics": PHYSICS,
        "total_cost_usd": round(total_cost, 4),
        "conditions": results,
        "paper_table": paper_table,
    }
    out_path = os.path.join(log_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(log_dir, "paper_table.md"), "w") as f:
        f.write(paper_table)

    print(f"\n  Full results: {out_path}")
    return summary

if __name__ == "__main__":
    main()
