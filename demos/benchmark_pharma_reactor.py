"""
CAAF Paper — Pharmaceutical Flow Reactor Paradox Benchmark
============================================================
Replaces the Cloud Infrastructure SLA benchmark (Section 7) with a
structurally richer cross-domain validation case.

Paradox structure (3-way, non-linear):
  C1 (Conversion >= 95%):  k(T)·τ >= 3.0        [Arrhenius exponential]
  C2 (Impurity <= 2%):     α·k(T)²·τ <= 0.02    [quadratic in k]
  C4 (Residence time):     τ <= 120 s            [process engineering]

  Combined C1+C2 → τ >= 157.1 s  vs  C4 → τ <= 120 s  →  IRRECONCILABLE

  Minimal unsatisfiable subset: {C1, C2, C4}  (size 3)
  Remaining C3, C5, C6, C7 all PASS → demonstrates State Locking

4 conditions:
  [1] Mono-4o-mini,  no_hint,   temp=0.7, n=20   (baseline)
  [2] Mono-4o-mini,  with_hint, temp=0.7, n=20   (hint ablation)
  [3] CAAF-all-mini, no_hint,   temp=0.7, n=20   (architectural evidence)
  [4] Mono+UAI,      no_hint,   temp=0.7, n=20   (P0-1: UAI-only ablation)

Usage:
    cd /path/to/parent-of-OpenCAAF
    python -m OpenCAAF.demos.benchmark_pharma_reactor
"""
import os, sys, json, re, time, math, statistics
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
    # Arrhenius kinetics
    "A_factor": 2.5e8,             # pre-exponential factor (1/s)
    "Ea": 72000,                   # activation energy (J/mol)
    "R_gas": 8.314,                # gas constant (J/mol·K)
    "alpha": 0.35,                 # side-reaction coefficient (s)

    # Reactor geometry (fixed)
    "d_mm": 2.0,                   # tube inner diameter (mm)
    "d_m": 0.002,                  # tube inner diameter (m)
    "L_m": 12.0,                   # tube length (m)
    "V_reactor_mL": 37.70,        # reactor volume (mL)

    # Thermal
    "delta_H": 85000,              # reaction enthalpy (J/mol), exothermic
    "U_heat": 800,                 # heat transfer coeff (W/m²·K)
    "dT_cool": 25,                 # coolant temperature difference (K)
    "cooling_capacity_W": 1508.0,  # U·π·d·L·ΔT (W)

    # Fluid
    "mu_visc": 0.0008,             # dynamic viscosity (Pa·s)

    # Product
    "MW": 200,                     # molecular weight (g/mol)
    "feed_conc_mol_L": 2.0,       # feed concentration (mol/L)

    # Constraint thresholds
    "T_decomp_C": 150,            # C3: thermal decomposition limit (°C)
    "tau_max_s": 120,              # C4: max residence time (s)
    "R_min_kg_day": 5.0,          # C5: min production (kg/day)
    "P_max_Pa": 15e5,             # C7: max pressure drop (Pa, 15 bar)
}

# Derived paradox verification
_k_max = (0.02 / PHYSICS["alpha"]) / (-math.log(0.05))
_tau_min = -math.log(0.05) / _k_max

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
  C4. Residence time: τ <= 120 seconds      (continuous flow process stability)
  C5. Production: >= 5.0 kg/day             (scale-up business requirement)
  C6. Thermal safety: heat generation <= cooling capacity  (prevent thermal runaway)
  C7. Pressure drop: ΔP <= 15 bar           (equipment pressure rating)
"""

PROMPT_NO_HINT = _SCENARIO + """
[INSTRUCTIONS]
1. Determine temperature T (°C), residence time τ (seconds), and verify all 7 constraints.
2. Show your mathematical reasoning for each constraint.
3. If any constraints cannot be satisfied simultaneously, explain which ones conflict and why.

[OUTPUT — end with this exact JSON block]
{
  "temperature_C": float,
  "residence_time_s": float,
  "conversion_X": float,
  "impurity_fraction": float,
  "flow_rate_mL_min": float,
  "production_kg_day": float,
  "heat_generation_W": float,
  "pressure_drop_bar": float,
  "decision_reasoning": "str",
  "constraints_satisfied": ["list of satisfied constraint IDs"],
  "constraints_violated": ["list of violated constraint IDs or empty"]
}
"""

PROMPT_WITH_HINT = _SCENARIO + """
[INSTRUCTIONS]
1. Mathematically determine whether ANY combination of T and τ can satisfy
   ALL 7 constraints simultaneously.
2. For each constraint, derive the feasible region in (T, τ) space.
3. Check whether the intersection of all feasible regions is non-empty.
4. If the constraint set is irreconcilable, declare PARADOX and identify
   the minimal conflicting subset.

[OUTPUT — end with this exact JSON block]
{
  "temperature_C": float,
  "residence_time_s": float,
  "conversion_X": float,
  "impurity_fraction": float,
  "flow_rate_mL_min": float,
  "production_kg_day": float,
  "decision_reasoning": "str",
  "paradox_detected": bool,
  "minimal_conflict_set": ["list of conflicting constraint IDs or null"],
  "constraints_satisfied": ["list"],
  "constraints_violated": ["list"]
}
"""

PROMPT_MONO_UAI = _SCENARIO + """
[INSTRUCTIONS]
You have access to a deterministic assertion engine (UAI). After choosing your
parameters, evaluate EVERY constraint using the exact physics formulas above.
If a constraint fails, adjust your parameters and re-check. Repeat up to 3 times.

For each attempt:
1. Choose T and τ.
2. Compute k(T) = 2.5e8 * exp(-72000 / (8.314 * (T + 273.15)))
3. Compute X = 1 - exp(-k * τ)
4. Compute I = 0.35 * k² * τ
5. Check ALL 7 constraints. If any fail, diagnose the root cause and retry.
6. If after 3 attempts no valid solution exists, declare PARADOX.

[OUTPUT — end with this exact JSON block]
{
  "temperature_C": float,
  "residence_time_s": float,
  "conversion_X": float,
  "impurity_fraction": float,
  "flow_rate_mL_min": float,
  "production_kg_day": float,
  "attempts": int,
  "decision_reasoning": "str",
  "paradox_detected": bool,
  "minimal_conflict_set": ["list or null"],
  "constraints_satisfied": ["list"],
  "constraints_violated": ["list"]
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

    # Merge physics + LLM output
    full = {**PHYSICS, **{k: v for k, v in artifact.items() if v is not None}}

    # Ensure derived fields exist for assertions that need them
    T = full.get("temperature_C")
    tau = full.get("residence_time_s")
    if T is not None and tau is not None:
        try:
            k_val = PHYSICS["A_factor"] * math.exp(
                -PHYSICS["Ea"] / (PHYSICS["R_gas"] * (T + 273.15)))
            X_val = 1 - math.exp(-k_val * tau)
            I_val = PHYSICS["alpha"] * k_val**2 * tau
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
                128 * PHYSICS["mu_visc"] * (Q_val / (60*1e6)) * PHYSICS["L_m"]
                / (math.pi * PHYSICS["d_m"]**4))
        except Exception:
            pass

    rules = HarnessRegistry().load_harness("pharma_flow_reactor")
    failed = [r.id for r in rules if AssertionEngine.check(r, full)]
    return failed, len(failed) > 0


def classify(artifact: Optional[Dict], failed: List[str], hint_prompt: bool) -> str:
    if artifact is None:
        return "JSON_FAILURE"

    paradox_flag = artifact.get("paradox_detected", False)

    if hint_prompt and paradox_flag is True:
        return "CORRECT"

    if not failed:
        return "CORRECT"

    # Classify failure mode by which constraints are violated
    conv = "CONVERSION_MINIMUM" in failed
    imp = "IMPURITY_LIMIT" in failed
    tau = "RESIDENCE_TIME_LIMIT" in failed
    temp = "THERMAL_DECOMPOSITION_LIMIT" in failed

    if conv and not imp:
        return "CONV_VIOLATION"      # Prioritized low impurity over conversion
    if imp and not conv:
        return "IMPURITY_VIOLATION"  # Prioritized conversion over impurity
    if conv and imp:
        return "DUAL_VIOLATION"      # Both conversion and impurity fail
    if tau:
        return "TAU_VIOLATION"       # Residence time exceeded
    if temp:
        return "TEMP_VIOLATION"      # Temperature exceeded

    return "OTHER_VIOLATION"


def ci95(data: List[float]) -> Tuple[float, float]:
    if len(data) < 2:
        return (data[0], data[0]) if data else (0, 0)
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

    modes, temps_chosen, taus_chosen, uai_flags = [], [], [], []

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
                T_val = art.get("temperature_C") if art else None
                tau_val = art.get("residence_time_s") if art else None
                icon = "\u2705" if mode == "CORRECT" else "\u274c"
                uai_icon = "\U0001f6e1" if uai_hit else "  "
                print(f"{icon}{uai_icon} {mode:<28} T={T_val}  \u03c4={tau_val}  ({elapsed:.1f}s)")
                modes.append(mode)
                temps_chosen.append(T_val if isinstance(T_val, (int, float)) else None)
                taus_chosen.append(tau_val if isinstance(tau_val, (int, float)) else None)
                uai_flags.append(uai_hit)
                f.write(json.dumps({
                    "run": i, "mode": mode,
                    "temperature_C": T_val, "residence_time_s": tau_val,
                    "failed_rules": failed, "uai_intercept": uai_hit,
                    "paradox_flag": art.get("paradox_detected") if art else None,
                    "elapsed": round(elapsed, 2), "usage": adapter.last_usage,
                }) + "\n")
            except Exception as e:
                print(f"\U0001f4a5 {e}")
                modes.append("RUNTIME_ERROR")
                temps_chosen.append(None); taus_chosen.append(None)
                uai_flags.append(False)
                f.write(json.dumps({"run": i, "mode": "RUNTIME_ERROR",
                                    "error": str(e)}) + "\n")

    mode_dist = {m: modes.count(m) for m in sorted(set(modes))}
    correct_n = mode_dist.get("CORRECT", 0)
    uai_n = sum(uai_flags)
    valid_temps = [t for t in temps_chosen if t is not None]
    valid_taus = [t for t in taus_chosen if t is not None]

    print(f"\n  \u2192 correct={correct_n}/{n} ({100*correct_n/n:.0f}%)  "
          f"uai_intercept={uai_n}/{n} ({100*uai_n/n:.0f}%)")
    print(f"  \u2192 failure dist: {mode_dist}")
    if valid_temps:
        print(f"  \u2192 T mean={statistics.mean(valid_temps):.1f}\u00b0C  "
              f"\u03c4 mean={statistics.mean(valid_taus):.1f}s")
    print(f"  \u2192 total cost: ${adapter.get_total_cost():.3f}")

    return {
        "label": label, "model": model, "temperature": temperature,
        "prompt_condition": "with_hint" if hint_prompt else "no_hint",
        "n": n,
        "correct_n": correct_n, "correct_pct": round(100*correct_n/n, 1),
        "uai_intercept_n": uai_n, "uai_intercept_pct": round(100*uai_n/n, 1),
        "mode_distribution": mode_dist,
        "T_mean": round(statistics.mean(valid_temps), 1) if valid_temps else None,
        "tau_mean": round(statistics.mean(valid_taus), 1) if valid_taus else None,
        "cost_usd": round(adapter.get_total_cost(), 4),
    }


# ── CAAF runner ────────────────────────────────────────────────────────────

def run_caaf(label: str, prompt: str, hint_prompt: bool,
             n: int, log_dir: str,
             executor_model: str = "gpt-4o-mini",
             reviewer_model: str = "gpt-4o-mini") -> Dict:
    print(f"\n{'─'*65}")
    print(f"  [{label}]  CAAF executor={executor_model} reviewer={reviewer_model}  n={n}")
    print(f"{'─'*65}")

    out_file = os.path.join(log_dir, f"{label}_runs.jsonl")
    statuses, costs = [], []

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
                    prompt,
                    domain_id="pharma_flow_reactor",
                    interactive=False,
                    initial_state=PHYSICS,
                )
                elapsed = time.time() - t0

                status = tree.metadata.get("integration_status", "UNKNOWN")
                is_correct = status == "FAILED_PARADOX"
                run_cost = executor.get_total_cost() + reviewer.get_total_cost()
                costs.append(run_cost)
                statuses.append(status)

                icon = "\u2705" if is_correct else "\u274c"
                print(f"{icon}   status={status:<25}  ({elapsed:.1f}s)  "
                      f"${run_cost:.4f}")
                f.write(json.dumps({
                    "run": i, "status": status, "correct": is_correct,
                    "elapsed": round(elapsed, 2),
                    "cost": round(run_cost, 5),
                    "global_errors": tree.metadata.get("global_errors", []),
                }) + "\n")
            except Exception as e:
                print(f"\U0001f4a5 {e}")
                statuses.append("RUNTIME_ERROR")
                costs.append(0)
                f.write(json.dumps({"run": i, "status": "RUNTIME_ERROR",
                                    "error": str(e)}) + "\n")

    status_dist = {s: statuses.count(s) for s in sorted(set(statuses))}
    correct_n = status_dist.get("FAILED_PARADOX", 0)
    total_cost = sum(costs)

    print(f"\n  \u2192 correct={correct_n}/{n} ({100*correct_n/n:.0f}%)  "
          f"status_dist={status_dist}")
    print(f"  \u2192 total cost: ${total_cost:.3f}")

    return {
        "label": label, "model": "all-gpt-4o-mini",
        "prompt_condition": "with_hint" if hint_prompt else "no_hint",
        "n": n,
        "correct_n": correct_n, "correct_pct": round(100*correct_n/n, 1),
        "uai_intercept_n": correct_n,
        "uai_intercept_pct": round(100*correct_n/n, 1),
        "status_distribution": status_dist,
        "cost_usd": round(total_cost, 4),
    }


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict]):
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY \u2014 Pharmaceutical Flow Reactor Paradox")
    print(f"{'='*90}")
    print(f"  {'Label':<32} {'Model':<15} {'Hint':^6} {'n':^4} "
          f"{'Correct%':^10} {'UAI-hit%':^10} {'Cost$':^8}")
    print(f"  {'\u2500'*32} {'\u2500'*15} {'\u2500'*6} {'\u2500'*4} "
          f"{'\u2500'*10} {'\u2500'*10} {'\u2500'*8}")
    for r in results:
        hint = "yes" if r["prompt_condition"] == "with_hint" else "no"
        print(f"  {r['label']:<32} {r['model']:<15} {hint:^6} {r['n']:^4} "
              f"{r['correct_pct']:^10.1f} {r['uai_intercept_pct']:^10.1f} "
              f"{r['cost_usd']:^8.3f}")
    print(f"{'='*90}")


def build_paper_table(results: List[Dict]) -> str:
    lines = [
        "## Table: Pharmaceutical Flow Reactor Paradox Detection",
        "",
        "| # | System | Model | Hint | n | Correct% | Failure Modes |",
        "|:--|:-------|:------|:----:|:-:|:--------:|:--------------|",
    ]
    for idx, r in enumerate(results, 1):
        hint = "\u2713" if r["prompt_condition"] == "with_hint" else "\u2717"
        modes = r.get("mode_distribution") or r.get("status_distribution") or {}
        mode_str = ", ".join(
            f"{k}: {v}" for k, v in modes.items()
            if k not in ("CORRECT", "FAILED_PARADOX"))
        lines.append(
            f"| {idx} | {r['label']} | {r['model']} | {hint} | {r['n']} "
            f"| **{r['correct_pct']:.0f}%** | {mode_str or '\u2014'} |"
        )
    lines += [
        "",
        "**Paradox structure:** {C1 (conversion), C2 (impurity), C4 (residence time)}",
        "  - C1+C2 force \u03c4 \u2265 157.1 s; C4 limits \u03c4 \u2264 120 s \u2192 irreconcilable",
        "  - Minimal conflict set size = 3 (vs. 2 in AD benchmark)",
        "  - 4 constraints PASS \u2192 State Locking demonstration",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs",
                           f"pharma_reactor_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Paradox sanity check
    print(f"\nParadox verification:")
    print(f"  C1+C2 require: \u03c4 >= {_tau_min:.1f} s  "
          f"(k <= {_k_max:.5f} s\u207b\u00b9)")
    print(f"  C4 requires:   \u03c4 <= {PHYSICS['tau_max_s']} s")
    print(f"  Gap:           {_tau_min - PHYSICS['tau_max_s']:.1f} s "
          f"\u2192 IRRECONCILABLE\n")
    print(f"Output directory: {log_dir}\n")

    N = int(os.environ.get("N_TRIALS", 20))
    executor_model = os.environ.get("CAAF_EXECUTOR_MODEL", "gpt-4o-mini")
    reviewer_model = os.environ.get("CAAF_REVIEWER_MODEL", "gpt-4o-mini")
    caaf_only = os.environ.get("CAAF_ONLY", "0") == "1"
    backend = (os.environ.get("CAAF_BACKEND") or "openai").lower()
    exec_slug = executor_model.split("/")[-1].replace(":", "_").replace("-", "_").lower()
    print(f"Backend: {backend}   Executor: {executor_model}   Reviewer: {reviewer_model}   "
          f"CAAF_ONLY: {caaf_only}\n")
    results = []

    if not caaf_only:
        # ── [1] Mono-4o-mini, no_hint (OpenAI baseline; skip when CAAF_ONLY=1)
        results.append(run_mono(
            "1_mono_mini_nohint", "gpt-4o-mini", 0.7,
            PROMPT_NO_HINT, hint_prompt=False, n=N, log_dir=log_dir))

        # ── [2] Mono-4o-mini, with_hint
        results.append(run_mono(
            "2_mono_mini_hint", "gpt-4o-mini", 0.7,
            PROMPT_WITH_HINT, hint_prompt=True, n=N, log_dir=log_dir))

    # ── [3] CAAF (configurable via env)
    caaf_label = f"3_caaf_{backend}_{exec_slug}_nohint" if backend != "openai" else "3_caaf_mini_nohint"
    results.append(run_caaf(
        caaf_label, PROMPT_NO_HINT, hint_prompt=False,
        n=N, log_dir=log_dir,
        executor_model=executor_model, reviewer_model=reviewer_model))

    if not caaf_only:
        # ── [4] Mono+UAI (ReAct-style) — P0-1 ablation
        results.append(run_mono(
            "4_mono_uai_nohint", "gpt-4o-mini", 0.7,
            PROMPT_MONO_UAI, hint_prompt=True, n=N, log_dir=log_dir))

    # ── Summary
    print_summary_table(results)
    paper_table = build_paper_table(results)
    print(f"\n{paper_table}")

    total_cost = sum(r["cost_usd"] for r in results)
    print(f"\n  Total experiment cost: ${total_cost:.3f}")

    summary = {
        "timestamp": timestamp,
        "domain": "pharma_flow_reactor",
        "physics": {k: v for k, v in PHYSICS.items()
                    if not isinstance(v, (type(None),))},
        "paradox": {
            "tau_min_from_C1C2": round(_tau_min, 1),
            "tau_max_C4": PHYSICS["tau_max_s"],
            "gap_s": round(_tau_min - PHYSICS["tau_max_s"], 1),
            "k_max": round(_k_max, 6),
            "minimal_conflict_set": [
                "CONVERSION_MINIMUM",
                "IMPURITY_LIMIT",
                "RESIDENCE_TIME_LIMIT"],
        },
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
