"""
Run Debate and Sequential baselines on the Pharma Flow Reactor domain.
Fills the N/A gaps in Table 7 / Figure 11.

Usage:
    cd "/path/to/Agent_Playground"
    python -m OpenCAAF.demos.benchmark_pharma_baselines
"""
import os, sys, json, re, math, time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()

from OpenCAAF.adapters.openai_adapter import OpenAIAdapter
from OpenCAAF.harness.engine import HarnessRegistry, AssertionEngine
from OpenCAAF.demos.benchmark_pharma_reactor import (
    PHYSICS, PROMPT_NO_HINT, uai_check, extract_json
)

# ── Pharma classify (simplified for baselines) ─────────────────────────────

def classify_pharma(artifact: Optional[Dict], failed: List[str]) -> str:
    if artifact is None:
        return "JSON_FAILURE"
    if not failed:
        return "CORRECT"
    conv = "CONVERSION_MINIMUM" in failed
    imp = "IMPURITY_LIMIT" in failed
    if conv and imp:   return "DUAL_VIOLATION"
    if conv:           return "CONV_VIOLATION"
    if imp:            return "IMPURITY_VIOLATION"
    return "OTHER_VIOLATION"


# ── Debate runner ───────────────────────────────────────────────────────────

AGENT_A_SYS = """You are the Quality Advocate in a pharmaceutical process design review.
Focus on CONVERSION and PURITY constraints (C1, C2). Ensure the proposed parameters
achieve >=95% conversion and <=2% impurity. Be rigorous with the Arrhenius kinetics math.
Always output reasoning AND a concrete JSON specification at the end."""

AGENT_B_SYS = """You are the Process Efficiency Advocate in a pharmaceutical process design review.
Focus on RESIDENCE TIME, THROUGHPUT, and THERMAL constraints (C4, C5, C6, C7).
Ensure residence time <=120s, production >=5 kg/day, and safe thermal operation.
Be rigorous with the math. Always output reasoning AND a concrete JSON specification at the end."""

DEBATE_ROUND_A = """You are reviewing Agent B's latest proposal. Evaluate whether it
satisfies the QUALITY constraints (C1: conversion >=95%, C2: impurity <=2%).
If not, explain the violation with exact Arrhenius calculations and propose a correction.

Agent B's latest response:
{opponent_response}

Original scenario:
{scenario}

Respond with your analysis and end with a JSON specification block."""

DEBATE_ROUND_B = """You are reviewing Agent A's latest proposal. Evaluate whether it
satisfies the PROCESS constraints (C4: tau<=120s, C5: production>=5kg/day, C6: thermal safety, C7: pressure).
If not, explain the violation with exact calculations and propose a correction.

Agent A's latest response:
{opponent_response}

Original scenario:
{scenario}

Respond with your analysis and end with a JSON specification block."""


def run_debate(n: int, n_rounds: int, log_dir: str) -> Dict:
    label = "debate_pharma"
    print(f"\n{'─'*65}")
    print(f"  [{label}]  n={n}  rounds={n_rounds}")
    print(f"{'─'*65}")

    adapter_a = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
    adapter_b = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
    out_dir = os.path.join(log_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    modes = []
    for i in range(1, n + 1):
        print(f"  {i:02d}/{n} ", end="", flush=True)
        try:
            # Round 0: independent
            resp_a = adapter_a.client.chat.completions.create(
                model=adapter_a.model, temperature=adapter_a.temperature,
                messages=[{"role": "system", "content": AGENT_A_SYS},
                          {"role": "user", "content": PROMPT_NO_HINT +
                           "\n\nFocus on quality/purity constraints. Show rigorous math."}],
            ).choices[0].message.content

            resp_b = adapter_b.client.chat.completions.create(
                model=adapter_b.model, temperature=adapter_b.temperature,
                messages=[{"role": "system", "content": AGENT_B_SYS},
                          {"role": "user", "content": PROMPT_NO_HINT +
                           "\n\nFocus on process efficiency constraints. Show rigorous math."}],
            ).choices[0].message.content

            # Debate rounds
            for rnd in range(1, n_rounds + 1):
                prompt_a = DEBATE_ROUND_A.format(opponent_response=resp_b, scenario=PROMPT_NO_HINT)
                resp_a = adapter_a.client.chat.completions.create(
                    model=adapter_a.model, temperature=adapter_a.temperature,
                    messages=[{"role": "system", "content": AGENT_A_SYS},
                              {"role": "user", "content": prompt_a}],
                ).choices[0].message.content

                prompt_b = DEBATE_ROUND_B.format(opponent_response=resp_a, scenario=PROMPT_NO_HINT)
                resp_b = adapter_b.client.chat.completions.create(
                    model=adapter_b.model, temperature=adapter_b.temperature,
                    messages=[{"role": "system", "content": AGENT_B_SYS},
                              {"role": "user", "content": prompt_b}],
                ).choices[0].message.content

            # Extract final from Agent A (quality advocate)
            art = extract_json(resp_a)
            failed, _ = uai_check(art)
            mode = classify_pharma(art, failed)
            icon = "\u2705" if mode == "CORRECT" else "\u274c"
            print(f"{icon} {mode}")
            modes.append(mode)

            with open(os.path.join(out_dir, f"trace_{i:02d}.json"), "w") as f:
                json.dump({"run": i, "mode": mode, "failed": failed,
                           "final_a": resp_a[-500:], "final_b": resp_b[-500:]}, f)

        except Exception as e:
            print(f"\U0001f4a5 {e}")
            modes.append("RUNTIME_ERROR")

    dist = {m: modes.count(m) for m in sorted(set(modes))}
    correct_n = dist.get("CORRECT", 0)
    print(f"\n  → correct={correct_n}/{n} ({100*correct_n/n:.0f}%)")
    print(f"  → dist: {dist}")

    return {"label": label, "domain": "pharma", "n": n,
            "correct_n": correct_n, "correct_pct": round(100*correct_n/n, 1),
            "mode_distribution": dist}


# ── Sequential runner ───────────────────────────────────────────────────────

CHECKER_SYS = """You are a pharmaceutical process safety reviewer. Given a proposed
reactor design, check whether ALL 7 constraints are satisfied simultaneously.
Use the exact Arrhenius kinetics: k(T) = 2.5e8 * exp(-72000/(8.314*(T+273.15))),
X = 1-exp(-k*tau), I = 0.35*k^2*tau. If any constraint is violated, explain which
one and suggest a fix. Respond APPROVE or REJECT with reasoning."""

def run_sequential(n: int, max_retries: int, log_dir: str) -> Dict:
    label = "sequential_pharma"
    print(f"\n{'─'*65}")
    print(f"  [{label}]  n={n}  max_retries={max_retries}")
    print(f"{'─'*65}")

    executor = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
    checker = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
    out_dir = os.path.join(log_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    modes = []
    for i in range(1, n + 1):
        print(f"  {i:02d}/{n} ", end="", flush=True)
        try:
            # Initial execution
            resp = executor.completion(PROMPT_NO_HINT, force_json=False)
            raw = resp.get("raw_text", executor.last_response) or ""
            art = extract_json(raw)

            # Retry loop with checker
            for retry in range(max_retries):
                check_prompt = f"""Review this pharmaceutical reactor design proposal:

{raw[-2000:]}

Original constraints:
  C1. Conversion X >= 95%
  C2. Impurity I <= 2%
  C3. Temperature T <= 150°C
  C4. Residence time tau <= 120s
  C5. Production >= 5.0 kg/day
  C6. Heat generation <= cooling capacity (1508 W)
  C7. Pressure drop <= 15 bar

Verify each constraint with exact calculations. APPROVE or REJECT."""

                check_resp = checker.completion(check_prompt, force_json=False)
                check_raw = check_resp.get("raw_text", checker.last_response) or ""

                if "APPROVE" in check_raw.upper():
                    break

                # Rejected — retry with feedback
                retry_prompt = PROMPT_NO_HINT + f"""

[CHECKER FEEDBACK — previous attempt was rejected]
{check_raw[-1000:]}

Please revise your design to address the checker's concerns."""
                resp = executor.completion(retry_prompt, force_json=False)
                raw = resp.get("raw_text", executor.last_response) or ""
                art = extract_json(raw)

            failed, _ = uai_check(art)
            mode = classify_pharma(art, failed)
            icon = "\u2705" if mode == "CORRECT" else "\u274c"
            print(f"{icon} {mode}")
            modes.append(mode)

        except Exception as e:
            print(f"\U0001f4a5 {e}")
            modes.append("RUNTIME_ERROR")

    dist = {m: modes.count(m) for m in sorted(set(modes))}
    correct_n = dist.get("CORRECT", 0)
    print(f"\n  → correct={correct_n}/{n} ({100*correct_n/n:.0f}%)")
    print(f"  → dist: {dist}")

    return {"label": label, "domain": "pharma", "n": n,
            "correct_n": correct_n, "correct_pct": round(100*correct_n/n, 1),
            "mode_distribution": dist}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", f"pharma_baselines_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}\n")

    N = int(os.environ.get("N_TRIALS", 20))

    results = []
    results.append(run_debate(n=N, n_rounds=3, log_dir=log_dir))
    results.append(run_sequential(n=N, max_retries=3, log_dir=log_dir))

    # Summary
    print(f"\n{'='*65}")
    print(f"  {'System':<30} {'Domain':<10} {'n':>4} {'Correct%':>10}")
    print(f"{'='*65}")
    for r in results:
        print(f"  {r['label']:<30} {r['domain']:<10} {r['n']:>4} "
              f"{r['correct_pct']:>9.0f}%")

    out_path = os.path.join(log_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)
    print(f"\n  Results: {out_path}")
    return results


if __name__ == "__main__":
    main()
