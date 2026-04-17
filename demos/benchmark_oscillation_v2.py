"""
Section 6: Stochastic Oscillation vs. Monotonic Convergence benchmark — n=20 per condition.
Tests whether naive reflection loops can converge vs. CAAF's state-locking guarantee.

Usage:
    cd "/path/to/Agent_Playground"
    python -m OpenCAAF.demos.benchmark_oscillation_v2
"""
import os, sys, json, time, re, statistics
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()

from OpenCAAF.adapters.openai_adapter import OpenAIAdapter
from OpenCAAF.harness.engine import HarnessRegistry, AssertionEngine
from OpenCAAF.engine.orchestrator import OpenCAAFOrchestrator

PHYSICS = {
    "vehicle_speed_kmph_t0": 120, "transition_window_seconds": 5,
    "m_per_sec_to_km_per_h_factor": 3.6, "road_friction_mu": 0.4,
    "g": 9.8, "max_deceleration_limit": 2.0, "perception_range_limit": 30,
}

REQUEST = """
[ROLE] Lead Autonomous Driving System Architect

[TASK] Design the "Degradation State Machine" for an L3 autonomous driving function.
This is an OFFLINE REQUIREMENTS ENGINEERING task.

[SCENARIO]
Highway cruise at 120 km/h. Torrential rain (80mm/h) reduces sensor perception range to 30m.
Transition window: 5 seconds.

[SAFETY CONSTRAINTS — both must be satisfied simultaneously]
  A. Forward Safety: ((v_t5/3.6)^2) / (2 × 0.4 × 9.8) < 30  → v_t5 ≤ 55 km/h
  B. Rear Safety: (120 - v_t5) / (5 × 3.6) ≤ 2.0 m/s²      → v_t5 ≥ 84 km/h

[OUTPUT — end with this exact JSON block]
{
  "safe_state_definition": "str",
  "vehicle_speed_kmph_t5": int,
  "decision_reasoning": "str",
  "constraint_relaxed": "str or null"
}
"""

FWD  = "FORWARD_COLLISION_PREVENTION_PERCEPTION"
REAR = "REAR_COLLISION_PREVENTION_DECELERATION"


def extract_json(text):
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    for m in reversed(list(re.finditer(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL))):
        try: return json.loads(m.group(1))
        except: continue
    return None


def uai_check(artifact):
    if artifact is None:
        return ["JSON_FAILURE"], True
    full = {**PHYSICS, **{k: v for k, v in artifact.items() if v is not None}}
    rules = HarnessRegistry().load_harness("ad_degradation")
    failed = [r.id for r in rules if AssertionEngine.check(r, full)]
    return failed, len(failed) > 0


def run_naive_single(max_iters=5):
    """Run one naive reflection trial. Returns trajectory of (speed, failed_rules)."""
    adapter = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
    prompt = REQUEST
    trajectory = []

    for i in range(1, max_iters + 1):
        resp = adapter.completion(prompt, force_json=False)
        raw = resp.get("raw_text", adapter.last_response) or ""
        art = extract_json(raw)
        failed, uai_hit = uai_check(art)
        speed = art.get("vehicle_speed_kmph_t5") if art else None
        trajectory.append({"iter": i, "speed": speed, "failed": failed})

        if not uai_hit:
            break  # accidentally converged

        # Build reflection feedback
        fb = "\n\n[SYSTEM FEEDBACK] Your previous output FAILED these constraints:\n"
        if FWD in failed:
            fb += "  - FORWARD: stopping distance exceeds 30m — speed too high\n"
        if REAR in failed:
            fb += "  - REAR: deceleration exceeds 2.0 m/s² — speed too low\n"
        if "JSON_FAILURE" in failed:
            fb += "  - OUTPUT: could not parse JSON block\n"
        fb += "Correct these errors. Output updated JSON.\n"
        prompt = REQUEST + fb

    return trajectory


def run_naive_batch(n, log_dir):
    print(f"\n  [Naive gpt-4o-mini | oscillation] n={n}")
    results = []
    for i in range(1, n + 1):
        print(f"    {i:02d}/{n} ", end="", flush=True)
        try:
            traj = run_naive_single(max_iters=5)
            speeds = [s["speed"] for s in traj]
            final = traj[-1]
            converged = not final["failed"]
            print(f"speeds={speeds}  converged={converged}  final_failed={final['failed']}")
            results.append({"run": i, "trajectory": traj,
                            "converged": converged, "final_speed": final["speed"]})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"run": i, "error": str(e)})
    with open(os.path.join(log_dir, "naive_oscillation.jsonl"), "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    return results


def run_caaf_batch(n, log_dir):
    print(f"\n  [CAAF-all-mini | monotonic] n={n}")
    results = []
    for i in range(1, n + 1):
        print(f"    {i:02d}/{n} ", end="", flush=True)
        try:
            executor = OpenAIAdapter(model="gpt-4o-mini", temperature=0.7)
            reviewer = OpenAIAdapter(model="gpt-4o-mini", temperature=0.0)
            orch = OpenCAAFOrchestrator(
                executor_adapter=executor, reviewer_adapter=reviewer,
                exact_log_dir=os.path.join(log_dir, f"caaf_run{i:02d}"))
            tree = orch.run_full_pipeline(REQUEST, domain_id="ad_degradation", interactive=False)
            status = tree.metadata.get("integration_status", "UNKNOWN")
            correct = status == "FAILED_PARADOX"
            print(f"status={status}")
            results.append({"run": i, "status": status, "correct": correct})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"run": i, "error": str(e)})
    with open(os.path.join(log_dir, "caaf_monotonic.jsonl"), "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    return results


def summarise_naive(results):
    converged = sum(1 for r in results if r.get("converged"))
    oscillated = 0
    stagnated = 0
    for r in results:
        traj = r.get("trajectory", [])
        speeds = [t["speed"] for t in traj if t.get("speed") is not None]
        if len(speeds) >= 3:
            diffs = [abs(speeds[j] - speeds[j-1]) for j in range(1, len(speeds))]
            if max(diffs) > 10:
                oscillated += 1
            elif speeds[-1] == speeds[-2] if len(speeds) >= 2 else False:
                stagnated += 1
    # Collect all speed trajectories
    all_trajs = [r.get("trajectory", []) for r in results]
    return {
        "n": len(results),
        "converged": converged,
        "oscillated_or_stagnated": len(results) - converged,
        "speed_trajectories": [[t["speed"] for t in traj] for traj in all_trajs],
    }


def summarise_caaf(results):
    correct = sum(1 for r in results if r.get("correct"))
    return {"n": len(results), "paradox_detected": correct,
            "paradox_pct": 100 * correct // len(results)}


def main():
    N = 20
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs", f"oscillation_v2_{timestamp}_n{N}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SECTION 6 — Oscillation vs Convergence Benchmark  (n={N})")
    print(f"  Output: {log_dir}")
    print(f"{'='*65}")

    naive_res = run_naive_batch(N, log_dir)
    caaf_res  = run_caaf_batch(N, log_dir)

    s_naive = summarise_naive(naive_res)
    s_caaf  = summarise_caaf(caaf_res)

    print(f"\n{'='*75}")
    print("  OSCILLATION RESULTS")
    print(f"{'='*75}")
    print(f"  Naive gpt-4o-mini | converged={s_naive['converged']}/{s_naive['n']} | oscillated/stagnated={s_naive['oscillated_or_stagnated']}/{s_naive['n']}")
    print(f"  Speed trajectories (per run):")
    for j, traj in enumerate(s_naive["speed_trajectories"], 1):
        print(f"    Run {j}: {traj}")
    print(f"  CAAF-all-mini     | paradox_detected={s_caaf['paradox_detected']}/{s_caaf['n']} ({s_caaf['paradox_pct']}%)")

    summary = {"n": N, "naive": s_naive, "caaf": s_caaf}
    out = os.path.join(log_dir, "oscillation_summary.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")
    return summary


if __name__ == "__main__":
    main()
