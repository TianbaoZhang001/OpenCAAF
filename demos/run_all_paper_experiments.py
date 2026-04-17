"""
CAAF Paper — Master Experiment Runner
======================================
Runs ALL experiments needed for the revised paper in sequence.

Usage:
    cd "/path/to/Agent_Playground"
    python -m OpenCAAF.demos.run_all_paper_experiments

Estimated total API cost: ~$2.50-4.00 (GPT-4o-mini dominated)
Estimated wall time: ~60-90 minutes (sequential API calls)

Experiments:
  1. Cloud Infrastructure SLA domain (3 conditions × n=20)
  2. Expanded Context Rot (4 conditions × n=20)
  3. Expanded Oscillation + Naive Reflection (2 conditions × n=20)
  4. Multi-Agent Debate baseline (2 domains × n=20)
  5. Sequential/LangGraph baseline (2 domains × n=20)
"""
import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_experiment(name: str, module_path: str):
    """Import and run one experiment module's main()."""
    print(f"\n{'='*70}")
    print(f"  STARTING: {name}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_path)
        result = mod.main()
        elapsed = time.time() - t0
        print(f"\n  ✅ {name} completed in {elapsed/60:.1f} min")
        return {"name": name, "status": "OK", "elapsed_min": round(elapsed/60, 1), "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ {name} FAILED after {elapsed/60:.1f} min: {e}")
        return {"name": name, "status": "FAILED", "elapsed_min": round(elapsed/60, 1), "error": str(e)}


def main():
    start = time.time()
    print(f"\nCAAF Paper — Full Experiment Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated cost: ~$2.50-4.00")
    print(f"Estimated time: ~60-90 minutes\n")

    experiments = [
        ("Cloud Infra SLA Domain (n=20, 3 conds)",
         "OpenCAAF.demos.benchmark_cloud_infra"),
        ("Context Rot Expanded (n=20, 4 conds)",
         "OpenCAAF.demos.benchmark_context_rot_v2"),
        ("Oscillation Expanded (n=20, 2 conds)",
         "OpenCAAF.demos.benchmark_oscillation_v2"),
        ("Debate Baseline (n=20, 2 domains)",
         "OpenCAAF.demos.benchmark_debate_baseline"),
        ("Sequential Baseline (n=20, 2 domains)",
         "OpenCAAF.demos.benchmark_sequential_baseline"),
    ]

    results = []
    for name, module in experiments:
        r = run_experiment(name, module)
        results.append(r)

    total_min = (time.time() - start) / 60

    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_min:.1f} min")
    print(f"{'='*70}")
    for r in results:
        icon = "✅" if r["status"] == "OK" else "❌"
        print(f"  {icon} {r['name']}: {r['status']} ({r['elapsed_min']:.1f} min)")
    print(f"{'='*70}\n")

    # Save summary
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(script_dir, "logs", f"all_experiments_{timestamp}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"timestamp": timestamp, "total_min": round(total_min, 1),
                    "experiments": [{k: v for k, v in r.items() if k != "result"} for r in results]},
                   f, indent=2)
    print(f"  Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
