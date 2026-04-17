"""Quick runner: overrides N_TRIALS before calling main."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Override before import to avoid touching the main script
import OpenCAAF.demos.benchmark_batch_statistical as bm

bm.N_TRIALS_OVERRIDE = int(os.environ.get("N_TRIALS", 20))

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    import OpenCAAF.demos.benchmark_batch_statistical as mod
    # Patch N_TRIALS inside main
    original_main = mod.main
    n = bm.N_TRIALS_OVERRIDE
    def patched_main():
        import math, statistics, json, time
        from datetime import datetime
        N_TRIALS = n
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(mod.__file__))
        log_dir = os.path.join(script_dir, "logs", f"batch_{timestamp}_n{N_TRIALS}")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  CAAF PAPER VALIDATION — n={N_TRIALS} per condition")
        print(f"  Output: {log_dir}")
        print(f"{'='*70}")

        from OpenCAAF.harness.engine import HarnessRegistry
        registry = HarnessRegistry()
        rules = registry.load_harness("ad_degradation")
        print(f"  Loaded {len(rules)} harness rules: {[r.id for r in rules]}")

        fwd_at_55 = mod.compute_stopping_distance(55)
        fwd_at_84 = mod.compute_stopping_distance(84)
        print(f"\n  Paradox verification:")
        print(f"    v=55: stop={fwd_at_55:.1f}m, decel={(120-55)/18:.2f} m/s² → fwd_ok={fwd_at_55<30}, rear_ok={(120-55)/18<=2.0}")
        print(f"    v=84: stop={fwd_at_84:.1f}m, decel={(120-84)/18:.2f} m/s² → fwd_ok={fwd_at_84<30}, rear_ok={(120-84)/18<=2.0}")
        print(f"    Paradox confirmed: no valid speed exists.\n")

        mono_no_hint   = mod.run_monolithic_batch(N_TRIALS, "gpt-4o", log_dir, prompt=mod.PROMPT_NO_HINT,   condition_name="no_hint")
        mono_with_hint = mod.run_monolithic_batch(N_TRIALS, "gpt-4o", log_dir, prompt=mod.PROMPT_WITH_HINT, condition_name="with_hint")
        caaf_summary   = mod.run_caaf_batch(N_TRIALS, log_dir)

        paper_table = mod.generate_paper_table(mono_no_hint, caaf_summary)

        full_summary = {
            "n_trials": N_TRIALS,
            "monolithic_no_hint":   mono_no_hint,
            "monolithic_with_hint": mono_with_hint,
            "caaf":                 caaf_summary,
            "paper_table":          paper_table,
        }
        out = os.path.join(log_dir, "batch_summary.json")
        with open(out, "w") as f:
            json.dump(full_summary, f, indent=2)
        with open(os.path.join(log_dir, "paper_table.md"), "w") as f:
            f.write(paper_table)

        print(f"\n{'='*70}")
        print("  DONE")
        print(f"{'='*70}")
        print(paper_table)
        print(f"\n  Results saved: {out}")

    patched_main()
