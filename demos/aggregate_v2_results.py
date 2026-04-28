"""
Aggregate V2 (and V1) experiment results into a unified paper-ready markdown table.

Scans all `demos/logs/` directories matching known experiment patterns and emits:
  - A full results matrix (monolithic / reasoning / reasoning+UAI / CAAF, by domain)
  - A cost-per-run comparison table (V2 Finding 8: the 100× cost gap)
  - A JSON roll-up with per-cell metadata

V1 data sources (pre-existing runs):
  - logs/fullexp_*           — full AD experiment suite
  - logs/pharma_reactor_*    — full Pharma experiment suite

V2 data sources (smoke_v2 + reasoning_uai_react):
  - logs/smoke_v2_*/smoke_results.json
  - logs/react_uai_{ad,pharma}_*/summary.json

Usage:
    cd "<PROJECT_ROOT>"
    python -m OpenCAAF.demos.aggregate_v2_results
"""
import os, sys, json, glob, statistics
from typing import Dict, Any, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")


def _latest_glob(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── V2 loaders ─────────────────────────────────────────────────────────────

def load_smoke_v2_latest() -> Dict[str, Dict[str, Any]]:
    """Scan all smoke_v2_* directories, take the most recent `smoke_results.json`
    for each smoke ID. Prefers `_patched.json` over the original when present.
    Returns {smoke_id: summary}."""
    picks: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for d in sorted(glob.glob(os.path.join(LOGS_DIR, "smoke_v2_*"))):
        ts = os.path.basename(d).replace("smoke_v2_", "")
        # Prefer patched file if it exists
        patched = os.path.join(d, "smoke_results_patched.json")
        original = os.path.join(d, "smoke_results.json")
        path = patched if os.path.exists(patched) else original
        data = _load_json(path)
        if not data:
            continue
        for sid, summary in (data.get("results") or {}).items():
            if not summary:
                continue
            n_valid = summary.get("n_valid", summary.get("n", 0))
            mdist = summary.get("status_distribution", {}) or summary.get("mode_distribution", {})
            valid_trials = n_valid - (mdist.get("RUNTIME_ERROR", 0) if "n_valid" not in summary else 0)
            if valid_trials <= 0:
                continue
            if sid not in picks or ts > picks[sid][0]:
                picks[sid] = (ts, summary)
    return {sid: s for sid, (_, s) in picks.items()}


def load_react_uai_latest() -> Dict[str, Dict[str, Any]]:
    """Scan react_uai_{ad,pharma}_* directories. For each domain, take the most
    recent COMPLETED run (one with summary.json present and n_valid > 0).
    A run still in progress (no summary.json yet) is skipped, falling back to
    the previous complete one."""
    out: Dict[str, Dict[str, Any]] = {}
    for domain in ("ad", "pharma"):
        candidates = sorted(glob.glob(os.path.join(LOGS_DIR, f"react_uai_{domain}_*")), reverse=True)
        for d in candidates:
            summary = _load_json(os.path.join(d, "summary.json"))
            if summary and (summary.get("n_valid", summary.get("n", 0)) > 0):
                out[domain] = summary
                break
    return out


# ── V1 loaders ─────────────────────────────────────────────────────────────

def load_v1_latest() -> Dict[str, Any]:
    """V1 AD + Pharma primary results.

    Pharma was run multiple times: a primary 4-condition run and add-on runs
    for Gemma-3-12B and Command-R7B. We merge cells across all dirs, keeping
    the latest n>=20 entry for each unique label.
    """
    v1: Dict[str, Any] = {"ad": None, "pharma_conditions": []}

    ad = _latest_glob(os.path.join(LOGS_DIR, "fullexp_*/results.json"))
    if ad:
        v1["ad"] = _load_json(ad)
        v1["_ad_path"] = ad

    # Merge all pharma dirs by label, prefer largest n.
    merged: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "pharma_reactor_*/results.json"))):
        ts = path.split("/")[-2]
        d = _load_json(path)
        if not d:
            continue
        for c in d.get("conditions", []):
            label = c.get("label", "")
            if not label:
                continue
            n = c.get("n", 0)
            if label not in merged or n > merged[label][1].get("n", 0):
                merged[label] = (ts, c)
    v1["pharma_conditions"] = [c for _, c in merged.values()]
    v1["_pharma_paths"] = sorted({ts for ts, _ in merged.values()})
    return v1


# ── Table builders ─────────────────────────────────────────────────────────

def _fmt_pct(n: Optional[int], total: Optional[int]) -> str:
    if n is None or not total:
        return "—"
    return f"{100*n/total:.0f}% ({n}/{total})"


def build_unified_table(v1: Dict, smoke: Dict, react: Dict) -> str:
    lines: List[str] = []
    lines.append("## V2 Unified Results Matrix\n")
    lines.append("| # | Category | System | Model | Domain | n | Correct% | $/run | Source |")
    lines.append("|---|----------|--------|-------|--------|---|----------|-------|--------|")

    row_idx = [0]
    def add(category, system, model, domain, n, correct_pct, cost_per_run, source):
        row_idx[0] += 1
        cost_str = f"${cost_per_run:.4f}" if isinstance(cost_per_run, (int, float)) and cost_per_run > 0 else "—"
        lines.append(
            f"| {row_idx[0]} | {category} | {system} | {model} | {domain} | {n} "
            f"| **{correct_pct}** | {cost_str} | {source} |"
        )

    # ── Monolithic standard (V1 data) ──
    if v1.get("ad"):
        per = v1["ad"].get("conditions", [])
        for r in per:
            label = r.get("label", "")
            if "mono" in label.lower() and "_uai" not in label:
                n = r.get("n", 0)
                cp = r.get("correct_pct")
                cost_per_run = (r.get("cost_usd", 0) / n) if n else 0
                add(
                    "Monolithic",
                    f"Mono ({label})",
                    r.get("model", ""),
                    "AD",
                    n, f"{cp:.0f}%" if isinstance(cp, (int, float)) else "?",
                    cost_per_run, "V1",
                )

    for r in v1.get("pharma_conditions", []):
        label = r.get("label", "")
        if "mono_mini_nohint" in label and "uai" not in label:
            n = r.get("n", 0)
            cp = r.get("correct_pct")
            cost_per_run = (r.get("cost_usd", 0) / n) if n else 0
            add(
                "Monolithic",
                f"Mono ({label})",
                r.get("model", ""),
                "Pharma",
                n, f"{cp:.0f}%" if isinstance(cp, (int, float)) else "?",
                cost_per_run, "V1",
            )

    # ── Reasoning monolithic (V2 Smoke B + D) ──
    sb = smoke.get("smoke_B")
    if sb:
        cpr = (sb["total_cost_usd"] / sb["n"]) if sb.get("n") else 0
        add("Reasoning Mono", "Opus-4 thinking (adaptive, high)", "claude-opus-4-7",
            "AD", sb["n"], f"{sb['correct_pct']:.0f}%", cpr, "V2 smoke B")
    sd = smoke.get("smoke_D")
    if sd:
        cpr = (sd["total_cost_usd"] / sd["n"]) if sd.get("n") else 0
        add("Reasoning Mono", "Opus-4 thinking (adaptive, high)", "claude-opus-4-7",
            "Pharma", sd["n"], f"{sd['correct_pct']:.0f}%", cpr, "V2 smoke D")

    # ── Reasoning + UAI (ReAct) — V2 Phase 3.5 ──
    for dom, data in react.items():
        n = data.get("n_valid", data.get("n", 0))
        if n <= 0:
            continue
        cpr = (data.get("total_cost_usd", 0) / n) if n else 0
        domain_label = "AD" if dom == "ad" else "Pharma"
        model = data.get("model", "")
        effort = data.get("thinking_effort") or "none"
        excl_note = f" (excl={data['excluded_n']})" if data.get("excluded_n") else ""
        add(
            "Reasoning + UAI (ReAct)",
            f"Mono+UAI (tool-call){excl_note}",
            f"{model} / effort={effort}",
            domain_label,
            n, f"{data.get('correct_pct', 0):.0f}%", cpr,
            "V2 react_uai",
        )

    # ── V1 Mono+UAI (prompt-simulated) ──
    for r in v1.get("pharma_conditions", []):
        if "mono_uai" in r.get("label", ""):
            n = r.get("n", 0)
            cp = r.get("correct_pct")
            cost_per_run = (r.get("cost_usd", 0) / n) if n else 0
            add(
                "Reasoning + UAI (prompt-sim)",
                f"Mono+UAI-prompt ({r.get('label')})",
                r.get("model", ""),
                "Pharma",
                n, f"{cp:.0f}%" if isinstance(cp, (int, float)) else "?",
                cost_per_run, "V1",
            )

    # ── CAAF executors (V1 + V2) ──
    if v1.get("ad"):
        for r in v1["ad"].get("conditions", []):
            if "caaf" in r.get("label", "").lower():
                n = r.get("n", 0)
                cp = r.get("correct_pct")
                cost_per_run = (r.get("cost_usd", 0) / n) if n else 0
                add(
                    "CAAF",
                    f"CAAF ({r.get('label')})",
                    r.get("model", "gpt-4o-mini"),
                    "AD",
                    n, f"{cp:.0f}%" if isinstance(cp, (int, float)) else "?",
                    cost_per_run, "V1",
                )
    for r in v1.get("pharma_conditions", []):
        if "caaf" in r.get("label", "").lower():
            n = r.get("n", 0)
            cp = r.get("correct_pct")
            cost_per_run = (r.get("cost_usd", 0) / n) if n else 0
            add(
                "CAAF",
                f"CAAF ({r.get('label')})",
                r.get("model", "gpt-4o-mini"),
                "Pharma",
                n, f"{cp:.0f}%" if isinstance(cp, (int, float)) else "?",
                cost_per_run, "V1",
            )

    # V2 CAAF executors
    smoke_to_cell = [
        ("smoke_A", "claude-haiku-4-5", "AD"),
        ("smoke_C", "claude-haiku-4-5", "Pharma"),
        ("smoke_F", "qwen/qwen3-14b",    "Pharma"),
    ]
    for sid, model, dom in smoke_to_cell:
        s = smoke.get(sid)
        if not s:
            continue
        # Patched-schema runs use n_valid; legacy use n.
        n = s.get("n_valid", s.get("n", 0))
        if n <= 0:
            continue
        # Skip if all runtime errors
        mdist = s.get("status_distribution", {}) or s.get("mode_distribution", {})
        if "n_valid" not in s and mdist.get("RUNTIME_ERROR", 0) >= n:
            continue
        cpr = (s["total_cost_usd"] / n) if n else 0
        excl_note = f" (excl={s['excluded_n']})" if s.get("excluded_n") else ""
        add("CAAF", f"CAAF all-{model.split('/')[-1]}{excl_note}", model, dom,
            n, f"{s['correct_pct']:.0f}%", cpr, f"V2 {sid}")

    return "\n".join(lines)


def build_cost_comparison(smoke: Dict, react: Dict, v1: Dict) -> str:
    """Finding 8: the cost argument. 100% runs only, Pharma preferred."""
    lines = []
    lines.append("\n## V2 Finding 8: Cost Comparison (Pharma at 100% correct)\n")
    lines.append("| System | Model | Correct% | Iters | $/run | Relative to CAAF-mini |")
    lines.append("|--------|-------|----------|-------|-------|-----------------------|")

    rows: List[Tuple[str, str, float, Optional[float], float]] = []  # (system, model, pct, iters, $)

    # V1 CAAF all-gpt-4o-mini Pharma (baseline)
    caaf_mini_cost = None
    for r in v1.get("pharma_conditions", []):
        if "3_caaf_mini" in r.get("label", ""):
            n = r.get("n", 0)
            cpr = (r.get("cost_usd", 0) / n) if n else 0
            caaf_mini_cost = cpr
            rows.append(("CAAF all-gpt-4o-mini (V1)", "gpt-4o-mini", r["correct_pct"], None, cpr))

    # V1 Mono+UAI prompt-sim
    for r in v1.get("pharma_conditions", []):
        if "mono_uai" in r.get("label", ""):
            n = r.get("n", 0)
            cpr = (r.get("cost_usd", 0) / n) if n else 0
            rows.append(("Mono+UAI (prompt-sim)", r.get("model", "gpt-4o-mini"), r["correct_pct"], None, cpr))

    # V1 CAAF open-weight rerun (Gemma-3-12B, Command-R7B)
    for r in v1.get("pharma_conditions", []):
        if "openrouter" in r.get("label", "") and "caaf" in r.get("label", ""):
            n = r.get("n", 0)
            cpr = (r.get("cost_usd", 0) / n) if n else 0
            model_short = r["label"].replace("3_caaf_openrouter_", "").replace("_nohint", "")
            rows.append((f"CAAF open-weight ({model_short})", r.get("model", model_short),
                         r["correct_pct"], None, cpr))

    # V2 Haiku CAAF Pharma (smoke C)
    s = smoke.get("smoke_C")
    if s:
        n = s.get("n_valid", s.get("n", 0))
        if n:
            rows.append(("CAAF all-Haiku-4.5 (V2)", "claude-haiku-4-5", s["correct_pct"],
                         None, s["total_cost_usd"]/n))

    # V2 Reasoning+UAI ReAct Pharma (U3)
    if "pharma" in react:
        r = react["pharma"]
        n = r.get("n_valid", r.get("n", 0))
        if n:
            iters = None
            if r.get("per_run"):
                ii = [pr.get("n_iters") for pr in r["per_run"] if isinstance(pr.get("n_iters"), int)]
                if ii:
                    iters = statistics.mean(ii)
            rows.append(("Mono+UAI ReAct (V2)",
                         f"{r.get('model')} / effort={r.get('thinking_effort')}",
                         r["correct_pct"], iters, r["total_cost_usd"]/n))

    for sys_name, model, pct, iters, cpr in rows:
        rel = f"{cpr/caaf_mini_cost:.0f}×" if caaf_mini_cost and caaf_mini_cost > 0 else "—"
        iters_str = f"{iters:.1f}" if iters else "—"
        lines.append(
            f"| {sys_name} | {model} | {pct:.0f}% | {iters_str} | ${cpr:.4f} | {rel} |"
        )
    return "\n".join(lines)


def main():
    v1 = load_v1_latest()
    smoke = load_smoke_v2_latest()
    react = load_react_uai_latest()

    print("# CAAF V2 Aggregated Results")
    print(f"\n_Sources scanned: {len(glob.glob(os.path.join(LOGS_DIR, 'smoke_v2_*')))} smoke_v2 runs, "
          f"{len(glob.glob(os.path.join(LOGS_DIR, 'react_uai_*')))} react_uai runs, "
          f"{len(glob.glob(os.path.join(LOGS_DIR, 'fullexp_*')))} V1 AD runs, "
          f"{len(glob.glob(os.path.join(LOGS_DIR, 'pharma_reactor_*')))} V1 pharma runs._\n")

    print(build_unified_table(v1, smoke, react))
    print(build_cost_comparison(smoke, react, v1))

    # Also dump a JSON roll-up
    out_path = os.path.join(LOGS_DIR, "_v2_aggregate.json")
    with open(out_path, "w") as f:
        json.dump({
            "v1": {"ad_path": v1.get("_ad_path"), "pharma_path": v1.get("_pharma_path")},
            "smoke": smoke,
            "react": react,
        }, f, indent=2, default=str)
    print(f"\n_Aggregate JSON: {out_path}_")


if __name__ == "__main__":
    main()
