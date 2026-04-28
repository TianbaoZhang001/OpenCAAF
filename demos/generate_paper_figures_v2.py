"""
Regenerate fig10, fig11, fig12 for the V2.2 arxiv paper, pulling from the
consolidated data sources:

  - V1 raw Pharma cells: `pharma_reactor_20260416_163225/` (+ patched V2
    rerun at `patch_v1_json_*/` for JSON-failed slots)
  - V1 Pharma baselines: `pharma_baselines_20260417_001955/` (patched by
    the same patch run)
  - Mono+UAI prompt-sim rerun: `pharma_uai_rerun_20260416_234533/result.json`
  - V2 aggregate (Opus thinking, Haiku CAAF, etc.): `_v2_aggregate.json`
  - Mono+UAI true tool-call Opus: `react_uai_pharma_20260423_080612/`
  - Mono+UAI true tool-call Haiku: `react_uai_pharma_20260423_112426/`
  - V1 AD full: `fullexp_20260411_171353/results_n30.json`

Figures written to `CAAF_paper/paper_v2/caaf_arxiv_v2/figures/` overwriting
the existing V1 files (fig10, fig11, fig12). Color palette matches the
archive style (Slate / Amber / Emerald) extended with purple/pink for
Mono+UAI variants and a darker green for frontier-reasoning cells.

Usage:
    cd "<PROJECT_ROOT>"
    python -m OpenCAAF.demos.generate_paper_figures_v2 [--patch-dir PATH]
"""
import os, sys, json, argparse, glob
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demos", "logs")
FIG_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "CAAF_paper", "paper_v2", "caaf_arxiv_v2", "figures"
)

# ── Color palette (aligned with fig7_failure_modes) ────────────────────────

# Palette aligned with fig7 (Flat UI). Extracted by sampling fig7's PDF:
#   Correct: #27ae60 (Nephritis); Forward: #e67e22 (Carrot);
#   Rear:    #e74c3c (Alizarin);  Forced:  #9b59b6 (Amethyst);
#   Text:    #2c3e50 (Midnight Blue).

# Failure-mode categorical colors (fig7-native)
COL_CORRECT = "#27ae60"    # Nephritis green
COL_CONV    = "#e67e22"    # Carrot orange (fig7 Forward Safety analogue)
COL_IMP     = "#e74c3c"    # Alizarin red  (fig7 Rear Safety analogue)
COL_OSC     = "#9b59b6"    # Amethyst purple (fig7 Forced Solution analogue)
COL_DUAL    = "#d35400"    # Pumpkin (darker orange — compound C1+C2 violation)
COL_JSON    = "#bdc3c7"    # Silver — parse failure (should not appear post-retry)
COL_OTHER   = "#95a5a6"    # Concrete gray — runtime error / other

# Architecture-family colors (used in fig11/fig12). CAAF maps to the Correct
# green so that 'full-green' bars read identically across figures. Mono+UAI
# maps to the Forced-Solution purple (UAI without architectural enforcement
# is a partial-success bucket). Monolithic / multi-agent baselines collapse
# to neutral gray since they carry no outcome information by themselves.
COL_MONO      = "#95a5a6"   # Concrete
COL_FRONTIER  = "#7f8c8d"   # Asbestos (darker concrete)
COL_DEBATE    = "#95a5a6"   # Concrete
COL_SEQ       = "#7f8c8d"   # Asbestos
COL_MONO_UAI  = "#9b59b6"   # Amethyst
COL_MONO_UAI_T= "#8e44ad"   # Wisteria (darker amethyst)
COL_CAAF      = "#27ae60"   # Nephritis (same as Correct)
COL_CAAF_ALT  = "#1e8449"   # Darker nephritis

# Matplotlib defaults to match fig7 look
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
})


# ── Data loaders ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tally_modes(rows: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        if r.get("_excluded"):
            continue
        m = r.get("mode", "UNKNOWN")
        out[m] = out.get(m, 0) + 1
    return out


def find_latest_patch_dir() -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(LOG_BASE, "patch_v1_json_*")))
    return cands[-1] if cands else None


def load_pharma_cond1(patch_dir: Optional[str]) -> Dict:
    """Mono GPT-4o-mini Pharma no-hint. Merge original jsonl with patched slots."""
    src = os.path.join(LOG_BASE, "pharma_reactor_20260416_163225", "1_mono_mini_nohint_runs.jsonl")
    patched = os.path.join(patch_dir, "1_mono_mini_nohint_runs_patched.jsonl") if patch_dir else None
    rows = load_jsonl(patched if patched and os.path.exists(patched) else src)
    modes = tally_modes(rows)
    n_valid = sum(1 for r in rows if not r.get("_excluded"))
    correct = modes.get("CORRECT", 0)
    return {"label": "Mono mini no-hint", "n": n_valid, "correct": correct,
            "correct_pct": 100 * correct / max(1, n_valid), "modes": modes, "rows": rows}


def load_pharma_cond2(patch_dir: Optional[str]) -> Dict:
    """Mono GPT-4o-mini Pharma with-hint."""
    src = os.path.join(LOG_BASE, "pharma_reactor_20260416_163225", "2_mono_mini_hint_runs.jsonl")
    patched = os.path.join(patch_dir, "2_mono_mini_hint_runs_patched.jsonl") if patch_dir else None
    rows = load_jsonl(patched if patched and os.path.exists(patched) else src)
    modes = tally_modes(rows)
    n_valid = sum(1 for r in rows if not r.get("_excluded"))
    correct = modes.get("CORRECT", 0)
    return {"label": "Mono mini +hint", "n": n_valid, "correct": correct,
            "correct_pct": 100 * correct / max(1, n_valid), "modes": modes, "rows": rows}


def load_pharma_cond3_caaf() -> Dict:
    """CAAF all-mini Pharma (V1)."""
    src = os.path.join(LOG_BASE, "pharma_reactor_20260416_163225", "3_caaf_mini_nohint_runs.jsonl")
    rows = load_jsonl(src)
    correct = sum(1 for r in rows if r.get("correct") or r.get("mode") == "CORRECT")
    n_valid = len(rows)
    return {"label": "CAAF all-mini", "n": n_valid, "correct": correct,
            "correct_pct": 100 * correct / max(1, n_valid),
            "modes": {"CORRECT": correct} if correct == n_valid else tally_modes(rows),
            "rows": rows}


def load_pharma_cond4_mono_uai() -> Dict:
    """Mono+UAI prompt-sim Pharma (V1 rerun)."""
    path = os.path.join(LOG_BASE, "pharma_uai_rerun_20260416_234533", "result.json")
    with open(path) as f:
        d = json.load(f)
    return {"label": "Mono+UAI prompt-sim", "n": d["n"], "correct": d["correct_n"],
            "correct_pct": d["correct_pct"], "modes": d["mode_distribution"]}


def load_v2_aggregate() -> Dict:
    with open(os.path.join(LOG_BASE, "_v2_aggregate.json")) as f:
        return json.load(f)


def load_react(domain: str = "pharma") -> Dict:
    """Most recent Mono+UAI true-tool-call (Opus) for the given domain."""
    agg = load_v2_aggregate()
    r = agg["react"][domain]
    modes = r.get("mode_distribution", {})
    return {"label": "Mono+UAI true tool-call (Opus)", "n": r["n_valid"], "correct": r["correct"],
            "correct_pct": r["correct_pct"], "modes": modes}


def load_haiku_true_tool_call() -> Dict:
    """Mono+UAI true-tool-call with Haiku 4.5 (V2.2 ablation)."""
    path = os.path.join(LOG_BASE, "react_uai_pharma_20260423_112426", "summary.json")
    with open(path) as f:
        d = json.load(f)
    return {"label": "Mono+UAI true tool-call (Haiku)", "n": d["n_valid"], "correct": d["correct"],
            "correct_pct": d["correct_pct"], "modes": d["mode_distribution"]}


def load_smoke_D() -> Dict:
    """Mono Opus 4 thinking Pharma (V2.1)."""
    agg = load_v2_aggregate()
    s = agg["smoke"]["smoke_D"]
    return {"label": "Mono Opus thinking", "n": s["n"], "correct": s["correct"],
            "correct_pct": s["correct_pct"], "modes": s.get("mode_distribution", {})}


def load_smoke_C_caaf_haiku() -> Dict:
    """CAAF all-Haiku-4.5 Pharma (V2.1)."""
    agg = load_v2_aggregate()
    s = agg["smoke"]["smoke_C"]
    correct = s.get("correct", 0)
    return {"label": "CAAF all-Haiku", "n": s.get("n", s.get("n_valid", 20)),
            "correct": correct, "correct_pct": s.get("correct_pct", 100 * correct / max(1, s.get("n", 20))),
            "modes": {"CORRECT": correct}}


def _load_targeted_summary(patch_dir: str) -> Optional[Dict]:
    p = os.path.join(patch_dir, "targeted_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_debate_pharma(patch_dir: Optional[str]) -> Dict:
    """Debate baseline Pharma — patched (V2 rerun) if available, else V1."""
    if patch_dir:
        ts = _load_targeted_summary(patch_dir)
        if ts and "patches" in ts and "debate_pharma" in ts["patches"]:
            p = ts["patches"]["debate_pharma"]
            return {"label": "Debate Pharma (v2)",
                    "n": p["n_valid"], "correct": p["correct_n"],
                    "correct_pct": 100 * p["correct_n"] / max(1, p["n_valid"]),
                    "modes": p["mode_distribution"]}
    # Fallback: original V1
    path = os.path.join(LOG_BASE, "pharma_baselines_20260417_001955", "results.json")
    with open(path) as f:
        d = json.load(f)
    for r in d["results"]:
        if r["label"] == "debate_pharma":
            return {"label": "Debate Pharma", "n": r["n"], "correct": r["correct_n"],
                    "correct_pct": r["correct_pct"], "modes": r["mode_distribution"]}


def load_sequential_pharma(patch_dir: Optional[str]) -> Dict:
    if patch_dir:
        ts = _load_targeted_summary(patch_dir)
        if ts and "patches" in ts and "sequential_pharma" in ts["patches"]:
            p = ts["patches"]["sequential_pharma"]
            # Filter EXCLUDED from mode distribution for display
            modes = {k: v for k, v in p["mode_distribution"].items() if k != "EXCLUDED"}
            return {"label": "Sequential Pharma (v2)",
                    "n": p["n_valid"], "correct": p["correct_n"],
                    "correct_pct": 100 * p["correct_n"] / max(1, p["n_valid"]),
                    "modes": modes}
    path = os.path.join(LOG_BASE, "pharma_baselines_20260417_001955", "results.json")
    with open(path) as f:
        d = json.load(f)
    for r in d["results"]:
        if r["label"] == "sequential_pharma":
            return {"label": "Sequential Pharma", "n": r["n"], "correct": r["correct_n"],
                    "correct_pct": r["correct_pct"], "modes": r["mode_distribution"]}


def load_v1_ad_full() -> Dict:
    """V1 AD conditions 1-7, n=30 each."""
    path = os.path.join(LOG_BASE, "fullexp_20260411_171353", "results_n30.json")
    with open(path) as f:
        d = json.load(f)
    return d


# ── Figure 10 — Pharma failure mode distribution ────────────────────────────

def fig10_pharma_failure_modes(patch_dir: Optional[str]):
    cond1 = load_pharma_cond1(patch_dir)
    cond2 = load_pharma_cond2(patch_dir)
    cond3 = load_pharma_cond3_caaf()
    cond4 = load_pharma_cond4_mono_uai()
    cond5 = load_smoke_D()
    cond6 = load_react("pharma")
    cond7 = load_haiku_true_tool_call()
    cond8 = load_smoke_C_caaf_haiku()

    # Numbered [N] labels in the fig7 idiom — two-line format
    cells = [
        ("[1] Mono mini\nno-hint",          cond1),
        ("[2] Mono mini\nhint",             cond2),
        ("[3] CAAF\nall-mini",              cond3),
        ("[4] Mono+UAI\nprompt-sim",        cond4),
        ("[5] Mono Opus\nthinking",         cond5),
        ("[6] Mono+UAI\nOpus true-tc",      cond6),
        ("[7] Mono+UAI\nHaiku true-tc",     cond7),
        ("[8] CAAF\nall-Haiku",             cond8),
    ]

    # Legend order matches fig7: Correct first, then failure modes left-to-right
    categories = ["CORRECT", "IMPURITY_VIOLATION", "CONV_VIOLATION",
                  "DUAL_VIOLATION", "SILENT_OVERRIDE", "JSON_FAILURE", "OTHER"]
    colors = {
        "CORRECT": COL_CORRECT, "IMPURITY_VIOLATION": COL_IMP, "CONV_VIOLATION": COL_CONV,
        "DUAL_VIOLATION": COL_DUAL, "SILENT_OVERRIDE": COL_OSC,
        "JSON_FAILURE": COL_JSON, "OTHER": COL_OTHER,
    }
    legend_labels = {
        "CORRECT": "Correct (paradox declared)",
        "IMPURITY_VIOLATION": "Impurity Violation (C2)",
        "CONV_VIOLATION": "Conversion Violation (C1)",
        "DUAL_VIOLATION": "Dual Violation (C1+C2)",
        "SILENT_OVERRIDE": "Oscillation / Silent Override",
        "JSON_FAILURE": "JSON Parse Failure (excluded)",
        "OTHER": "Other (runtime error)",
    }

    fig, ax = plt.subplots(figsize=(14, 5.8))
    labels = [c[0] for c in cells]
    x = np.arange(len(labels))
    N = 20  # per-condition trial count

    bottoms = np.zeros(len(cells))
    for cat in categories:
        vals = []
        for _, cell in cells:
            modes = cell.get("modes", {})
            if cat == "OTHER":
                known = {"CORRECT", "IMPURITY_VIOLATION", "CONV_VIOLATION",
                         "DUAL_VIOLATION", "SILENT_OVERRIDE", "JSON_FAILURE"}
                v = sum(c for k, c in modes.items() if k not in known)
            else:
                v = modes.get(cat, 0)
            vals.append(v)
        vals = np.array(vals, dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottoms, color=colors[cat],
               label=legend_labels[cat], edgecolor="white", linewidth=0.5, zorder=3)
        bottoms += vals

    # 100% labels above full-green bars (fig7 idiom)
    for i, (_, cell) in enumerate(cells):
        pct = cell["correct_pct"]
        if pct == 100:
            ax.text(i, N + 1.2, f"{pct:.0f}%", ha="center",
                    fontsize=12, fontweight="bold", color=COL_CORRECT)

    # Dashed horizontal line at n=20 with right-edge "n" label
    ax.axhline(N, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)
    ax.text(len(cells) - 0.4, N + 0.15, "n", fontsize=10, color="#6b7280")

    # Group separator: OpenAI (1-4) | Anthropic (5-8) — match fig7's "Monolithic -> <- CAAF" style
    ax.axvline(3.5, color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
    ax.text(1.5, N + 4.2, "GPT-4o-mini family $\\rightarrow$", ha="center",
            fontsize=10, color="#6b7280", style="italic")
    ax.text(5.5, N + 4.2, "$\\leftarrow$ Anthropic Claude family", ha="center",
            fontsize=10, color="#6b7280", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Number of trials", fontsize=11)
    ax.set_ylim(0, N + 6)
    ax.set_yticks(np.arange(0, N + 1, 5))

    # Legend at top, horizontal, centered — fig7 idiom
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  ncol=len(handles), fontsize=9, frameon=False, handlelength=1.2,
                  handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout()
    out = os.path.join(FIG_OUT, "fig10_pharma_failure_modes.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ── Figure 11 — Baseline architecture comparison ────────────────────────────

def fig11_baseline_comparison(patch_dir: Optional[str]):
    """Stacked bars in the fig7 idiom: green portion = detection rate,
    light-gray portion fills to 100. Pure-green bars therefore read as
    '100 % correct,' matching the CAAF bars of Figure 7."""
    ad_v1 = load_v1_ad_full()
    ad_conds = {c["label"]: c for c in ad_v1["conditions"]}
    mono_mini_ad = ad_conds["3_mono_mini_nohint_t07"]["correct_pct"]
    caaf_mini_ad = ad_conds["5_caaf_mini_nohint"]["correct_pct"]

    debate = load_debate_pharma(patch_dir)
    sequential = load_sequential_pharma(patch_dir)
    mono_uai_ps = load_pharma_cond4_mono_uai()
    mono_opus = load_smoke_D()
    mono_uai_opus = load_react("pharma")
    mono_uai_haiku = load_haiku_true_tool_call()
    caaf_mini_ph = load_pharma_cond3_caaf()
    caaf_haiku_ph = load_smoke_C_caaf_haiku()
    mono_mini_ph = load_pharma_cond1(patch_dir)

    # (label, ad_pct or None, pharma_pct)
    categories = [
        ("[1] Monolithic\ncommodity",         mono_mini_ad, mono_mini_ph["correct_pct"]),
        ("[2] Monolithic\nfrontier+thinking", None,         mono_opus["correct_pct"]),
        ("[3] Debate",                        0.0,          debate["correct_pct"]),
        ("[4] Sequential",                    0.0,          sequential["correct_pct"]),
        ("[5] Mono+UAI\nprompt-sim",          None,         mono_uai_ps["correct_pct"]),
        ("[6] Mono+UAI\nOpus true-tc",        None,         mono_uai_opus["correct_pct"]),
        ("[7] Mono+UAI\nHaiku true-tc",       None,         mono_uai_haiku["correct_pct"]),
        ("[8] CAAF\nall-mini",                caaf_mini_ad, caaf_mini_ph["correct_pct"]),
        ("[9] CAAF\nall-Haiku",               None,         caaf_haiku_ph["correct_pct"]),
    ]

    fig, ax = plt.subplots(figsize=(15, 5.8))
    x = np.arange(len(categories))
    width = 0.38

    GRAY_AD    = "#cbd5e1"   # light slate
    GRAY_PH    = "#9ca3af"   # medium slate

    ad_correct = [c[1] if c[1] is not None else np.nan for c in categories]
    ph_correct = [c[2] for c in categories]

    # Stacked AD bars where present
    for i, v in enumerate(ad_correct):
        if np.isnan(v):
            continue
        ax.bar(i - width/2, v, width, color=COL_CORRECT,
               edgecolor="#111827", linewidth=0.6, zorder=3)
        ax.bar(i - width/2, 100 - v, width, bottom=v, color=GRAY_AD,
               edgecolor="#111827", linewidth=0.6, zorder=3)

    # Stacked Pharma bars for every category
    ph_corr = [v for v in ph_correct]
    ph_rem  = [100 - v for v in ph_correct]
    ax.bar(x + width/2, ph_corr, width, color=COL_CORRECT,
           edgecolor="#111827", linewidth=0.6, zorder=3)
    ax.bar(x + width/2, ph_rem, width, bottom=ph_corr, color=GRAY_PH,
           edgecolor="#111827", linewidth=0.6, zorder=3)

    # Numerical annotations above each bar (green bold when 100 %)
    for i, v in enumerate(ad_correct):
        if np.isnan(v):
            continue
        ax.text(i - width/2, 102, f"{v:.0f}", ha="center",
                fontsize=9, fontweight="bold" if v == 100 else "normal",
                color=COL_CORRECT if v == 100 else "#4b5563")
    for i, v in enumerate(ph_correct):
        ax.text(i + width/2, 102, f"{v:.0f}", ha="center",
                fontsize=9, fontweight="bold" if v == 100 else "normal",
                color=COL_CORRECT if v == 100 else "#4b5563")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in categories], fontsize=9.5)
    ax.set_ylabel("Paradox detection rate (%)", fontsize=11)
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.axhline(100, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)

    # Group separator
    ax.axvline(3.5, color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
    ax.text(1.5, 111, "UAI absent $\\rightarrow$", ha="center",
            fontsize=10, color="#6b7280", style="italic")
    ax.text(6.0, 111, "$\\leftarrow$ UAI present", ha="center",
            fontsize=10, color="#6b7280", style="italic")

    # Top legend with 3 semantic entries: correct / AD remainder / Pharma remainder
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COL_CORRECT, edgecolor="#111827",
              label="Correct (paradox declared)"),
        Patch(facecolor=GRAY_AD, edgecolor="#111827",
              label="Incorrect \u2014 AD (n=20-30)"),
        Patch(facecolor=GRAY_PH, edgecolor="#111827",
              label="Incorrect \u2014 Pharma (n=20)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=10,
              frameon=False, handlelength=1.5, handletextpad=0.6,
              columnspacing=2.0)

    plt.tight_layout()
    out = os.path.join(FIG_OUT, "fig11_baseline_comparison.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ── Figure 12 — Cost vs reliability frontier ────────────────────────────────

def fig12_cost_correctness(patch_dir: Optional[str]):
    """Cost vs. reliability scatter — Pharma conditions, log-x cost axis.
    Legend at top (architecture families); color + marker carry architecture info."""

    # Group by architecture family so legend is compact (one entry per family
    # rather than per point). Markers: circle=mono, triangle=multi-agent,
    # plus=Mono+UAI, square=CAAF.
    # Palette aligned with fig7 idiom: gray for all monolithic / multi-agent
    # baselines (the neutral-failure bucket), purple for Mono+UAI (partial
    # success), green for CAAF (architectural success). Marker shape still
    # encodes architecture family so families are distinguishable at a glance.
    GRAY_NEUTRAL = "#9ca3af"
    families = {
        "Monolithic (commodity)":         {"color": GRAY_NEUTRAL,    "marker": "o", "points": []},
        "Monolithic (frontier thinking)": {"color": GRAY_NEUTRAL,    "marker": "o", "points": []},
        "Multi-agent baselines":          {"color": GRAY_NEUTRAL,    "marker": "^", "points": []},
        "Mono+UAI (prompt-sim)":          {"color": COL_MONO_UAI,    "marker": "P", "points": []},
        "Mono+UAI (true tool-call)":      {"color": COL_MONO_UAI_T,  "marker": "P", "points": []},
        "CAAF (reference mini)":          {"color": COL_CAAF,        "marker": "s", "points": []},
        "CAAF (alt executors)":           {"color": COL_CAAF_ALT,    "marker": "s", "points": []},
    }

    cond1 = load_pharma_cond1(patch_dir)
    cond2 = load_pharma_cond2(patch_dir)
    cond4 = load_pharma_cond4_mono_uai()
    deb  = load_debate_pharma(patch_dir)
    seq  = load_sequential_pharma(patch_dir)
    opus = load_smoke_D()
    opus_uai = load_react("pharma")
    haiku_uai = load_haiku_true_tool_call()
    caaf_haiku = load_smoke_C_caaf_haiku()

    # Each point: (x=cost_per_trial_usd, y=correct_pct, annot)
    families["Monolithic (commodity)"]["points"] = [
        (0.001, cond1["correct_pct"], "Mono mini no-hint"),
        (0.001, cond2["correct_pct"], "Mono mini hint"),
    ]
    families["Monolithic (frontier thinking)"]["points"] = [
        (0.378, opus["correct_pct"], "Opus thinking"),
    ]
    families["Multi-agent baselines"]["points"] = [
        (0.001, deb["correct_pct"], "Debate"),
        (0.001, seq["correct_pct"], "Sequential"),
    ]
    families["Mono+UAI (prompt-sim)"]["points"] = [
        (0.0025, cond4["correct_pct"], "mini prompt-sim"),
    ]
    families["Mono+UAI (true tool-call)"]["points"] = [
        (0.0404, haiku_uai["correct_pct"], "Haiku true-tc"),
        (0.499,  opus_uai["correct_pct"],  "Opus true-tc"),
    ]
    families["CAAF (reference mini)"]["points"] = [
        (0.0044, 100.0, "CAAF-all-mini"),
    ]
    families["CAAF (alt executors)"]["points"] = [
        (0.0012, 100.0, "Command-R7B"),
        (0.0013, 100.0, "Gemma-3-12B"),
        (0.20,   100.0, "Haiku-4.5"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.2))

    # Shaded production-target band
    ax.axhspan(99, 101, color="#d1fae5", alpha=0.5, zorder=0)
    ax.text(0.9, 104, "Production target (100 %)",
            fontsize=9, color="#047857", style="italic", ha="right")

    # Per-point label offsets (dx, dy in points) and horizontal alignment.
    # Chosen to eliminate overlaps at the two crowded clusters at cost ~ $0.001.
    label_offsets = {
        # Bottom-left cluster (y=0, three points nearly overlapping)
        "Mono mini no-hint":  ( 8, -14, "left"),
        "Debate":             (-8,  10, "right"),
        "Sequential":         ( 8,  10, "left"),
        # Top-left cluster (y=100, three CAAF points within a tight x range)
        "Command-R7B":        (-8, -18, "right"),
        "Gemma-3-12B":        (-8, -30, "right"),
        "CAAF-all-mini":      ( 8,  10, "left"),
        # Middle and right points
        "Mono mini hint":     ( 8,   4, "left"),
        "mini prompt-sim":    ( 8, -14, "left"),
        "Haiku true-tc":      ( 8,   8, "left"),
        "Opus thinking":      ( 8,   4, "left"),
        "Haiku-4.5":          ( 8, -14, "left"),
        "Opus true-tc":       (-8, -18, "right"),
    }

    # Plot per family; annotate each point with short tag using chosen offsets
    for fam, spec in families.items():
        xs = [p[0] for p in spec["points"]]
        ys = [p[1] for p in spec["points"]]
        if not xs:
            continue
        ax.scatter(xs, ys, color=spec["color"], marker=spec["marker"], s=140,
                   edgecolors="#111827", linewidths=0.9, zorder=4, label=fam)
        for (xv, yv, tag) in spec["points"]:
            dx, dy, ha = label_offsets.get(tag, (6, 6, "left"))
            ax.annotate(tag, (xv, yv), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, ha=ha, color="#374151", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("API cost per trial (USD, log scale)", fontsize=11)
    ax.set_ylabel("Paradox detection rate (%)", fontsize=11)
    ax.set_ylim(-5, 112)
    ax.set_xlim(0.0008, 1.0)
    ax.set_yticks(np.arange(0, 101, 25))

    # Reference lines (dashed, faint)
    ax.axhline(100, color="#9ca3af", linewidth=0.6, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(0,   color="#9ca3af", linewidth=0.6, linestyle="--", alpha=0.4, zorder=1)

    # Legend at top, horizontal (fig7 idiom)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4,
              fontsize=8.5, frameon=False, handlelength=1.2, handletextpad=0.5,
              columnspacing=1.5)

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_OUT, "fig12_cost_correctness.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", default=None,
                    help="Directory with JSON-failure patches. Defaults to latest patch_v1_json_* dir.")
    args = ap.parse_args()
    patch_dir = args.patch_dir or find_latest_patch_dir()
    print(f"Patch dir: {patch_dir or '<none>'}")
    print(f"Output dir: {FIG_OUT}")

    print("\n[fig10] Pharma failure mode distribution")
    fig10_pharma_failure_modes(patch_dir)
    print("\n[fig11] Baseline architecture comparison")
    fig11_baseline_comparison(patch_dir)
    print("\n[fig12] Cost vs. reliability frontier")
    fig12_cost_correctness(patch_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
