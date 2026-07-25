# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(".")
OUT_DIR = Path(".")   # keep outputs in current directory

# CSVs expected. Missing files are ignored.
EXPECTED_FILES = {
    "framed": "framed.csv",
    "domain": "domain.csv",
    "efficiency": "efficiency.csv",
    "hdbscan": "hdbscan.csv",
    "kmediods": "kmediods.csv",
    "topsis": "topsis.csv",
    "weight": "weight.csv",
    "con50": "con50.csv",
    "con75": "con75.csv",
    "con90": "con90.csv",
}

# SOIs used as votes for recomputing consensus.
# Do not include framed or already-combined conXX files to avoid circularity.
BASE_SOIS_FOR_CONSENSUS = [
    "domain",
    "efficiency",
    "hdbscan",
    "kmediods",
    "topsis",
    "weight",
]

# All subsets to compare, if present.
COMPARISON_ORDER = [
    "framed",
    "domain",
    "efficiency",
    "hdbscan",
    "kmediods",
    "topsis",
    "weight",
    "con50",
    "con75",
    "con90",
]

# Objective / indicator columns for descriptive profiling.
PROFILE_COLS_CANDIDATES = [
    "satisfaction",
    "effort",
    "time",
    "productivity",
    "response",
    "opportunity",
    "scope",
    "squandering",
    "stcov_cv1",
    "stcov_cv2",
    "stcov_cv3",
    "stcov_cv4",
]

# Consensus thresholds to recompute from base SOIs.
CONSENSUS_THRESHOLDS = [0.50, 0.75, 0.90]


# ============================================================
# HELPERS
# ============================================================

def read_sois():
    sois = {}
    for name, filename in EXPECTED_FILES.items():
        path = DATA_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            if "id" not in df.columns:
                print(f"[WARN] {filename} skipped: no 'id' column.")
                continue
            df["id"] = df["id"].astype(int)
            sois[name] = df
            print(f"[OK] loaded {name}: {df.shape[0]} rows")
        else:
            print(f"[MISS] {filename}")
    return sois


def get_req_cols(df):
    return [c for c in df.columns if re.match(r"^req_\d+$", c)]


def safe_id_set(df):
    return set(df["id"].dropna().astype(int).tolist())


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b) if len(a | b) else np.nan


def overlap_coeff(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return np.nan
    return len(a & b) / min(len(a), len(b))


def containment(a, b):
    """Proportion of A contained in B."""
    a, b = set(a), set(b)
    if not a:
        return np.nan
    return len(a & b) / len(a)


def requirement_metrics(df):
    req_cols = get_req_cols(df)
    n = len(df)

    if n == 0 or len(req_cols) == 0:
        return {
            "req_density": np.nan,
            "active_req_count": 0,
            "core_req_count": 0,
            "core_req_ratio": np.nan,
            "variable_req_count": 0,
            "req_variability": np.nan,
        }

    X = df[req_cols].astype(float)

    freq = X.mean(axis=0)
    active = freq > 0
    core = freq == 1
    variable = (freq > 0) & (freq < 1)

    active_count = int(active.sum())
    core_count = int(core.sum())

    # Bernoulli variability averaged over all requirements.
    # Maximum for a requirement is 0.25 at p=0.5.
    variability = (freq * (1 - freq)).mean()

    return {
        "req_density": float(X.values.mean()),
        "active_req_count": active_count,
        "core_req_count": core_count,
        "core_req_ratio": float(core_count / active_count) if active_count > 0 else np.nan,
        "variable_req_count": int(variable.sum()),
        "req_variability": float(variability),
    }


def objective_profile_metrics(df, columns):
    out = {}
    for c in columns:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out[f"{c}_mean"] = float(df[c].mean())
            out[f"{c}_std"] = float(df[c].std(ddof=0))
            out[f"{c}_min"] = float(df[c].min())
            out[f"{c}_max"] = float(df[c].max())
            out[f"{c}_spread"] = float(df[c].max() - df[c].min())
    return out


def compute_summary(sois):
    framed_size = len(sois["framed"]) if "framed" in sois else None

    # Count in how many compared SOIs each id appears, for unique contribution later.
    id_to_sois = {}
    for name, df in sois.items():
        for sid in safe_id_set(df):
            id_to_sois.setdefault(sid, []).append(name)

    rows = []
    for name in COMPARISON_ORDER:
        if name not in sois:
            continue

        df = sois[name]
        ids = safe_id_set(df)
        row = {
            "soi": name,
            "size": len(ids),
        }

        if framed_size:
            row["selectivity_vs_framed"] = len(ids) / framed_size
        else:
            row["selectivity_vs_framed"] = np.nan

        unique_ids = [
            sid for sid in ids
            if len([x for x in id_to_sois.get(sid, []) if x != "framed"]) == 1
        ]
        row["unique_solution_count_excluding_framed"] = len(unique_ids)

        row.update(requirement_metrics(df))
        row.update(objective_profile_metrics(df, PROFILE_COLS_CANDIDATES))

        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


def compute_pairwise(sois, metric="jaccard"):
    names = [n for n in COMPARISON_ORDER if n in sois]
    sets = {n: safe_id_set(sois[n]) for n in names}

    M = pd.DataFrame(index=names, columns=names, dtype=float)

    for a in names:
        for b in names:
            if metric == "jaccard":
                M.loc[a, b] = jaccard(sets[a], sets[b])
            elif metric == "overlap":
                M.loc[a, b] = overlap_coeff(sets[a], sets[b])
            elif metric == "containment":
                M.loc[a, b] = containment(sets[a], sets[b])
            else:
                raise ValueError(metric)

    return M


def compute_solution_consensus(sois, base_names):
    present_base = [n for n in base_names if n in sois]
    if not present_base:
        raise ValueError("No base SOIs available for consensus.")

    all_ids = sorted(set().union(*[safe_id_set(sois[n]) for n in present_base]))

    rows = []
    for sid in all_ids:
        support_names = [n for n in present_base if sid in safe_id_set(sois[n])]
        C = len(support_names) / len(present_base)
        rows.append({
            "id": sid,
            "consensus_score": C,
            "support_count": len(support_names),
            "support_names": ";".join(support_names),
        })

    return pd.DataFrame(rows).sort_values(
        ["consensus_score", "support_count", "id"],
        ascending=[False, False, True]
    )


def compute_requirement_frequency_by_soi(sois):
    rows = []
    for name in COMPARISON_ORDER:
        if name not in sois:
            continue
        df = sois[name]
        req_cols = get_req_cols(df)
        if not req_cols:
            continue
        freq = df[req_cols].mean(axis=0)
        for req, val in freq.items():
            rows.append({"soi": name, "requirement": req, "frequency": float(val)})
    return pd.DataFrame(rows)


def save_heatmap(matrix, filename, title, vmin=0, vmax=1):
    if matrix.empty:
        return

    fig_w = max(7, 0.55 * len(matrix.columns))
    fig_h = max(5, 0.45 * len(matrix.index))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values.astype(float), vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(summary):
    if summary.empty:
        return

    plot_df = summary.copy()
    plot_df = plot_df.set_index("soi")

    cols = ["size", "selectivity_vs_framed", "core_req_ratio", "req_density", "req_variability"]
    cols = [c for c in cols if c in plot_df.columns]

    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 2.4 * len(cols)), sharex=True)

    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.bar(plot_df.index, plot_df[c])
        ax.set_ylabel(c)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SOI descriptive characterization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_soi_summary_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_requirement_frequency_heatmap(req_freq):
    if req_freq.empty:
        return

    mat = req_freq.pivot(index="soi", columns="requirement", values="frequency")
    # Natural sort req_1, req_2...
    req_cols = sorted(mat.columns, key=lambda x: int(x.split("_")[1]))
    mat = mat[req_cols]

    fig_w = max(12, 0.28 * len(req_cols))
    fig_h = max(4, 0.45 * len(mat.index))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(req_cols)))
    ax.set_xticklabels(req_cols, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title("Requirement inclusion frequency by SOI", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_requirement_frequency_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_solution_consensus_membership(sois, consensus_df, base_names, max_solutions=60):
    present_base = [n for n in base_names if n in sois]
    if not present_base or consensus_df.empty:
        return

    # Keep solutions with some support, sorted by consensus.
    top_ids = consensus_df.head(max_solutions)["id"].tolist()

    data = []
    for sid in top_ids:
        row = []
        for n in present_base:
            row.append(1 if sid in safe_id_set(sois[n]) else 0)
        data.append(row)

    mat = pd.DataFrame(data, index=[str(x) for x in top_ids], columns=present_base)

    fig_w = max(7, 0.6 * len(present_base))
    fig_h = max(6, 0.18 * len(top_ids))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_ylabel("Solution ID")
    ax.set_title("Solution membership across base SOIs", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_solution_consensus_membership.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

sois = read_sois()

if not sois:
    raise SystemExit("No SOI CSV files found.")

# Summary table
summary = compute_summary(sois)
summary.to_csv(OUT_DIR / "out_soi_summary.csv", index=False)

# Smaller LaTeX table for the paper
latex_cols = [
    "soi", "size", "selectivity_vs_framed",
    "active_req_count", "core_req_count", "core_req_ratio",
    "req_density", "req_variability",
]
latex_cols = [c for c in latex_cols if c in summary.columns]
summary[latex_cols].to_latex(
    OUT_DIR / "out_soi_metrics_latex.tex",
    index=False,
    float_format="%.3f"
)

# Pairwise similarities
J = compute_pairwise(sois, "jaccard")
O = compute_pairwise(sois, "overlap")
D = compute_pairwise(sois, "containment")

J.to_csv(OUT_DIR / "out_pairwise_jaccard.csv")
O.to_csv(OUT_DIR / "out_pairwise_overlap.csv")
D.to_csv(OUT_DIR / "out_pairwise_containment.csv")

save_heatmap(J, "fig_pairwise_jaccard.png", "Pairwise Jaccard similarity")
save_heatmap(O, "fig_pairwise_overlap.png", "Pairwise overlap coefficient")

# Consensus by solution from base lenses only
present_base = [n for n in BASE_SOIS_FOR_CONSENSUS if n in sois]

if present_base:
    consensus_df = compute_solution_consensus(sois, BASE_SOIS_FOR_CONSENSUS)

    # Add thresholds membership
    for tau in CONSENSUS_THRESHOLDS:
        consensus_df[f"in_consensus_{tau:.2f}"] = consensus_df["consensus_score"] >= tau

    consensus_df.to_csv(OUT_DIR / "out_consensus_by_solution.csv", index=False)

    # Consensus sizes table
    consensus_rows = []
    for tau in CONSENSUS_THRESHOLDS:
        n_tau = int((consensus_df["consensus_score"] >= tau).sum())
        consensus_rows.append({
            "tau": tau,
            "consensus_size": n_tau,
            "interpretation": (
                "broad consensus pool" if tau < 0.75 else "compact consensus core"
            )
        })
    consensus_sizes = pd.DataFrame(consensus_rows)
    consensus_sizes.to_csv(OUT_DIR / "out_consensus_sizes.csv", index=False)
    consensus_sizes.to_latex(
        OUT_DIR / "out_consensus_sizes_latex.tex",
        index=False,
        float_format="%.2f"
    )

    plot_solution_consensus_membership(sois, consensus_df, BASE_SOIS_FOR_CONSENSUS)

else:
    print("[WARN] No base SOIs found for consensus calculation.")

# Requirement frequencies
req_freq = compute_requirement_frequency_by_soi(sois)
req_freq.to_csv(OUT_DIR / "out_requirement_frequency_by_soi.csv", index=False)

# Plots
plot_summary_bars(summary)
plot_requirement_frequency_heatmap(req_freq)

print("\nGenerated outputs:")
for f in [
    "out_soi_summary.csv",
    "out_soi_metrics_latex.tex",
    "out_pairwise_jaccard.csv",
    "out_pairwise_overlap.csv",
    "out_pairwise_containment.csv",
    "out_consensus_by_solution.csv",
    "out_consensus_sizes.csv",
    "out_consensus_sizes_latex.tex",
    "out_requirement_frequency_by_soi.csv",
    "fig_pairwise_jaccard.png",
    "fig_pairwise_overlap.png",
    "fig_soi_summary_bars.png",
    "fig_requirement_frequency_heatmap.png",
    "fig_solution_consensus_membership.png",
]:
    if (OUT_DIR / f).exists():
        print(" -", f)