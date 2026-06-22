"""
make_report.py — turn the experiment CSVs into figures + filled report tables
=============================================================================
Reads everything under AutoTS/results/cs/ and writes:

    results/cs/figures/*.png   line plots + heatmaps for the writeup
    results/cs/report_tables.md filled-in versions of the proposal templates

    python -m sparse_localization.make_report

Robust to missing experiments: any CSV that is absent is skipped with a note.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = "./results/cs"
FIG = os.path.join(ROOT, "figures")
K_ORDER = ["2", "3", "5", "10", "20", "all"]
METHOD_ORDER = ["mean", "median", "geometric_median", "dbscan", "nsal",
                "l1_sor", "omp", "cosamp", "sp", "uspa"]
SPARSE = ["l1_sor", "omp", "cosamp", "sp"]


def _exists(*parts):
    p = os.path.join(*parts)
    return p if os.path.exists(p) else None


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Experiment 1 — main table
# ---------------------------------------------------------------------------

def exp1(md):
    path = _exists(ROOT, "exp1", "summary_exp1.csv")
    if not path:
        return
    df = pd.read_csv(path).set_index("method").reindex(METHOD_ORDER + ["kmeans"])
    df = df.dropna(how="all")
    rows = [[m, f"{r.mae:.3f}", f"{r.rmse:.3f}", f"{r.r1m:.2f}", f"{r.r2m:.2f}",
             f"{r.fallback_rate*100:.1f}%"] for m, r in df.iterrows()]
    md.append("## Experiment 1 — Main localization table\n")
    md.append(_md_table(["Method", "MAE ↓", "RMSE ↓", "R@1m ↑", "R@2m ↑", "fallback"], rows))
    md.append("")

    # Bar chart of MAE / RMSE
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df)); w = 0.4
    ax.bar(x - w/2, df["mae"], w, label="MAE")
    ax.bar(x + w/2, df["rmse"], w, label="RMSE")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("meters"); ax.set_title("Experiment 1 — full point sets")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp1_main_table.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2 — controlled sparse k
# ---------------------------------------------------------------------------

def exp2(md):
    path = _exists(ROOT, "exp2", "summary_exp2.csv")
    if not path:
        return
    df = pd.read_csv(path, dtype={"k": str})
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    piv = df.pivot(index="k", columns="method", values="mae_mean").reindex(K_ORDER)

    rows = [[k] + [f"{piv.loc[k, m]:.3f}" for m in methods] for k in K_ORDER]
    md.append("## Experiment 2 — Controlled sparse-k benchmark (MAE, mean over seeds)\n")
    md.append("_k ≤ 5: sparse methods fall back to geometric median (k-guard). "
              "Compare within a row — each k uses a different sign subset._\n")
    md.append(_md_table(["k"] + methods, rows))
    md.append("")

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(K_ORDER))
    for m in methods:
        ax.plot(xs, [piv.loc[k, m] for k in K_ORDER], marker="o", label=m,
                linewidth=2 if m in SPARSE + ["nsal", "uspa"] else 1,
                alpha=1.0 if m in SPARSE + ["nsal", "uspa"] else 0.5)
    ax.set_xticks(xs); ax.set_xticklabels(K_ORDER)
    ax.set_xlabel("observations per sign (k)"); ax.set_ylabel("MAE (m)")
    ax.set_title("Experiment 2 — MAE vs number of observations")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp2_mae_vs_k.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3 — outlier stress
# ---------------------------------------------------------------------------

def exp3(md):
    path = _exists(ROOT, "exp3", "summary_exp3.csv")
    if not path:
        return
    df = pd.read_csv(path)
    headline = 10.0 if 10.0 in df["magnitude"].unique() else df["magnitude"].iloc[0]
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    ratios = sorted(df["ratio"].unique())

    sub = df[df["magnitude"] == headline]
    piv = sub.pivot(index="ratio", columns="method", values="mae_mean").reindex(ratios)
    rows = [[f"{r:.0%}"] + [f"{piv.loc[r, m]:.3f}" for m in methods] for r in ratios]
    md.append(f"## Experiment 3 — Outlier stress test (MAE, magnitude {headline:.0f} m)\n")
    md.append(_md_table(["Outlier ratio"] + methods, rows))
    md.append("")

    # Line: MAE vs ratio
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in methods:
        ax.plot(ratios, [piv.loc[r, m] for r in ratios], marker="o", label=m,
                linewidth=2 if m in SPARSE + ["mean", "nsal"] else 1,
                alpha=1.0 if m in SPARSE + ["mean", "nsal"] else 0.5)
    ax.set_xlabel("outlier ratio"); ax.set_ylabel("MAE (m)")
    ax.set_title(f"Experiment 3 — robustness to gross outliers ({headline:.0f} m)")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_mae_vs_ratio.png"), dpi=150)
    plt.close(fig)

    # Heatmaps of MAE per magnitude
    mags = sorted(df["magnitude"].unique())
    fig, axes = plt.subplots(1, len(mags), figsize=(5 * len(mags), 4.5), squeeze=False)
    for ax, mag in zip(axes[0], mags):
        pv = df[df["magnitude"] == mag].pivot(index="method", columns="ratio",
                                              values="mae_mean").reindex(methods)
        im = ax.imshow(pv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ratios))); ax.set_xticklabels([f"{r:.0%}" for r in ratios])
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=8)
        ax.set_title(f"magnitude {mag:.0f} m"); ax.set_xlabel("outlier ratio")
        for i in range(len(methods)):
            for j in range(len(ratios)):
                ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Experiment 3 — MAE heatmaps (method × outlier ratio)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_heatmaps.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 4 — USPA ablation
# ---------------------------------------------------------------------------

def exp4(md):
    path = _exists(ROOT, "exp4", "summary_exp4.csv")
    if not path:
        return
    df = pd.read_csv(path)
    rows = [[r.cue, f"{r.mae:.3f}", f"{r.rmse:.3f}", f"{r.r1m:.2f}", f"{r.r2m:.2f}"]
            for _, r in df.iterrows()]
    md.append("## Experiment 4 — USPA reliability ablation\n")
    md.append(_md_table(["Reliability cue", "MAE ↓", "RMSE ↓", "R@1m ↑", "R@2m ↑"], rows))
    md.append("")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(df)); w = 0.4
    a1.bar(x - w/2, df["mae"], w, label="MAE")
    a1.bar(x + w/2, df["rmse"], w, label="RMSE")
    a1.set_title("MAE / RMSE"); a1.set_ylabel("meters"); a1.legend()
    a2.bar(x - w/2, df["r1m"], w, label="R@1m")
    a2.bar(x + w/2, df["r2m"], w, label="R@2m")
    a2.set_title("Recall"); a2.set_ylabel("%"); a2.legend()
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(df["cue"], rotation=30, ha="right", fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Experiment 4 — reliability cues")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp4_uspa_ablation.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 5 — synthetic phase transition
# ---------------------------------------------------------------------------

def exp5(md):
    path = _exists(ROOT, "exp5", "summary_exp5.csv")
    if not path:
        return
    df = pd.read_csv(path)
    methods = list(dict.fromkeys(df["method"]))
    ms = sorted(df["m"].unique()); ss = sorted(df["s"].unique())

    md.append("## Experiment 5 — Synthetic phase transition (exact-recovery rate)\n")
    for method in methods:
        pv = df[df["method"] == method].pivot(index="s", columns="m",
                                              values="exact_recovery").reindex(ss)
        rows = [[s] + [f"{pv.loc[s, m]:.2f}" for m in ms] for s in ss]
        md.append(f"**{method}** (rows s, cols m)\n")
        md.append(_md_table(["s \\ m"] + [str(m) for m in ms], rows))
        md.append("")

    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4),
                             squeeze=False)
    for ax, method in zip(axes[0], methods):
        pv = df[df["method"] == method].pivot(index="s", columns="m",
                                              values="exact_recovery").reindex(ss)
        im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms)
        ax.set_yticks(range(len(ss))); ax.set_yticklabels(ss)
        ax.set_xlabel("m (observations)"); ax.set_ylabel("s (outliers)")
        ax.set_title(method)
        for i in range(len(ss)):
            for j in range(len(ms)):
                ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=8)
    fig.suptitle("Experiment 5 — exact-recovery rate (phase transition)")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025)
    fig.savefig(os.path.join(FIG, "exp5_phase_transition.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    md = ["# COMP5340 — Filled Result Tables",
          "",
          "_Auto-generated by `sparse_localization/make_report.py` from "
          "`results/cs/`. Figures in `results/cs/figures/`._",
          ""]
    for fn in (exp1, exp2, exp3, exp4, exp5):
        try:
            fn(md)
        except Exception as e:  # keep going if one experiment is missing/odd
            md.append(f"_({fn.__name__} skipped: {e})_\n")
            print(f"WARN {fn.__name__}: {e}")

    out = os.path.join(ROOT, "report_tables.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {out}")
    print(f"Figures -> {FIG}/")
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
