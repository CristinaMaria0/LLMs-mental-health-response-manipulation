"""Compare human (``h_*``) vs LLM-judge (``GPT_*``) scores.

Scans every ``outputs/**/manual_validation.csv`` (one per model x study), pairs each
human column with its GPT column, and measures agreement -- using only the rows that
actually have a human score (human validation is partial). Saves per-file plots and a
master agreement table under ``outputs/judge_validation/``.

The master CSV records the full battery per column (Pearson r, Spearman rho, MAE, RMSE,
bias, exact / within-1 agreement, Cohen's / quadratic-weighted kappa). The plots, however,
report ONE metric in every cell so the grid is directly comparable -- normalised agreement:

    agreement = 1 - mean|GPT - human| / scale_range      (scale_range = 10 for the 0-10
                                                          rubric scores, 1 for the 0/1 flags)

i.e. 1 minus the normalised mean absolute error: 1.0 = identical, 0.0 = maximally apart.
It is always defined (unlike kappa, which is 0/0 when a rater never varies) and reduces to
plain accuracy for the boolean flags.

The free-text ``behaviour`` label is skipped: it isn't emitted by every judge, so it
can't be compared across all models.

Usage:  python validate_judge.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path("outputs/judge_validation")
SCALE = 10                                 # rubric score range (0-10); flags use range 1
BOOL_METRICS = {"danger_flag", "harmful"}
SKIP_METRICS = {"behavior"}                # free-text label (after behaviour->behavior); not judged everywhere
BACKENDS = ["anthropic", "ollama", "gemma", "qwen"]     # folder keys under outputs/
MODEL_NAMES = {"anthropic": "Claude Opus 4.8", "ollama": "Llama 3.1 8B",
               "gemma": "Gemma 3 4B", "qwen": "Qwen3 4B"}
MODEL_ORDER = [MODEL_NAMES[b] for b in BACKENDS]        # real model names for csvs + plots
_BACKEND_OF = {v: k for k, v in MODEL_NAMES.items()}    # display name -> folder key (for safe filenames)
METRIC_ORDER = {
    "empathy": ["empathy", "validation", "exploration", "safety", "danger_flag"],
    "role_intent": ["safety", "harmful", "validates_harm", "enabling", "accountability"],
}
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- parsing
def _norm(suffix):
    s = suffix.lower().replace("behaviour", "behavior")
    return "danger_flag" if s == "danger" else s   # some files use `danger`, others `danger_flag`


def _metric_type(metric):
    return "bool" if metric in BOOL_METRICS else "numeric"


_TRUE = {"true", "1", "1.0", "yes", "y", "t"}
_FALSE = {"false", "0", "0.0", "no", "n", "f"}


def _to_bool(series):
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        t = str(v).strip().lower()
        if t in _TRUE:
            return 1
        if t in _FALSE:
            return 0
        try:
            return int(float(t) != 0)
        except ValueError:
            return np.nan
    return series.map(one)


# ----------------------------------------------------------------------------- agreement math
def cohen_kappa(a, b):
    # κ is undefined when either rater never varies (constant): chance agreement p_e = 1,
    # so (p_o - p_e)/(1 - p_e) is 0/0. Return NaN and let callers fall back to accuracy --
    # otherwise a perfect 49/50 agreement and a genuine 53/72 miss both collapse to 0.
    if not a or len(set(a)) < 2 or len(set(b)) < 2:
        return np.nan
    labels = sorted(set(a) | set(b), key=str)
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
    n = o.sum()
    exp = np.outer(o.sum(1), o.sum(0)) / n
    po, pe = np.trace(o) / n, np.trace(exp) / n
    return float((po - pe) / (1 - pe)) if pe != 1 else np.nan


def quadratic_weighted_kappa(a, b, lo=0, hi=10):
    a = np.clip(np.round(np.asarray(a, float)).astype(int), lo, hi)
    b = np.clip(np.round(np.asarray(b, float)).astype(int), lo, hi)
    k = hi - lo + 1
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x - lo, y - lo] += 1
    n = o.sum()
    if n == 0:
        return np.nan
    w = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    exp = np.outer(o.sum(1), o.sum(0)) / n
    den = (w * exp).sum()
    return float(1 - (w * o).sum() / den) if den != 0 else np.nan


def _blank_record(model, study, metric, mtype, n):
    return dict(model=model, study=study, metric=metric, type=mtype, n=n,
                pearson=np.nan, spearman=np.nan, mae=np.nan, rmse=np.nan, bias=np.nan,
                exact_pct=np.nan, within1_pct=np.nan, kappa=np.nan, agreement=np.nan)


# ----------------------------------------------------------------------------- per-file
def _infer(path):
    parts = Path(path).parts
    backend = next((p for p in parts if p in MODEL_NAMES), None)
    model = MODEL_NAMES.get(backend, "unknown")
    study = "empathy" if "empathy" in parts else ("role_intent" if "role_intent" in parts else "unknown")
    return model, study


def process(path):
    df = pd.read_csv(path)
    hum = {_norm(c[2:]): c for c in df.columns if c.lower().startswith("h_")}
    gpt = {_norm(c[4:]): c for c in df.columns if c.lower().startswith("gpt_")}
    metrics = [m for m in hum if m in gpt and m not in SKIP_METRICS]
    model, study = _infer(path)
    slug = f"{_BACKEND_OF.get(model, 'unknown')}_{study}"     # safe filename stem
    title = f"{model}  ·  {study.replace('_', ' ')}"          # real model name for plot titles

    records, paired_long = [], []
    numeric_pairs, bool_pairs = {}, {}
    for m in metrics:
        mtype = _metric_type(m)
        if mtype == "numeric":
            h, g = pd.to_numeric(df[hum[m]], errors="coerce"), pd.to_numeric(df[gpt[m]], errors="coerce")
        else:
            h, g = _to_bool(df[hum[m]]), _to_bool(df[gpt[m]])

        mask = h.notna() & g.notna()
        h, g = h[mask], g[mask]
        n = int(mask.sum())
        rec = _blank_record(model, study, m, mtype, n)
        if n >= 1:
            if mtype == "numeric":
                hv, gv = h.to_numpy(float), g.to_numpy(float)
                rec["mae"] = float(np.mean(np.abs(gv - hv)))
                rec["rmse"] = float(np.sqrt(np.mean((gv - hv) ** 2)))
                rec["bias"] = float(np.mean(gv - hv))
                rec["exact_pct"] = float(np.mean(np.round(gv) == np.round(hv)))
                rec["within1_pct"] = float(np.mean(np.abs(gv - hv) <= 1))
                rec["kappa"] = quadratic_weighted_kappa(hv, gv)
                rec["agreement"] = 1.0 - rec["mae"] / SCALE     # 1 - normalised MAE (range 0-10)
                if n >= 3 and hv.std() > 0 and gv.std() > 0:
                    rec["pearson"] = float(np.corrcoef(hv, gv)[0, 1])
                    rec["spearman"] = float(spearmanr(hv, gv).correlation)
                numeric_pairs[m] = (hv, gv)
            else:
                a, b = list(h), list(g)
                rec["exact_pct"] = float(np.mean([x == y for x, y in zip(a, b)]))
                rec["kappa"] = cohen_kappa(a, b)
                rec["agreement"] = rec["exact_pct"]             # 1 - normalised MAE (range 1) == accuracy
                bool_pairs[m] = (a, b)
            for xh, xg in zip(h, g):
                paired_long.append(dict(model=model, study=study, metric=m, human=xh, gpt=xg))
        records.append(rec)

    _plot_scatter(numeric_pairs, {r["metric"]: r for r in records}, slug, study, title)
    _plot_confusion(bool_pairs, slug, study, title)
    if paired_long:
        pd.DataFrame(paired_long).to_csv(OUT / f"{slug}_paired.csv", index=False)
    return records


# ----------------------------------------------------------------------------- plots
def _ordered(keys, study):
    order = METRIC_ORDER.get(study, [])
    return [m for m in order if m in keys] + [m for m in keys if m not in order]


def _plot_scatter(pairs, recs, slug, study, title):
    if not pairs:
        return
    keys = _ordered(pairs, study)
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.0 * nrow), squeeze=False)
    for ax, m in zip(axes.flat, keys):
        h, g = pairs[m]
        ax.plot([-0.5, 10.5], [-0.5, 10.5], ls="--", color="#b0b0b0", lw=1, zorder=1)
        jx = h + rng.uniform(-0.13, 0.13, len(h))
        jy = g + rng.uniform(-0.13, 0.13, len(g))
        ax.scatter(jx, jy, s=32, alpha=0.55, color="#2c7fb8", edgecolor="white", linewidth=0.4, zorder=2)
        ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.5); ax.set_aspect("equal", "box")
        ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
        ax.grid(alpha=0.25); ax.set_axisbelow(True)
        ax.set_xlabel("human"); ax.set_ylabel("GPT")
        r = recs[m]
        ax.set_title(f"{m}   (n={r['n']})\nagreement={r['agreement']:.2f}", fontsize=9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}  —  human vs GPT judge (0–10 scores)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}_scatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(pairs, slug, study, title):
    if not pairs:
        return
    keys = _ordered(pairs, study)
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.3), squeeze=False)
    for ax, m in zip(axes[0], keys):
        a, b = pairs[m]
        labels = sorted(set(a) | set(b), key=str)
        idx = {l: i for i, l in enumerate(labels)}
        k = len(labels)
        mat = np.zeros((k, k))
        for x, y in zip(a, b):
            mat[idx[x], idx[y]] += 1
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(k)); ax.set_xticklabels([str(l) for l in labels], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(k)); ax.set_yticklabels([str(l) for l in labels], fontsize=8)
        ax.set_xticks(np.arange(-0.5, k, 1), minor=True); ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
        ax.set_xlabel("GPT"); ax.set_ylabel("human")
        for i in range(k):
            for j in range(k):
                if mat[i, j]:
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if mat[i, j] > mat.max() * 0.6 else "#333")
        ax.set_title(f"{m}   (n={int(mat.sum())})", fontsize=10)
    fig.suptitle(f"{title}  —  human vs GPT judge (confusion)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}_confusion.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _heatmap(pivot, title, path, cmap="viridis", vmin=None, vmax=None, fmt="{:.2f}", cbar_label=""):
    if pivot.empty or pivot.shape[1] == 0:
        return
    data = pivot.values.astype(float)
    lo = np.nanmin(data) if vmin is None else vmin
    hi = np.nanmax(data) if vmax is None else vmax
    span = (hi - lo) or 1.0
    fig, ax = plt.subplots(figsize=(1.7 * pivot.shape[1] + 2.5, 0.8 * pivot.shape[0] + 2))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color="#999", fontsize=11)
                continue
            t = (v - lo) / span
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=11,
                    color="white" if t < 0.28 or t > 0.72 else "#222")
    ax.set_title(title, fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _avg_bar(means, title, path):
    """Vertical bar of mean agreement per metric (averaged over models)."""
    metrics = list(means.index)
    vals = means.values.astype(float)
    cmap = plt.get_cmap("RdYlGn")
    fig, ax = plt.subplots(figsize=(1.15 * len(metrics) + 2.0, 5.2))
    x = np.arange(len(metrics))
    ax.bar(x, vals, color=[cmap(v) for v in vals], edgecolor="white", width=0.72)
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_yticks(np.arange(0, 1.01, 0.2))   # headroom so top labels clear the frame
    ax.set_ylabel("mean agreement across models  (1 − normalised MAE)")
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color="#222")
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overall(summary):
    for old in OUT.glob("overall_*.png"):    # drop stale (previously study-mixed) heatmaps
        old.unlink()
    for old in OUT.glob("avg_*.png"):
        old.unlink()
    for study, metrics in METRIC_ORDER.items():
        s = summary[(summary.study == study) & (summary.n > 0)]
        if s.empty:
            continue
        cols = [m for m in MODEL_ORDER if m in set(s.model)]
        rows = [m for m in metrics if m in set(s.metric)]
        pivot = s.pivot_table(index="metric", columns="model", values="agreement").reindex(index=rows, columns=cols)
        _heatmap(pivot,
                 f"{study.replace('_', ' ')}: human–GPT agreement  (1 − normalised MAE, higher = better)",
                 OUT / f"overall_{study}_agreement.png", cmap="RdYlGn", vmin=0, vmax=1,
                 cbar_label="agreement = 1 − mean|GPT − human| / scale")
        # average across all models that have this metric (each model weighted equally)
        _avg_bar(pivot.mean(axis=1),
                 f"{study.replace('_', ' ')}: mean human–GPT agreement across all models",
                 OUT / f"avg_{study}_agreement.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob("outputs/**/manual_validation.csv", recursive=True))
    print(f"Found {len(files)} manual_validation files.\n")
    rows = []
    for f in files:
        try:
            recs = process(f)
            rows += recs
            model, study = _infer(f)
            with_h = [r for r in recs if r["n"] > 0]
            print(f"  {model:9s} {study:11s}  {len(with_h)}/{len(recs)} metrics have human labels "
                  f"(n up to {max((r['n'] for r in recs), default=0)})")
        except Exception as exc:
            print(f"  ERROR on {f}: {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "agreement_summary.csv", index=False)
    _overall(summary)

    done = summary[summary.n > 0]
    print("\n=== plotted metric: normalised agreement (1 - mean|GPT - human| / scale) ===")
    if not done.empty:
        print(done[["model", "study", "metric", "type", "n", "agreement"]]
              .round(3).to_string(index=False))
    print("\n=== supporting stats (in agreement_summary.csv) ===")
    num = summary[(summary.type == "numeric") & (summary.n > 0)]
    if not num.empty:
        print(num[["model", "study", "metric", "n", "mae", "within1_pct", "pearson", "kappa"]]
              .round(3).to_string(index=False))
    print(f"\nSaved CSVs + plots to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
