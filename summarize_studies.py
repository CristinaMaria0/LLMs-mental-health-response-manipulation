"""Cross-model summaries for both studies.

For each study, reads every model's ``outputs/<backend>/<study>/scored_responses.csv``,
computes the mean (+ standard error, n) of all metrics per model x cell, and writes a
tidy table under ``outputs/summary/``:

  * empathy     -> empathy_metrics.csv       (cell = prompt condition)
  * role_intent -> role_intent_metrics.csv   (cell = role x intent)

It also draws a grouped barplot of empathy per model across all prompt conditions
(``outputs/summary/empathy_by_model.png``), read straight from the empathy summary.

Usage:  python summarize_studies.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/summary")
BACKENDS = ["anthropic", "ollama", "gemma", "qwen"]      # folder keys under outputs/
MODEL_NAMES = {"anthropic": "Claude Opus 4.8", "ollama": "Llama 3.1 8B",
               "gemma": "Gemma 3 4B", "qwen": "Qwen3 4B"}
MODEL_ORDER = [MODEL_NAMES[b] for b in BACKENDS]         # real model names, in order, for csvs + plots
MODEL_COLORS = {"Claude Opus 4.8": "#2c7fb8", "Llama 3.1 8B": "#41ab5d",
                "Gemma 3 4B": "#e6a817", "Qwen3 4B": "#d1495b"}

CONDITION_ORDER = ["default", "supportive", "attack_persona", "attack_injection", "attack_detachment"]
ROLE_ORDER = ["victim", "perpetrator"]
INTENT_ORDER = ["help_seeking", "validation", "how_to"]

# every numeric metric each pipeline records (behaviour is free-text categorical -> skipped)
EMPATHY_METRICS = ["empathy", "validation", "exploration", "safety", "danger_flag",
                   "vader", "second_person", "questions", "resp_words"]
ROLE_METRICS = ["safety", "harmful", "validates_harm", "enabling", "accountability"]


def _find_scored(model, study):
    """Locate a model's scored_responses.csv (some backends nest it under a judge folder)."""
    direct = Path(f"outputs/{model}/{study}/scored_responses.csv")
    if direct.exists():
        return direct
    hits = sorted(glob.glob(f"outputs/{model}/{study}/**/scored_responses.csv", recursive=True))
    gpt = [h for h in hits if "gpt" in h.lower()]        # prefer the shared GPT judge when several exist
    return Path(gpt[0]) if gpt else (Path(hits[0]) if hits else None)


def load_scored(study, metrics, group_cols):
    """Concatenate every model's per-response scores (model + group cols + numeric metrics)."""
    frames = []
    for backend in BACKENDS:
        path = _find_scored(backend, study)
        name = MODEL_NAMES[backend]
        if path is None:
            print(f"  (skip {name}: no scored_responses.csv)")
            continue
        df = pd.read_csv(path)
        keep = group_cols + [m for m in metrics if m in df.columns]
        sub = df[keep].copy()
        for c in metrics:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub["model"] = name
        frames.append(sub)
        print(f"  {name:16s}  n={len(df):4d}  <- {path}")
    return pd.concat(frames, ignore_index=True)


def summarize(df, group_cols, metrics, cat_orders):
    """Tidy table: one row per (model, *group_cols, metric) with mean, sem, n."""
    metrics = [m for m in metrics if m in df.columns]
    long = df.melt(id_vars=["model"] + group_cols, value_vars=metrics,
                   var_name="metric", value_name="value").dropna(subset=["value"])
    summ = (long.groupby(["model"] + group_cols + ["metric"])["value"]
                .agg(mean="mean", sem="sem", n="count").reset_index())
    for col, order in {"model": MODEL_ORDER, "metric": metrics, **cat_orders}.items():
        summ[col] = pd.Categorical(summ[col], order)
    return summ.sort_values(["metric", "model"] + group_cols).reset_index(drop=True)


def _grouped_bar(mean, sem, cats, labels, title, ylabel, xlabel, path, ymax=10):
    """Grouped barplot: one bar group per category (columns of mean/sem), one bar per model."""
    models = [m for m in MODEL_ORDER if m in mean.index]
    x = np.arange(len(cats))
    w = 0.8 / max(len(models), 1)
    floor = ymax * 0.012                                 # floor so a ~0 mean still shows a sliver
    fig, ax = plt.subplots(figsize=(1.55 * len(cats) + 2.5, 5.4))
    for k, m in enumerate(models):
        off = (k - (len(models) - 1) / 2) * w
        vals = [mean.loc[m, c] if c in mean.columns else np.nan for c in cats]
        vals = [floor if (not np.isnan(v) and v < floor) else v for v in vals]
        errs = [sem.loc[m, c] if c in sem.columns else np.nan for c in cats]
        ax.bar(x + off, vals, w, yerr=errs, capsize=3, label=m,
               color=MODEL_COLORS.get(m, "#888"), edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=1, ecolor="#555"))

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, ymax); ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    ax.legend(title="model", frameon=False, ncol=len(models), loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_empathy(summ, path):
    """Grouped barplot of mean empathy per model across all prompt conditions."""
    sub = summ[summ.metric == "empathy"]
    mean = sub.pivot(index="model", columns="condition", values="mean")
    sem = sub.pivot(index="model", columns="condition", values="sem")
    cats = [c for c in CONDITION_ORDER if c in mean.columns and mean[c].notna().any()]
    labels = [c.replace("attack_", "") for c in cats]
    _grouped_bar(mean, sem, cats, labels, "Empathy by model across all prompts",
                 "mean empathy  (0-10, GPT judge)", "prompt condition", path)


def plot_role_intent_safety(summ, path):
    """Grouped barplot of mean safety per model across the 6 role x intent cells."""
    sub = summ[summ.metric == "safety"].copy()
    sub["cell"] = sub["role"].astype(str) + " / " + sub["intent"].astype(str)
    mean = sub.pivot(index="model", columns="cell", values="mean")
    sem = sub.pivot(index="model", columns="cell", values="sem")
    cats = [f"{r} / {i}" for r in ROLE_ORDER for i in INTENT_ORDER if f"{r} / {i}" in mean.columns]
    labels = [c.replace(" / ", "\n") for c in cats]
    _grouped_bar(mean, sem, cats, labels, "Role x intent safety by model",
                 "mean safety  (0-10, GPT judge)", "role / intent", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Empathy study:")
    emp = load_scored("empathy", EMPATHY_METRICS, ["condition"])
    emp_summ = summarize(emp, ["condition"], EMPATHY_METRICS, {"condition": CONDITION_ORDER})
    (OUT / "empathy_by_model.csv").unlink(missing_ok=True)   # replaced by the all-metrics table
    emp_summ.to_csv(OUT / "empathy_metrics.csv", index=False)
    plot_empathy(emp_summ, OUT / "empathy_by_model.png")

    print("\nRole-intent study:")
    ri = load_scored("role_intent", ROLE_METRICS, ["role", "intent"])
    ri_summ = summarize(ri, ["role", "intent"], ROLE_METRICS, {"role": ROLE_ORDER, "intent": INTENT_ORDER})
    ri_summ.to_csv(OUT / "role_intent_metrics.csv", index=False)
    plot_role_intent_safety(ri_summ, OUT / "role_intent_safety_by_model.png")

    print("\nMean empathy per model (over all prompts):")
    print(emp.groupby("model")["empathy"].mean().reindex(MODEL_ORDER).round(2).to_string())
    print("\nMean P(harmful) per model (over all cells):")
    print(ri.groupby("model")["harmful"].mean().reindex(MODEL_ORDER).round(3).to_string())
    print(f"\nSaved: {(OUT / 'empathy_metrics.csv').resolve()}")
    print(f"Saved: {(OUT / 'role_intent_metrics.csv').resolve()}")
    print(f"Saved: {(OUT / 'empathy_by_model.png').resolve()}")
    print(f"Saved: {(OUT / 'role_intent_safety_by_model.png').resolve()}")


if __name__ == "__main__":
    main()
