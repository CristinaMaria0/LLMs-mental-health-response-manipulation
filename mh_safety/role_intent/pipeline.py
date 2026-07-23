"""End-to-end role x intent safety pipeline (generate -> judge -> analyze ->
plot -> save). Each step is a function so notebooks stay thin."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

from ..config import model_display_name
from ..llm import LLMClient, judge_client
from ..stats import risk_ratio, chi2_p
from ..visual import annotated_heatmap
from .scenarios import build_prompt_df, ROLES, INTENTS
from . import prompts as P
from . import judge as J


def generate_responses(cfg, prompts_df, client):
    responses = []
    for r in tqdm(prompts_df.itertuples(), total=len(prompts_df), desc="generating"):
        text = client.generate(P.NEUTRAL_SYSTEM, r.prompt)
        responses.append(text)
    responses_df = prompts_df.copy()
    responses_df["response"] = responses
    return responses_df


def judge_responses(cfg, responses_df, judge=None):
    judge = judge or judge_client(cfg.judge_llm)          # shared judge (default: OpenAI GPT-5)
    recs, n_fail = [], 0
    for r in tqdm(responses_df.itertuples(), total=len(responses_df), desc="judging"):
        try:
            recs.append(judge.judge_json(J.JUDGE_SYSTEM, J.judge_prompt(r.role, r.intent, r.prompt, r.response), J.JUDGE_SCHEMA))
        except Exception:  # judge returned unparseable JSON -> skip this one response
            recs.append({})
            n_fail += 1
    if n_fail:
        print(f"[judge] {n_fail}/{len(responses_df)} responses could not be judged (invalid JSON); dropped.")
    scored = pd.concat([responses_df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    scored = scored.dropna(subset=["harmful"]).reset_index(drop=True)
    scored["harmful"] = scored["harmful"].astype(int)
    return scored


def _logit_table(df):
    """Adjusted odds ratios for harmful ~ role * intent. Returns a DataFrame or a
    note string when the data is (quasi-)perfectly separated."""
    try:
        dd = df.copy()
        dd["role"] = pd.Categorical(dd["role"], categories=ROLES)
        dd["intent"] = pd.Categorical(dd["intent"], categories=INTENTS)
        res = smf.logit("harmful ~ C(role) * C(intent)", data=dd).fit(disp=0, maxiter=200)
        if np.abs(res.params).max() > 15 or bool((res.pvalues > 0.99).all()):
            raise RuntimeError("quasi-perfect separation")
        ci = np.exp(res.conf_int())
        return pd.DataFrame({"odds_ratio": np.exp(res.params).round(3),
                             "ci_low": ci[0].round(3), "ci_high": ci[1].round(3),
                             "p_value": res.pvalues.round(4)})
    except Exception as e:
        return (f"(finite odds ratios unavailable - {e}); a cell is 0% or 100% harmful "
                "(perfect separation, common with a well-aligned model). Rely on the "
                "risk-ratio + chi-square results above.")


def analyze(cfg, scored_df):
    df = scored_df
    harm_pivot = df.pivot_table(index="role", columns="intent", values="harmful",
                                aggfunc="mean").reindex(index=ROLES, columns=INTENTS)
    safe_pivot = df.pivot_table(index="role", columns="intent", values="safety",
                                aggfunc="mean").reindex(index=ROLES, columns=INTENTS)

    def rate(mask):
        s = df.loc[mask, "harmful"]
        return float(s.mean()), int(s.sum()), int(len(s))

    pv, cv, nv = rate(df.role == "victim")
    pp, cp_, npp = rate(df.role == "perpetrator")
    rr_role = risk_ratio(cp_, npp, cv, nv)
    p_role = chi2_p(pd.crosstab(df.role, df.harmful))

    by_intent = {it: rate(df.intent == it) for it in INTENTS}
    ch, nh = by_intent["help_seeking"][1], by_intent["help_seeking"][2]
    rr_val = risk_ratio(by_intent["validation"][1], by_intent["validation"][2], ch, nh)
    rr_how = risk_ratio(by_intent["how_to"][1], by_intent["how_to"][2], ch, nh)
    p_int = chi2_p(pd.crosstab(df.intent, df.harmful))

    pe, ce, ne = rate((df.role == "perpetrator") & (df.intent == "how_to"))
    pb, cb, nb = rate((df.role == "victim") & (df.intent == "help_seeking"))
    rr_x = risk_ratio(ce, ne, cb, nb)

    headline = {
        "n_prompts": int(len(df)),
        "P_harm_by_role": {"victim": pv, "perpetrator": pp},
        "P_harm_by_intent": {it: by_intent[it][0] for it in INTENTS},
        "P_harm_2x3": {f"{r}|{i}": float(harm_pivot.loc[r, i]) for r in ROLES for i in INTENTS},
        "risk_ratio_perp_vs_victim": rr_role,
        "risk_ratio_validation_vs_help": rr_val,
        "risk_ratio_howto_vs_help": rr_how,
    }
    return {"harm_pivot": harm_pivot, "safe_pivot": safe_pivot, "by_intent": by_intent,
            "role": (pv, pp, rr_role, p_role), "intent": (rr_val, rr_how, p_int),
            "interaction": (pb, pe, rr_x), "logit": _logit_table(df), "headline": headline}


def print_report(cfg, scored_df, A):
    print("P(harmful) by role x intent:\n", A["harm_pivot"].round(3), "\n")
    print("mean safety by role x intent:\n", A["safe_pivot"].round(2), "\n")
    pv, pp, rr_role, p_role = A["role"]
    print(f"Q1 ROLE: P(harm|victim)={pv:.3f}  P(harm|perpetrator)={pp:.3f}  "
          f"RR(perp/victim)={rr_role:.2f}  chi2 p={p_role:.3g}")
    rr_val, rr_how, p_int = A["intent"]
    for it in INTENTS:
        print(f"   P(harm|{it:12s}) = {A['by_intent'][it][0]:.3f}")
    print(f"Q2 INTENT: RR(validation/help)={rr_val:.2f}  RR(how_to/help)={rr_how:.2f}  chi2 p={p_int:.3g}")
    pb, pe, rr_x = A["interaction"]
    print(f"Interaction: victim+help={pb:.3f} vs perpetrator+how_to={pe:.3f}  (RR={rr_x:.2f})")
    print("\nLogistic regression (harmful ~ role * intent):")
    print(A["logit"] if isinstance(A["logit"], str) else A["logit"].to_string())


def make_plots(cfg, scored_df, A, show=False):
    df = scored_df
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    BEHAVIORS = ["empathetic_support", "safe_redirect", "challenge_repair",
                 "sycophantic_validation", "enabling_howto", "dismissive", "other"]
    BCOLORS = {"empathetic_support": "#2ecc71", "safe_redirect": "#27ae60", "challenge_repair": "#16a085",
               "sycophantic_validation": "#e67e22", "enabling_howto": "#c0392b",
               "dismissive": "#7f3f98", "other": "#95a5a6"}

    paths, figs = [], []
    mname = model_display_name(cfg.llm)                       # e.g. "Claude Opus 4.8"
    fslug = re.sub(r'[^\w.\- ]+', '_', mname).strip()         # safe filename fragment

    def _finish(fig, stem):
        fig.tight_layout()
        p = out / f"{fslug} - {stem}.png"
        fig.savefig(p, dpi=120)
        paths.append(p); figs.append(fig)

    # 1. P(harmful) heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    annotated_heatmap(ax, A["harm_pivot"], f"{mname}: P(harmful) by role x intent", fig, vmax=1.0, fmt="{:.2f}")
    _finish(fig, "role_intent_harm_heatmap")

    # 2. mean safety heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    annotated_heatmap(ax, A["safe_pivot"], f"{mname}: mean safety by role x intent", fig, vmax=10.0, fmt="{:.1f}")
    _finish(fig, "role_intent_safety_heatmap")

    # 3. harm probability by intent, split by role
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(INTENTS)); w = 0.38
    for k, role in enumerate(ROLES):
        ax.bar(x + (k - 0.5) * w, [A["harm_pivot"].loc[role, it] for it in INTENTS], w, label=role)
    ax.set_xticks(x); ax.set_xticklabels(INTENTS, rotation=10)
    ax.set_ylim(0, 1); ax.set_ylabel("P(harmful)")
    ax.set_title(f"{mname}: harm probability by intent, split by role"); ax.legend()
    _finish(fig, "role_intent_harm_by_intent")

    # 4. behaviour mix per cell
    fig, ax = plt.subplots(figsize=(8, 5))
    cells = [f"{r} / {i}" for r in ROLES for i in INTENTS]
    comp = (df.assign(behavior=pd.Categorical(df.behavior, categories=BEHAVIORS))
              .pivot_table(index="cell", columns="behavior", values="uid", aggfunc="count", observed=False)
              .reindex(cells).fillna(0))
    comp = comp.div(comp.sum(axis=1), axis=0)
    bottom = np.zeros(len(cells))
    for b in BEHAVIORS:
        if b in comp.columns:
            ax.bar(range(len(cells)), comp[b].values, bottom=bottom, color=BCOLORS[b], label=b)
            bottom += comp[b].values
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells, rotation=25, ha="right")
    ax.set_ylim(0, 1); ax.set_title(f"{mname}: what the model does")
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    _finish(fig, "role_intent_behavior_mix")

    if show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)
    return paths


def save_results(cfg, scored_df, A):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out / "scored_responses.csv", index=False)
    A["harm_pivot"].to_csv(out / "harm_rate_2x3.csv")
    A["safe_pivot"].to_csv(out / "safety_2x3.csv")
    if not isinstance(A["logit"], str):
        A["logit"].to_csv(out / "logistic_odds_ratios.csv")
    (out / "headline.json").write_text(json.dumps(A["headline"], indent=2), encoding="utf-8")
    return out


def run(cfg, client=None, show=False):
    """Full pipeline. Returns a dict with prompts, scored frame and analysis."""
    client = client or LLMClient(cfg.llm)
    prompts_df = build_prompt_df(cfg)
    responses = generate_responses(cfg, prompts_df, client)
    scored = judge_responses(cfg, responses)
    A = analyze(cfg, scored)
    make_plots(cfg, scored, A, show=show)
    save_results(cfg, scored, A)
    return {"prompts": prompts_df, "scored": scored, **A}
