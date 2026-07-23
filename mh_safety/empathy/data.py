"""Load, scrub, filter, risk-stratify and sample the Reddit Mental Health posts."""
from pathlib import Path

import pandas as pd

from ..text import scrub

try:
    from ftfy import fix_text as _fix_text  # repairs mojibake from mixed-encoding files
except ImportError:  # fallback: fix the most common UTF-8-read-as-cp1252 punctuation
    _MOJIBAKE = {"â€™": "’", "â€˜": "‘", "â€œ": "“", "â€\x9d": "”",
                 "â€”": "—", "â€“": "–", "â€¦": "…", "Â": ""}

    def _fix_text(text):
        text = str(text)
        for bad, good in _MOJIBAKE.items():
            text = text.replace(bad, good)
        return text

USECOLS = ["subreddit", "author", "date", "post", "n_words", "sent_compound",
           "suicidality_total", "isolation_total", "substance_use_total"]
DELETED = {"[deleted]", "[removed]", "", "nan", "none"}
EN_DATASET = "data/EN_dataset.csv"  # curated set (same schema as data/raw); used by load_sample


def _find_file(sub, timeframe, data_dir):
    p = Path(data_dir) / f"{sub}_{timeframe}_features_tfidf_256.csv"
    if p.exists():
        return p
    cands = sorted(Path(data_dir).glob(f"{sub}_*_features_tfidf_256.csv"))
    return cands[0] if cands else None


def load_raw(cfg):
    frames = []
    for sub in cfg.subreddits:
        fp = _find_file(sub, cfg.timeframe, cfg.data_dir)
        if fp is None:
            continue
        df = pd.read_csv(fp, usecols=lambda c: c in USECOLS)
        df["source_file"] = fp.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No subreddit CSVs found under {cfg.data_dir} for {cfg.subreddits}")
    return pd.concat(frames, ignore_index=True)


def filter_posts(cfg, raw_df):
    d = raw_df.dropna(subset=["post"])
    d = d[~d["post"].astype(str).str.strip().str.lower().isin(DELETED)].copy()
    d["post_clean"] = d["post"].map(scrub)
    d["wc"] = d["post_clean"].str.split().map(len)
    d = d[(d["wc"] >= cfg.min_words) & (d["wc"] <= cfg.max_words)]
    return d.drop_duplicates(subset=["post_clean"]).reset_index(drop=True)


def risk_tier_row(row):
    sub = str(row.get("subreddit", "")).lower()
    suic = row.get("suicidality_total", 0) or 0
    comp = row.get("sent_compound", 0.0)
    if sub == "suicidewatch" or suic >= 1:
        return "high"
    if comp <= -0.6:
        return "elevated"
    return "moderate"


def stratified_sample(cfg, filtered_df):
    df = filtered_df.copy()
    df["risk_tier"] = df.apply(risk_tier_row, axis=1)
    tiers = ["high", "elevated", "moderate"]
    per = max(1, cfg.n_posts // len(tiers))
    parts = [df[df.risk_tier == t].sample(min(per, int((df.risk_tier == t).sum())), random_state=cfg.seed)
             for t in tiers]
    s = pd.concat(parts)
    if len(s) < cfg.n_posts:
        extra = df[~df["post_clean"].isin(s["post_clean"])]
        s = pd.concat([s, extra.sample(min(cfg.n_posts - len(s), len(extra)), random_state=cfg.seed)])
    s = s.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    s["post_id"] = ["p%03d" % i for i in range(len(s))]
    return s[["post_id", "subreddit", "risk_tier", "wc", "suicidality_total", "sent_compound", "post_clean"]]


def load_sample(cfg):
    """Return *all* posts from ``EN_DATASET`` (data/EN_dataset.csv) in the sampling
    format the pipeline expects. No random selection -- every post is returned, so
    ``cfg.n_posts`` is not used. The CSV shares the Reddit Mental Health schema, so
    the same columns / risk tiers apply. PII is still scrubbed and empty/deleted
    posts are dropped (they cannot be generated on)."""
    try:
        df = pd.read_csv(EN_DATASET, usecols=lambda c: c in USECOLS)
    except UnicodeDecodeError:  # this export is Windows-1252, not UTF-8
        df = pd.read_csv(EN_DATASET, usecols=lambda c: c in USECOLS, encoding="cp1252")
    df = df.dropna(subset=["post"])
    df = df[~df["post"].astype(str).str.strip().str.lower().isin(DELETED)].copy()
    df["post_clean"] = df["post"].map(lambda t: scrub(_fix_text(t)))
    df["wc"] = df["post_clean"].str.split().map(len)
    df = df[df["wc"] > 0].reset_index(drop=True)
    df["risk_tier"] = df.apply(risk_tier_row, axis=1)
    for col in ("suicidality_total", "sent_compound"):
        if col not in df.columns:
            df[col] = 0
    df["post_id"] = ["p%03d" % i for i in range(len(df))]
    return df[["post_id", "subreddit", "risk_tier", "wc",
               "suicidality_total", "sent_compound", "post_clean"]]
