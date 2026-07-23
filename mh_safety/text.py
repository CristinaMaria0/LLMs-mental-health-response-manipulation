"""Shared text utilities: PII scrubbing, VADER sentiment, lexical metrics."""
import re
from nltk.sentiment import SentimentIntensityAnalyzer

_user_re  = re.compile(r"(?:^|\s)/?u/[A-Za-z0-9_\-]+")
_sub_re   = re.compile(r"(?:^|\s)/?r/[A-Za-z0-9_\-]+")
_url_re   = re.compile(r"http\S+|www\.\S+")
_email_re = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# zero-width artifacts: the HTML entities plus the literal zero-width-space char
_zw_re    = re.compile("&amp;#x200b;|&#x200b;|​", re.I)


def scrub(text):
    """Replace usernames / subreddit refs / URLs / emails and collapse whitespace."""
    t = str(text)
    t = _zw_re.sub(" ", t)
    t = t.replace("&amp;", "&")
    t = _url_re.sub("[URL]", t)
    t = _email_re.sub("[EMAIL]", t)
    t = _user_re.sub(" [USER]", t)
    t = _sub_re.sub(" [SUB]", t)
    return re.sub(r"\s+", " ", t).strip()


def load_vader():
    """Return a function text -> VADER compound in [-1, 1]."""
    sia = SentimentIntensityAnalyzer()
    return lambda t: sia.polarity_scores(str(t))["compound"]


def lexical_metrics(text, vader_fn):
    """Model-free signals for a reply: sentiment, affect ratio, engagement, length."""
    t = str(text)
    low = t.lower()
    words = re.findall(r"[a-z']+", low)
    second = len(re.findall(r"\b(you|your|youre|you're)\b", low))
    return {
        "vader": vader_fn(t),
        "second_person": second,
        "questions": t.count("?"),
        "resp_words": len(words),
    }
