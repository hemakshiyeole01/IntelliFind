import re

_TOKEN = re.compile(r"[a-z0-9]+")        # keeps model numbers: s23, 128gb, 270
STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or the to with".split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """The ONLY tokenizer in the project. Lowercase, alnum tokens, no stemming."""
    toks = _TOKEN.findall((text or "").lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks
