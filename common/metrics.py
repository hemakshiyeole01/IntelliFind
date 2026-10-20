import math


def precision_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def average_precision(ranked, rel):
    """AP over the retrieved list (depth 100); denominator = ALL relevant docs."""
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, gains, k):
    """Linear gain: DCG = sum gain_i / log2(i+1). Ideal list comes from all judged docs."""
    dcg = sum(gains.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
