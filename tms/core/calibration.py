from math import sqrt


def wilson_lower(successes, n, z=1.96):
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def at_threshold(samples, threshold):
    kept = [correct for conf, correct in samples if conf >= threshold]
    n = len(kept)
    good = sum(kept)
    return {"threshold": threshold, "n": n, "precision": good / n if n else 0.0,
            "lower": wilson_lower(good, n)}


def best_threshold(samples, target, min_n):
    for threshold in sorted({conf for conf, _ in samples}):
        row = at_threshold(samples, threshold)
        if row["n"] < min_n:
            return None
        if row["lower"] >= target:
            return row
    return None
