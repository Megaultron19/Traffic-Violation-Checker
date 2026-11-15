from dataclasses import dataclass
from statistics import fmean

CHECK_MODEL = {"no_helmet": "helmet", "triple_riding": "detector", "no_seatbelt": "seatbelt"}
LABELS = {"no_helmet": "No helmet", "triple_riding": "Triple riding", "no_seatbelt": "No seatbelt"}
CHECKS = {
    "motorcycle": (("no_helmet", "helmet", "no_helmet"), ("triple_riding", "riders", "triple")),
    "car": (("no_seatbelt", "seatbelt", "no_belt"),),
}
UNCLEAR = "unclear"


@dataclass(frozen=True)
class Decision:
    confidence: float
    hits: int
    usable: int
    total: int


def decide(votes, bad, min_usable=5, min_ratio=0.6):
    usable = [(label, conf) for label, conf in votes if label != UNCLEAR]
    if len(usable) < min_usable:
        return None
    hits = [conf for label, conf in usable if label == bad]
    ratio = len(hits) / len(usable)
    if ratio < min_ratio:
        return None
    return Decision(round(ratio * fmean(hits), 4), len(hits), len(usable), len(votes))


def violations(kind, votes, min_usable=5, min_ratio=0.6):
    found = []
    for violation, key, bad in CHECKS.get(kind, ()):
        decision = decide(votes.get(key, ()), bad, min_usable, min_ratio)
        if decision:
            found.append((violation, key, decision))
    return found
