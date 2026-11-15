import re
from dataclasses import dataclass
from itertools import product

STANDARD = re.compile(r"^([A-Z]{2})[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
BHARAT = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
STATES = frozenset(
    "AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TN TR TS UA UK UP WB".split()
)
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
            "G": "6", "T": "7", "B": "8"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "T", "8": "B"}


@dataclass(frozen=True)
class Plate:
    raw: str
    text: str
    valid: bool


def is_valid(text):
    match = STANDARD.match(text)
    if match:
        return match.group(1) in STATES
    return bool(BHARAT.match(text))


def _masks(length):
    for digits, letters in product((1, 2), (0, 1, 2, 3)):
        if 2 + digits + letters + 4 == length:
            yield "L" * 2 + "D" * digits + "L" * letters + "D" * 4
    for letters in (1, 2):
        if 2 + 2 + 4 + letters == length:
            yield "DD" + "BH" + "DDDD" + "L" * letters


def _apply(text, mask):
    out, cost = [], 0
    for char, want in zip(text, mask):
        if want in "BH":
            fixed = want if char == want or TO_LETTER.get(char) == want else None
        elif want == "D":
            fixed = char if char.isdigit() else TO_DIGIT.get(char)
        else:
            fixed = char if char.isalpha() else TO_LETTER.get(char)
        if fixed is None:
            return None
        cost += fixed != char
        out.append(fixed)
    return "".join(out), cost


def parse(raw):
    text = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if text.startswith("IND") and len(text) >= 10:
        text = text[3:]
    if is_valid(text):
        return Plate(raw, text, True)
    candidates = {}
    for mask in _masks(len(text)):
        fixed = _apply(text, mask)
        if fixed and is_valid(fixed[0]):
            candidates[fixed[0]] = min(fixed[1], candidates.get(fixed[0], 99))
    if candidates:
        best = min(candidates.values())
        winners = [plate for plate, cost in candidates.items() if cost == best]
        if len(winners) == 1:
            return Plate(raw, winners[0], True)
    return Plate(raw, text, False)
