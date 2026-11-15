from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    track_id: int
    kind: str
    box: tuple
    conf: float


@dataclass(frozen=True)
class PlateRead:
    text: str
    conf: float
    crop: object
