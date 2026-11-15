from dataclasses import dataclass, field

import cv2

from .core import geometry, plates, rules

MIN_HEAD = 16
MIN_CAR_WIDTH = 160


@dataclass
class Best:
    conf: float
    frame_idx: int
    frame_jpg: bytes
    crop_jpg: bytes


@dataclass
class Track:
    id: int
    kind: str
    last_seen: int
    votes: dict = field(default_factory=dict)
    best: dict = field(default_factory=dict)
    plate: object = None
    plate_conf: float = 0.0
    plate_jpg: bytes = b""


@dataclass
class Finding:
    track_id: int
    kind: str
    violation: str
    decision: rules.Decision
    frame_idx: int
    frame_jpg: bytes
    crop_jpg: bytes
    plate: object
    plate_jpg: bytes


def jpg(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return data.tobytes() if ok else b""


def cut(frame, box):
    height, width = frame.shape[:2]
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(width, box[2]), min(height, box[3])
    return frame[y1:y2, x1:x2]


class Pipeline:
    def __init__(self, detector, helmet=None, seatbelt=None, plates=None,
                 stale_after=50, min_usable=5, min_ratio=0.6):
        self.detector = detector
        self.helmet = helmet
        self.seatbelt = seatbelt
        self.plates = plates
        self.stale_after = stale_after
        self.min_usable = min_usable
        self.min_ratio = min_ratio
        self.tracks = {}

    def step(self, idx, frame):
        detections = self.detector(frame)
        persons = {d.track_id: d for d in detections if d.kind == "person"}
        bikes = {d.track_id: d for d in detections if d.kind == "motorcycle"}
        links = geometry.link_riders({i: d.box for i, d in persons.items()},
                                     {i: d.box for i, d in bikes.items()})
        for bike_id, rider_ids in links.items():
            self._bike(idx, frame, bikes[bike_id], [persons[i] for i in rider_ids])
        for car in detections:
            if car.kind == "car":
                self._car(idx, frame, car)
        stale = [t for t in self.tracks.values() if idx - t.last_seen > self.stale_after]
        return [finding for track in stale for finding in self._close(track)]

    def flush(self):
        return [finding for track in list(self.tracks.values())
                for finding in self._close(track)]

    def _track(self, det, idx):
        track = self.tracks.get(det.track_id)
        if track is None:
            track = self.tracks[det.track_id] = Track(det.track_id, det.kind, idx)
        track.last_seen = idx
        return track

    def _vote(self, track, key, label, conf, bad, idx, frame, box, crop_box):
        track.votes.setdefault(key, []).append((label, conf))
        if label != bad:
            return False
        best = track.best.get(key)
        if best is None or conf > best.conf:
            marked = frame.copy()
            cv2.rectangle(marked, box[:2], box[2:], (0, 0, 255), 2)
            cv2.rectangle(marked, crop_box[:2], crop_box[2:], (0, 255, 255), 2)
            track.best[key] = Best(conf, idx, jpg(marked), jpg(cut(frame, crop_box)))
        return True

    def _bike(self, idx, frame, bike, riders):
        track = self._track(bike, idx)
        box = geometry.union([bike.box] + [r.box for r in riders])
        label = "triple" if len(riders) >= 3 else "ok"
        flagged = self._vote(track, "riders", label, min(r.conf for r in riders), "triple",
                             idx, frame, box, box)
        if self.helmet:
            heads = [geometry.head_box(r.box) for r in riders]
            usable = [h for h in heads if h[3] - h[1] >= MIN_HEAD and cut(frame, h).size]
            results = self.helmet([cut(frame, h) for h in usable])
            bare = [(conf, head) for head, (name, conf) in zip(usable, results)
                    if name == "no_helmet"]
            if bare:
                conf, head = max(bare)
                flagged |= self._vote(track, "helmet", "no_helmet", conf, "no_helmet",
                                      idx, frame, box, head)
            elif results and len(usable) == len(heads) and all(n == "helmet" for n, _ in results):
                self._vote(track, "helmet", "helmet", min(c for _, c in results), "no_helmet",
                           idx, frame, box, box)
            else:
                self._vote(track, "helmet", rules.UNCLEAR, 0.0, "no_helmet", idx, frame, box, box)
        if flagged:
            self._read_plate(track, frame, box)

    def _car(self, idx, frame, car):
        if not self.seatbelt or car.box[2] - car.box[0] < MIN_CAR_WIDTH:
            return
        track = self._track(car, idx)
        screen = geometry.windscreen_box(car.box)
        crop = cut(frame, screen)
        if not crop.size:
            return
        label, conf = self.seatbelt([crop])[0]
        if self._vote(track, "seatbelt", label, conf, "no_belt", idx, frame, car.box, screen):
            self._read_plate(track, frame, car.box)

    def _read_plate(self, track, frame, box):
        if not self.plates:
            return
        vehicle = cut(frame, box)
        if not vehicle.size:
            return
        read = self.plates(vehicle)
        if read is None:
            return
        plate = plates.parse(read.text)
        current = (track.plate.valid, track.plate_conf) if track.plate else (False, -1.0)
        if (plate.valid, read.conf) > current:
            track.plate, track.plate_conf, track.plate_jpg = plate, read.conf, jpg(read.crop)

    def _close(self, track):
        del self.tracks[track.id]
        findings = []
        for violation, key, decision in rules.violations(
                track.kind, track.votes, self.min_usable, self.min_ratio):
            best = track.best[key]
            findings.append(Finding(track.id, track.kind, violation, decision, best.frame_idx,
                                    best.frame_jpg, best.crop_jpg, track.plate, track.plate_jpg))
        return findings
