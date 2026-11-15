import cv2

from .core.types import Detection, PlateRead

KINDS = {"person": "person", "motorcycle": "motorcycle", "car": "car"}


class Detector:
    def __init__(self, path, imgsz=960, conf=0.3):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.conf = conf
        self.classes = [i for i, name in self.model.names.items() if name in KINDS]

    def reset(self):
        for tracker in getattr(self.model.predictor, "trackers", None) or []:
            tracker.reset()

    def __call__(self, frame):
        boxes = self.model.track(frame, persist=True, imgsz=self.imgsz, conf=self.conf,
                                 classes=self.classes, tracker="bytetrack.yaml",
                                 verbose=False)[0].boxes
        if boxes.id is None:
            return []
        return [
            Detection(track_id, KINDS[self.model.names[cls]], tuple(box), conf)
            for box, track_id, cls, conf in zip(
                boxes.xyxy.int().tolist(), boxes.id.int().tolist(),
                boxes.cls.int().tolist(), boxes.conf.tolist())
        ]


class Classifier:
    def __init__(self, path, labels):
        from ultralytics import YOLO

        self.model = YOLO(str(path), task="classify")
        missing = set(labels) - set(self.model.names.values())
        if missing:
            raise ValueError(f"{path} lacks classes: {sorted(missing)}")

    def __call__(self, crops):
        if not crops:
            return []
        return [(self.model.names[r.probs.top1], float(r.probs.top1conf))
                for r in self.model(crops, verbose=False)]


class PlateReader:
    def __init__(self, path, conf=0.3):
        from rapidocr_onnxruntime import RapidOCR
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.conf = conf
        self.ocr = RapidOCR()

    def __call__(self, vehicle):
        boxes = self.model(vehicle, conf=self.conf, verbose=False)[0].boxes
        if not len(boxes):
            return None
        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].int().tolist()
        crop = vehicle[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        scale = max(1.0, 96 / crop.shape[0])
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        lines, _ = self.ocr(enlarged)
        if not lines:
            return None
        row = enlarged.shape[0] / 3
        lines.sort(key=lambda line: (round(line[0][0][1] / row), line[0][0][0]))
        text = "".join(line[1] for line in lines)
        score = min(float(line[2]) for line in lines)
        return PlateRead(text, min(score, float(boxes.conf[best])), crop)


def build(entries):
    parts = {"detector": Detector(entries["detector"].path)}
    if "helmet" in entries:
        parts["helmet"] = Classifier(entries["helmet"].path, ("helmet", "no_helmet", "unclear"))
    if "seatbelt" in entries:
        parts["seatbelt"] = Classifier(entries["seatbelt"].path, ("belt", "no_belt", "unclear"))
    if "plate" in entries:
        parts["plates"] = PlateReader(entries["plate"].path)
    return parts
