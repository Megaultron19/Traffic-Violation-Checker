import argparse
from pathlib import Path

import cv2

from ..core import geometry, modelreg
from ..models import Detector
from ..pipeline import MIN_CAR_WIDTH, MIN_HEAD, cut


def main():
    parser = argparse.ArgumentParser(prog="python -m tms.training.crops")
    parser.add_argument("video")
    parser.add_argument("out")
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.out)
    folders = {name: out / name for name in ("heads", "windscreens", "vehicles")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    detector = Detector(modelreg.resolve()["detector"].path)
    cap = cv2.VideoCapture(args.video)
    stem, idx, saved = Path(args.video).stem, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        detections = detector(frame)
        if idx % args.every:
            continue
        persons = {d.track_id: d.box for d in detections if d.kind == "person"}
        bikes = {d.track_id: d.box for d in detections if d.kind == "motorcycle"}
        crops = []
        for bike_id, riders in geometry.link_riders(persons, bikes).items():
            boxes = [persons[r] for r in riders]
            crops.append(("vehicles", bike_id, geometry.union([bikes[bike_id]] + boxes)))
            crops += [("heads", r, geometry.head_box(persons[r])) for r in riders
                      if geometry.head_box(persons[r])[3] - persons[r][1] >= MIN_HEAD]
        for d in detections:
            if d.kind == "car" and d.box[2] - d.box[0] >= MIN_CAR_WIDTH:
                crops.append(("vehicles", d.track_id, d.box))
                crops.append(("windscreens", d.track_id, geometry.windscreen_box(d.box)))
        for name, track_id, box in crops:
            image = cut(frame, box)
            if image.size:
                cv2.imwrite(str(folders[name] / f"{stem}_{idx:06d}_{track_id}.jpg"), image)
                saved += 1
    print(f"{idx} frames, {saved} crops written to {out}")


if __name__ == "__main__":
    main()
