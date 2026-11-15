import argparse
from pathlib import Path

BAD = {"helmet": "no_helmet", "seatbelt": "no_belt"}
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def score(pairs, label):
    predicted = sum(1 for _, guess in pairs if guess == label)
    actual = sum(1 for truth, _ in pairs if truth == label)
    correct = sum(1 for truth, guess in pairs if truth == guess == label)
    return (correct / predicted if predicted else 0.0, correct / actual if actual else 0.0,
            actual)


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(prog="python -m tms.training.eval")
    parser.add_argument("kind", choices=("helmet", "seatbelt", "plate"))
    parser.add_argument("weights")
    parser.add_argument("data")
    args = parser.parse_args()

    if args.kind == "plate":
        metrics = YOLO(args.weights).val(data=args.data, verbose=False)
        print(f"precision {metrics.box.mp:.3f}  recall {metrics.box.mr:.3f}  "
              f"mAP50 {metrics.box.map50:.3f}")
        return
    model = YOLO(args.weights, task="classify")
    pairs = []
    for folder in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        files = [str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGES]
        for start in range(0, len(files), 64):
            for result in model(files[start:start + 64], verbose=False):
                pairs.append((folder.name, model.names[result.probs.top1]))
    print(f"{len(pairs)} images")
    for label in sorted({truth for truth, _ in pairs}):
        precision, recall, count = score(pairs, label)
        mark = "  <- violation class" if label == BAD[args.kind] else ""
        print(f"{label:<12} n {count:>5}  precision {precision:.3f}  recall {recall:.3f}{mark}")


if __name__ == "__main__":
    main()
