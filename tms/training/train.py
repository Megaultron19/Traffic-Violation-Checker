import argparse
import json
import shutil
from pathlib import Path

from ..core.modelreg import file_sha

BASES = {"helmet": "yolo11n-cls.pt", "seatbelt": "yolo11n-cls.pt", "plate": "yolo11n.pt"}
SIZES = {"helmet": 128, "seatbelt": 224, "plate": 640}


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(prog="python -m tms.training.train")
    parser.add_argument("kind", choices=BASES)
    parser.add_argument("data")
    parser.add_argument("version")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--base")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    model = YOLO(args.base or BASES[args.kind])
    result = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz or SIZES[args.kind],
                         project="runs", name=f"{args.kind}-{args.version}", exist_ok=True)
    target = Path(args.models) / f"{args.kind}-{args.version}.pt"
    shutil.copy(Path(result.save_dir) / "weights" / "best.pt", target)
    entry = {args.kind: {"path": target.name, "version": args.version, "sha256": file_sha(target)}}
    print(f"saved {target}\nadd this to {Path(args.models) / 'models.json'}:")
    print(json.dumps(entry, indent=2)[2:-2])


if __name__ == "__main__":
    main()
