# Traffic Violation Checker

Finds helmet, triple-riding and seatbelt violations in road footage, reads the number plate, and turns each one into a case that an officer reviews. Built for Indian roads and Indian number plates.

## How it works

1. A detector finds people, motorcycles and cars in each frame and follows them across frames.
2. Riders are matched to the motorcycle they are sitting on.
3. Small crops are checked by separate models: the rider's head (helmet), the driver's side of the windscreen (seatbelt) and the number plate (read with OCR, then checked against the Indian plate format).
4. A vehicle is judged across all the frames it appears in. One blurry frame never creates a case.
5. Each violation becomes a case with the best frame, the crops, the plate, the confidence and the model versions used. Evidence images are stored with a SHA-256 fingerprint.
6. Officers approve or reject cases in the web app. Every action goes into a tamper-evident audit log.

Automatic issuing is off by default. An admin can switch it on for one check only after at least 300 reviewed cases show, with 95% confidence, a precision of 99% or better. Even then, 5% of automatic cases still go to an officer, a case needs a valid plate to be issued automatically, and replacing a model resets its check.

## Models

| Model | Used for | Included |
|---|---|---|
| `detector` | People, motorcycles, cars | Downloaded on first run (YOLO11n, COCO) |
| `helmet` | `helmet` / `no_helmet` / `unclear` on head crops | No, train it |
| `seatbelt` | `belt` / `no_belt` / `unclear` on windscreen crops | No, train it |
| `plate` | Finding the number plate on a vehicle | No, train it |

Without the three trained models the system still runs, but it only finds triple riding and cases have no plate. A check starts working once its model is listed in `models/models.json`.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Skip the `torch` line on a machine with an NVIDIA GPU so the GPU build is installed.

Create the first admin:

```bash
python -m tms.manage create-user admin --role admin
```

Run the two processes in separate terminals:

```bash
python -m tms.worker
```

```bash
uvicorn tms.web:app --port 8000
```

Open http://localhost:8000, add a camera under Admin, then upload footage.

To check a video from the command line without the web app:

```bash
python -m tms.manage scan path/to/video.mp4
```

## Training the models

Cut crops out of your own footage for labelling:

```bash
python -m tms.training.crops path/to/video.mp4 crops
```

This writes `crops/heads`, `crops/windscreens` and `crops/vehicles`. Sort the head and windscreen crops into class folders (`train/helmet`, `train/no_helmet`, `train/unclear`, and the same under `val` and `test`). Draw plate boxes on the vehicle crops in a labelling tool and export in YOLO format.

Train and register a model:

```bash
python -m tms.training.train helmet data/helmet v1
```

```bash
python -m tms.training.train seatbelt data/seatbelt v1
```

```bash
python -m tms.training.train plate data/plate/data.yaml v1
```

Each command saves the weights into `models/` and prints the entry to add to `models/models.json`. Check a model on held-out footage from the target cameras before using it:

```bash
python -m tms.training.eval helmet models/helmet-v1.pt data/helmet/test
```

Restart the worker after changing `models/models.json`. The worker refuses to start if a listed model is missing or does not match its fingerprint.

## Live cameras

Give a camera a stream address (`rtsp://...`) under Admin and press "Start live processing". Each live camera occupies one worker, so start one worker per live camera plus one for uploads.

## Settings

Set these as environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `TMS_DATA` | `data` | Database, uploads and evidence |
| `TMS_MODELS` | `models/models.json` | Model registry |
| `TMS_SECRET` | generated | Session signing key |
| `TMS_HTTPS` | off | Set to `1` behind HTTPS so cookies are HTTPS-only |
| `TMS_KEEP_FOOTAGE` | off | Set to `1` to keep uploaded footage after processing |
| `TMS_REJECTED_DAYS` | `30` | Days before evidence of rejected cases is deleted |
| `TMS_PRECISION_TARGET` | `0.99` | Precision needed for automatic issuing |
| `TMS_MIN_REVIEWED` | `300` | Reviewed cases needed for automatic issuing |
| `TMS_SAMPLE_RATE` | `0.05` | Share of automatic cases sent to an officer |
| `TMS_MIN_FREE_GB` | `5` | Uploads and processing pause below this much free disk |

## Other commands

```bash
python -m tms.manage verify-audit
```

```bash
python -m tms.manage retention
```

## Docker

```bash
docker compose up --build
```

```bash
docker compose run --rm web python -m tms.manage create-user admin --role admin
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Privacy

Footage contains faces and number plates. Uploaded footage is deleted after processing unless told otherwise, evidence is only served to signed-in users, and nothing is stored beyond what a case needs. The system does not recognise faces.

## Author

Harshit Singh ([Megaultron19](https://github.com/Megaultron19))
