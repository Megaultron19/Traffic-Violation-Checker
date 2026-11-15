import os
import secrets
from pathlib import Path

DATA = Path(os.environ.get("TMS_DATA", "data")).resolve()
MODELS = Path(os.environ.get("TMS_MODELS", "models/models.json"))
KEEP_FOOTAGE = os.environ.get("TMS_KEEP_FOOTAGE") == "1"
HTTPS_ONLY = os.environ.get("TMS_HTTPS") == "1"
MIN_FREE_BYTES = int(os.environ.get("TMS_MIN_FREE_GB", "5")) * 1024**3
REJECTED_RETENTION_DAYS = int(os.environ.get("TMS_REJECTED_DAYS", "30"))
PRECISION_TARGET = float(os.environ.get("TMS_PRECISION_TARGET", "0.99"))
MIN_REVIEWED = int(os.environ.get("TMS_MIN_REVIEWED", "300"))
SAMPLE_RATE = float(os.environ.get("TMS_SAMPLE_RATE", "0.05"))
HEARTBEAT_TIMEOUT = 120


def db_path():
    return DATA / "tms.db"


def uploads():
    path = DATA / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evidence():
    path = DATA / "evidence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def secret():
    value = os.environ.get("TMS_SECRET")
    if value:
        return value
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "secret.key"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()
