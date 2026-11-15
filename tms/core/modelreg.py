import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import config

OPTIONAL = ("helmet", "seatbelt", "plate")


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    path: Path
    version: str


@lru_cache(maxsize=32)
def _sha(path, mtime, size):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha(path):
    stat = Path(path).stat()
    return _sha(str(path), stat.st_mtime_ns, stat.st_size)


def resolve(registry=None):
    registry = Path(registry or config.MODELS)
    if not registry.exists():
        raise ModelError(f"model registry not found: {registry}")
    spec = json.loads(registry.read_text())
    if "detector" not in spec:
        raise ModelError("model registry has no 'detector' entry")
    entries = {}
    for name, item in spec.items():
        path = registry.parent / item["path"]
        wanted = item.get("sha256")
        version = item.get("version", "unversioned")
        if path.exists():
            actual = file_sha(path)
            if wanted and actual != wanted:
                raise ModelError(f"model '{name}' does not match its sha256: {path}")
            version = f"{version}+{actual[:12]}"
        elif wanted or name != "detector":
            raise ModelError(f"model '{name}' file is missing: {path}")
        entries[name] = Entry(name, path, version)
    return entries


def versions(registry=None):
    return {name: entry.version for name, entry in resolve(registry).items()}
