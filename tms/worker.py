import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from . import config, db
from .core import audit, modelreg, rules
from .pipeline import Pipeline

LIVE_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://")
MAX_READ_FAILURES = 5


def is_live(source):
    return source.lower().startswith(LIVE_PREFIXES)


def remove_cases(conn, job_id):
    ids = [row["id"] for row in conn.execute("SELECT id FROM cases WHERE job_id = ?", (job_id,))]
    for case_id in ids:
        shutil.rmtree(config.evidence() / str(case_id), ignore_errors=True)
    conn.execute("DELETE FROM evidence WHERE case_id IN (SELECT id FROM cases WHERE job_id = ?)",
                 (job_id,))
    conn.execute("DELETE FROM cases WHERE job_id = ?", (job_id,))


def save_case(conn, job, finding, versions, occurred_at):
    model_version = versions.get(rules.CHECK_MODEL[finding.violation], "")
    plate = finding.plate
    row = conn.execute(
        "SELECT threshold FROM calibration WHERE check_name = ? AND model_version = ? "
        "AND enabled = 1", (finding.violation, model_version)).fetchone()
    auto = bool(row and plate and plate.valid and finding.decision.confidence >= row["threshold"])
    sampled = auto and random.random() < config.SAMPLE_RATE
    decision = finding.decision
    case_id = conn.execute(
        "INSERT INTO cases (job_id, camera_id, track_id, violation, occurred_at, plate_raw, "
        "plate, plate_valid, confidence, votes_json, model_version, model_versions_json, "
        "status, sampled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job["id"], job["camera_id"], finding.track_id, finding.violation, occurred_at,
         plate.raw if plate else "", plate.text if plate else "",
         int(bool(plate and plate.valid)), decision.confidence,
         json.dumps({"hits": decision.hits, "usable": decision.usable, "total": decision.total,
                     "frame": finding.frame_idx}),
         model_version, json.dumps(versions, sort_keys=True),
         "auto_issued" if auto else "pending", int(sampled)),
    ).lastrowid
    folder = config.evidence() / str(case_id)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, data in (("frame", finding.frame_jpg), ("crop", finding.crop_jpg),
                       ("plate", finding.plate_jpg)):
        if not data:
            continue
        path = folder / f"{kind}.jpg"
        path.write_bytes(data)
        conn.execute("INSERT INTO evidence (case_id, kind, path, sha256) VALUES (?, ?, ?, ?)",
                     (case_id, kind, str(path.relative_to(config.DATA)),
                      hashlib.sha256(data).hexdigest()))
    if auto:
        audit.append(conn, None, "auto_issue", f"case:{case_id}",
                     {"confidence": decision.confidence, "threshold": row["threshold"],
                      "sampled": sampled})
    return case_id


def wait_for_space(conn, job_id):
    while shutil.disk_usage(config.DATA).free < config.MIN_FREE_BYTES:
        conn.execute("UPDATE jobs SET heartbeat = ? WHERE id = ?", (db.now(), job_id))
        conn.commit()
        time.sleep(30)


def process_job(conn, job, parts, versions):
    live = bool(job["live"])
    cap = cv2.VideoCapture(job["source"])
    if not cap.isOpened():
        raise RuntimeError("could not open the video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = 0 if live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = None if live else datetime.fromisoformat(job["recorded_start"])
    if not live:
        remove_cases(conn, job["id"])
        conn.commit()
    if hasattr(parts["detector"], "reset"):
        parts["detector"].reset()
    pipe = Pipeline(parts["detector"], parts.get("helmet"), parts.get("seatbelt"),
                    parts.get("plates"), stale_after=int(fps * 2))

    def store(findings):
        for finding in findings:
            if live:
                at = db.now()
            else:
                at = (start + timedelta(seconds=finding.frame_idx / fps)).isoformat(
                    timespec="seconds")
            save_case(conn, job, finding, versions, at)

    idx, failures, stopped = 0, 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            if not live:
                break
            failures += 1
            if failures > MAX_READ_FAILURES:
                raise RuntimeError("lost the camera stream")
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(job["source"])
            continue
        failures = 0
        idx += 1
        store(pipe.step(idx, frame))
        if idx % 25 == 0:
            status = conn.execute("SELECT status FROM jobs WHERE id = ?", (job["id"],)).fetchone()
            if status["status"] == "stopping":
                stopped = True
                break
            conn.execute("UPDATE jobs SET progress = ?, heartbeat = ? WHERE id = ?",
                         (idx / total if total else 0, db.now(), job["id"]))
            conn.commit()
            wait_for_space(conn, job["id"])
    cap.release()
    store(pipe.flush())
    conn.execute("UPDATE jobs SET status = 'done', progress = 1, finished_at = ? WHERE id = ?",
                 (db.now(), job["id"]))
    conn.commit()
    source = Path(job["source"])
    if not live and not config.KEEP_FOOTAGE and config.uploads() in source.parents:
        source.unlink(missing_ok=True)
    return idx, stopped


def claim(conn):
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=config.HEARTBEAT_TIMEOUT)).isoformat(
        timespec="seconds")
    conn.execute("UPDATE jobs SET status = 'queued' WHERE status = 'running' AND heartbeat < ?",
                 (cutoff,))
    conn.commit()
    row = conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
    if row is None:
        return None
    claimed = conn.execute(
        "UPDATE jobs SET status = 'running', heartbeat = ?, error = NULL "
        "WHERE id = ? AND status = 'queued'", (db.now(), row["id"])).rowcount
    conn.commit()
    if not claimed:
        return None
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()


def retention(conn):
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=config.REJECTED_RETENTION_DAYS)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT DISTINCT c.id FROM cases c JOIN evidence e ON e.case_id = c.id "
        "WHERE c.status = 'rejected' AND c.decided_at < ?", (cutoff,)).fetchall()
    for row in rows:
        shutil.rmtree(config.evidence() / str(row["id"]), ignore_errors=True)
        conn.execute("DELETE FROM evidence WHERE case_id = ?", (row["id"],))
    if rows:
        audit.append(conn, None, "retention", "evidence", {"cases": [r["id"] for r in rows]})
    conn.commit()
    return len(rows)


def run_once(conn, parts, versions):
    job = claim(conn)
    if job is None:
        return False
    try:
        process_job(conn, job, parts, versions)
    except Exception as error:
        conn.rollback()
        conn.execute("UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
                     (str(error), db.now(), job["id"]))
        conn.commit()
    return True


def main():
    from . import models

    db.init()
    entries = modelreg.resolve()
    parts = models.build(entries)
    versions = modelreg.versions()
    conn = db.connect()
    print("worker ready:", ", ".join(f"{k}={v}" for k, v in versions.items()), flush=True)
    last_retention = 0.0
    while True:
        if time.time() - last_retention > 86400:
            retention(conn)
            last_retention = time.time()
        if not run_once(conn, parts, versions):
            time.sleep(2)


if __name__ == "__main__":
    main()
