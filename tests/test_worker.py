import json

import cv2
import pytest
from fakes import Scene, frame, labeller, plate_reader

from tms import config, db, worker
from tms.core import audit

VERSIONS = {"detector": "d1", "helmet": "h1", "seatbelt": "s1"}


def video(path, frames=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
    for _ in range(frames):
        writer.write(frame())
    writer.release()
    return str(path)


def parts(plate="UP32AB1234"):
    return {"detector": Scene(), "helmet": labeller("no_helmet"), "seatbelt": labeller("no_belt"),
            "plates": plate_reader(plate)}


def queue(conn, source):
    conn.execute("INSERT OR IGNORE INTO cameras (id, name) VALUES (1, 'Gate 1')")
    job_id = conn.execute(
        "INSERT INTO jobs (camera_id, source, recorded_start, created_at) VALUES (1, ?, ?, ?)",
        (source, "2026-11-01T10:00:00", db.now())).lastrowid
    conn.commit()
    return job_id


def test_job_creates_cases_with_evidence(conn, tmp_path):
    job_id = queue(conn, video(tmp_path / "clip.mp4"))
    assert worker.run_once(conn, parts(), VERSIONS)
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert job["status"] == "done" and job["progress"] == 1
    cases = conn.execute("SELECT * FROM cases ORDER BY violation").fetchall()
    assert [c["violation"] for c in cases] == ["no_helmet", "no_seatbelt"]
    case = cases[0]
    assert case["status"] == "pending" and case["plate"] == "UP32AB1234" and case["plate_valid"]
    assert case["occurred_at"].startswith("2026-11-01T10:00:00")
    assert case["model_version"] == "h1" and json.loads(case["model_versions_json"]) == VERSIONS
    files = conn.execute("SELECT * FROM evidence WHERE case_id = ?", (case["id"],)).fetchall()
    assert {f["kind"] for f in files} == {"frame", "crop", "plate"}
    assert all((config.DATA / f["path"]).is_file() and len(f["sha256"]) == 64 for f in files)
    assert not worker.run_once(conn, parts(), VERSIONS)


def test_bad_source_fails_the_job(conn, tmp_path):
    job_id = queue(conn, str(tmp_path / "missing.mp4"))
    worker.run_once(conn, parts(), VERSIONS)
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert job["status"] == "failed" and "could not open" in job["error"]


def test_stalled_job_is_rerun_without_duplicates(conn, tmp_path):
    job_id = queue(conn, video(tmp_path / "clip.mp4"))
    worker.run_once(conn, parts(), VERSIONS)
    video(tmp_path / "clip.mp4")
    conn.execute("UPDATE jobs SET status = 'running', heartbeat = '2000-01-01T00:00:00+00:00' "
                 "WHERE id = ?", (job_id,))
    conn.commit()
    worker.run_once(conn, parts(), VERSIONS)
    assert conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 2


def enable(conn, check, version, threshold=0.5):
    conn.execute("INSERT INTO calibration VALUES (?, ?, ?, 0.99, 400, 1, NULL, ?)",
                 (check, version, threshold, db.now()))
    conn.commit()


def statuses(conn):
    return {c["violation"]: c["status"] for c in conn.execute("SELECT * FROM cases")}


def test_auto_issue_only_for_calibrated_checks(conn, tmp_path):
    enable(conn, "no_helmet", "h1")
    queue(conn, video(tmp_path / "clip.mp4"))
    worker.run_once(conn, parts(), VERSIONS)
    assert statuses(conn) == {"no_helmet": "auto_issued", "no_seatbelt": "pending"}
    actions = [r["action"] for r in conn.execute("SELECT action FROM audit_log")]
    assert actions == ["auto_issue"] and audit.verify(conn) is None


@pytest.mark.parametrize("change", ["version", "threshold", "plate"])
def test_auto_issue_is_withheld(conn, tmp_path, change):
    enable(conn, "no_helmet", "h0" if change == "version" else "h1",
           0.99 if change == "threshold" else 0.5)
    queue(conn, video(tmp_path / "clip.mp4"))
    worker.run_once(conn, parts("??7" if change == "plate" else "UP32AB1234"), VERSIONS)
    assert statuses(conn)["no_helmet"] == "pending"


def test_retention_removes_old_rejected_evidence(conn, tmp_path):
    queue(conn, video(tmp_path / "clip.mp4"))
    worker.run_once(conn, parts(), VERSIONS)
    conn.execute("UPDATE cases SET status = 'rejected', decided_at = '2020-01-01T00:00:00+00:00' "
                 "WHERE violation = 'no_helmet'")
    conn.commit()
    assert worker.retention(conn) == 1
    kept = conn.execute("SELECT DISTINCT c.violation FROM evidence e JOIN cases c "
                        "ON c.id = e.case_id").fetchall()
    assert [k["violation"] for k in kept] == ["no_seatbelt"]
