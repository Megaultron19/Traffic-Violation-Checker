import re

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from tms import config, db, web

PASSWORD = "correct-horse-battery"


@pytest.fixture
def site(conn):
    web.failures.clear()
    digest = PasswordHasher().hash(PASSWORD)
    conn.executemany("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     [("asha", digest, "officer"), ("ravi", digest, "officer"),
                      ("boss", digest, "admin")])
    conn.execute("INSERT INTO cameras (id, name) VALUES (1, 'Gate 1')")
    conn.execute("INSERT INTO jobs (id, camera_id, source, status, created_at) "
                 "VALUES (1, 1, 'x.mp4', 'done', ?)", (db.now(),))
    conn.execute("INSERT INTO jobs (id, camera_id, source, status, created_at) "
                 "VALUES (2, 1, 'y.mp4', 'running', ?)", (db.now(),))
    conn.commit()
    return conn


def add_case(conn, confidence=0.9, job=1, plate="UP32AB1234", status="pending", sampled=0,
             verdict=None):
    case_id = conn.execute(
        "INSERT INTO cases (job_id, camera_id, track_id, violation, occurred_at, plate_raw, plate, "
        "plate_valid, confidence, votes_json, model_version, model_versions_json, status, sampled, "
        "verdict, decided_by) VALUES (?, 1, 1, 'triple_riding', '2026-11-01T10:00:00', ?, ?, 1, ?, "
        "'{\"hits\": 8, \"usable\": 10, \"total\": 12}', ?, '{}', ?, ?, ?, ?)",
        (job, plate, plate, confidence, web.model_versions()["detector"], status, sampled, verdict,
         1 if verdict is not None else None)).lastrowid
    conn.commit()
    return case_id


def token(client, path="/login"):
    return re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text).group(1)


def login(username):
    client = TestClient(web.app, follow_redirects=False)
    response = client.post("/login", data={"username": username, "password": PASSWORD,
                                           "csrf_token": token(client)})
    assert response.status_code == 303 and response.headers["location"] == "/cases"
    client.csrf = token(client, "/cases")
    return client


def test_pages_need_a_login(site):
    client = TestClient(web.app, follow_redirects=False)
    for path in ("/cases", "/jobs", "/upload", "/evidence/1", "/export.csv", "/audit"):
        response = client.get(path)
        assert response.status_code == 303 and response.headers["location"] == "/login"


def test_wrong_password_and_lockout(site):
    client = TestClient(web.app, follow_redirects=False)
    csrf = token(client)
    for _ in range(web.MAX_FAILURES):
        bad = client.post("/login", data={"username": "asha", "password": "nope",
                                          "csrf_token": csrf})
        assert "Wrong username or password" in bad.text
    locked = client.post("/login", data={"username": "asha", "password": PASSWORD,
                                         "csrf_token": csrf})
    assert "Too many attempts" in locked.text


def test_forms_need_the_csrf_token(site):
    client = login("asha")
    case_id = add_case(site)
    response = client.post(f"/cases/{case_id}/decide", data={"action": "approve",
                                                             "plate": "UP32AB1234"})
    assert response.status_code == 403


def test_admin_pages_are_admin_only(site):
    officer = login("asha")
    assert officer.get("/calibration").status_code == 403
    assert officer.get("/audit").status_code == 403
    assert officer.get("/admin").status_code == 403
    denied = officer.post("/calibration/triple_riding", data={"enable": 1,
                                                              "csrf_token": officer.csrf})
    assert denied.status_code == 403
    assert login("boss").get("/calibration").status_code == 200


def test_approve_then_second_officer_is_told(site):
    case_id = add_case(site)
    first, second = login("asha"), login("ravi")
    form = {"action": "approve", "plate": "up32 ab 1234"}
    done = first.post(f"/cases/{case_id}/decide", data={**form, "csrf_token": first.csrf})
    assert done.status_code == 303
    late = second.post(f"/cases/{case_id}/decide",
                       data={"action": "reject", "reason": "blurry", "csrf_token": second.csrf})
    assert "Already decided by asha" in late.text
    row = site.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    assert row["status"] == "approved" and row["verdict"] == 1 and row["plate"] == "UP32AB1234"


def test_approve_needs_valid_plate_and_reject_needs_reason(site):
    case_id = add_case(site)
    client = login("asha")
    bad_plate = client.post(f"/cases/{case_id}/decide",
                            data={"action": "approve", "plate": "HELLO", "csrf_token": client.csrf})
    assert "not a valid Indian number" in bad_plate.text
    no_reason = client.post(f"/cases/{case_id}/decide",
                            data={"action": "reject", "csrf_token": client.csrf})
    assert "Give a reason" in no_reason.text
    assert site.execute("SELECT status FROM cases").fetchone()["status"] == "pending"


def test_spot_check_can_overturn_an_automatic_case(site):
    case_id = add_case(site, status="auto_issued", sampled=1)
    client = login("asha")
    assert f"/cases/{case_id}" in client.get("/cases").text
    client.post(f"/cases/{case_id}/decide",
                data={"action": "reject", "reason": "helmet visible", "csrf_token": client.csrf})
    row = site.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    assert row["status"] == "rejected" and row["verdict"] == 0


def test_cases_of_unfinished_jobs_are_hidden(site):
    hidden = add_case(site, job=2)
    client = login("asha")
    assert f"/cases/{hidden}" not in client.get("/cases?status=").text
    assert client.get(f"/cases/{hidden}").status_code == 404


def test_evidence_and_export(site):
    case_id = add_case(site, status="approved")
    add_case(site, status="rejected")
    folder = config.evidence() / str(case_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "frame.jpg").write_bytes(b"\xff\xd8jpeg")
    evidence_id = site.execute(
        "INSERT INTO evidence (case_id, kind, path, sha256) VALUES (?, 'frame', ?, 'abc')",
        (case_id, str((folder / "frame.jpg").relative_to(config.DATA)))).lastrowid
    site.commit()
    client = login("asha")
    assert client.get(f"/evidence/{evidence_id}").content == b"\xff\xd8jpeg"
    assert client.get("/evidence/999").status_code == 404
    lines = client.get("/export.csv").text.strip().splitlines()
    assert len(lines) == 2 and "UP32AB1234" in lines[1]
    assert client.get(f"/cases/{case_id}/evidence.zip").content[:2] == b"PK"


def test_auto_issue_is_locked_until_the_bar_is_met(site):
    boss = login("boss")
    for _ in range(50):
        add_case(site, status="approved", verdict=1)
    early = boss.post("/calibration/triple_riding", data={"enable": 1, "csrf_token": boss.csrf})
    assert early.status_code == 400
    for _ in range(950):
        add_case(site, status="approved", verdict=1)
    assert boss.post("/calibration/triple_riding",
                     data={"enable": 1, "csrf_token": boss.csrf}).status_code == 303
    row = site.execute("SELECT * FROM calibration").fetchone()
    assert row["enabled"] == 1 and row["threshold"] == 0.9 and row["n"] == 1000
    boss.post("/calibration/triple_riding", data={"enable": 0, "csrf_token": boss.csrf})
    assert site.execute("SELECT enabled FROM calibration").fetchone()["enabled"] == 0


def test_admin_adds_users_and_cameras(site):
    boss = login("boss")
    boss.post("/admin/users", data={"username": "meena", "password": PASSWORD,
                                    "role": "officer", "csrf_token": boss.csrf})
    boss.post("/admin/cameras", data={"name": "Flyover", "source_url": "rtsp://10.0.0.5/main",
                                      "csrf_token": boss.csrf})
    assert login("meena").get("/cases").status_code == 200
    camera = site.execute("SELECT * FROM cameras WHERE name = 'Flyover'").fetchone()
    boss.post(f"/admin/cameras/{camera['id']}/start", data={"csrf_token": boss.csrf})
    job = site.execute("SELECT * FROM jobs WHERE live = 1").fetchone()
    assert job["source"] == "rtsp://10.0.0.5/main" and job["status"] == "queued"


def test_upload_queues_a_job(site):
    client = login("asha")
    response = client.post(
        "/upload", files={"video": ("clip.mp4", b"not really a video", "video/mp4")},
        data={"camera_id": 1, "recorded_start": "2026-11-01T10:00:00", "csrf_token": client.csrf})
    assert response.status_code == 303
    job = site.execute("SELECT * FROM jobs ORDER BY id DESC").fetchone()
    assert job["status"] == "queued" and job["source"].endswith(".mp4")
    wrong = client.post(
        "/upload", files={"video": ("notes.txt", b"x", "text/plain")},
        data={"camera_id": 1, "recorded_start": "2026-11-01T10:00:00", "csrf_token": client.csrf})
    assert "Use an MP4" in wrong.text
