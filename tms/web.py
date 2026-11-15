import csv
import hmac
import io
import json
import secrets
import shutil
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import config, db
from .core import audit, calibration, modelreg, plates, rules
from .worker import is_live

VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv"}
QUEUE = "(c.status = 'pending' OR (c.status = 'auto_issued' AND c.sampled = 1 " \
        "AND c.decided_by IS NULL))"
VISIBLE = "(j.status = 'done' OR j.live = 1)"
MAX_FAILURES = 5
LOCK_SECONDS = 300

hasher = PasswordHasher()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.globals["labels"] = rules.LABELS
failures = {}

db.init()
app = FastAPI(title="Helmet and seatbelt checker", docs_url=None, redoc_url=None,
              openapi_url=None)
app.add_middleware(SessionMiddleware, secret_key=config.secret(), same_site="strict",
                   https_only=config.HTTPS_ONLY, max_age=8 * 3600)


class LoginRequired(Exception):
    pass


@app.exception_handler(LoginRequired)
def to_login(request, error):
    return RedirectResponse("/login", 303)


def conn():
    connection = db.connect()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def current_user(request: Request, c=Depends(conn)):
    row = c.execute("SELECT * FROM users WHERE id = ? AND active = 1",
                    (request.session.get("uid"),)).fetchone()
    if row is None:
        request.session.pop("uid", None)
        raise LoginRequired()
    return row


def admin(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admins only")
    return user


def token(request):
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    return request.session["csrf"]


def csrf(request: Request, csrf_token: str = Form("")):
    if not hmac.compare_digest(csrf_token, request.session.get("csrf", "")) or not csrf_token:
        raise HTTPException(403, "Invalid form token")


def page(request, name, user=None, **context):
    return templates.TemplateResponse(
        request, name, {"user": user, "csrf_token": token(request), **context})


def model_versions():
    try:
        return modelreg.versions()
    except modelreg.ModelError:
        return {}


@app.get("/login")
def login_form(request: Request):
    return page(request, "login.html")


@app.post("/login", dependencies=[Depends(csrf)])
def login(request: Request, username: str = Form(...), password: str = Form(...),
          c=Depends(conn)):
    count, until = failures.get(username, (0, 0))
    if until > time.time():
        return page(request, "login.html", error="Too many attempts. Try again in a few minutes.")
    row = c.execute("SELECT * FROM users WHERE username = ? AND active = 1",
                    (username,)).fetchone()
    try:
        hasher.verify(row["password_hash"] if row else hasher.hash("-"), password)
        valid = row is not None
    except VerificationError:
        valid = False
    if not valid:
        count += 1
        failures[username] = (count, time.time() + LOCK_SECONDS if count >= MAX_FAILURES else 0)
        audit.append(c, None, "login_failed", f"user:{username}")
        return page(request, "login.html", error="Wrong username or password.")
    failures.pop(username, None)
    request.session.clear()
    request.session["uid"] = row["id"]
    audit.append(c, row["id"], "login", f"user:{username}")
    return RedirectResponse("/cases", 303)


@app.post("/logout", dependencies=[Depends(csrf)])
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", 303)


@app.get("/")
def home(user=Depends(current_user)):
    return RedirectResponse("/cases", 303)


@app.get("/upload")
def upload_form(request: Request, user=Depends(current_user), c=Depends(conn)):
    cameras = c.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    return page(request, "upload.html", user, cameras=cameras)


@app.post("/upload", dependencies=[Depends(csrf)])
def upload(request: Request, video: UploadFile, camera_id: int = Form(...),
           recorded_start: str = Form(...), user=Depends(current_user), c=Depends(conn)):
    cameras = c.execute("SELECT * FROM cameras ORDER BY name").fetchall()

    def fail(message):
        return page(request, "upload.html", user, cameras=cameras, error=message)

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_TYPES:
        return fail("Use an MP4, AVI, MOV or MKV file.")
    if not any(cam["id"] == camera_id for cam in cameras):
        return fail("Choose a camera.")
    try:
        started = datetime.fromisoformat(recorded_start)
    except ValueError:
        return fail("Enter when the recording started.")
    if shutil.disk_usage(config.DATA).free < config.MIN_FREE_BYTES + (video.size or 0):
        return fail("Not enough free disk space for this upload.")
    target = config.uploads() / f"{uuid.uuid4().hex}{suffix}"
    with open(target, "wb") as out:
        shutil.copyfileobj(video.file, out)
    job_id = c.execute(
        "INSERT INTO jobs (camera_id, source, recorded_start, created_by, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (camera_id, str(target), started.isoformat(timespec="seconds"), user["id"], db.now()),
    ).lastrowid
    audit.append(c, user["id"], "upload", f"job:{job_id}",
                 {"file": video.filename, "camera": camera_id})
    return RedirectResponse("/jobs", 303)


@app.get("/jobs")
def jobs(request: Request, user=Depends(current_user), c=Depends(conn)):
    rows = c.execute(
        "SELECT j.*, cam.name AS camera, "
        "(SELECT COUNT(*) FROM cases WHERE job_id = j.id) AS cases "
        "FROM jobs j JOIN cameras cam ON cam.id = j.camera_id ORDER BY j.id DESC LIMIT 100"
    ).fetchall()
    active = any(row["status"] in ("queued", "running", "stopping") for row in rows)
    return page(request, "jobs.html", user, jobs=rows, active=active)


@app.post("/jobs/{job_id}/stop", dependencies=[Depends(csrf)])
def stop_job(job_id: int, user=Depends(admin), c=Depends(conn)):
    c.execute("UPDATE jobs SET status = 'stopping' WHERE id = ? AND status = 'running'", (job_id,))
    audit.append(c, user["id"], "stop_job", f"job:{job_id}")
    return RedirectResponse("/jobs", 303)


@app.get("/cases")
def cases(request: Request, status: str = "queue", violation: str = "", camera: str = "",
          day: str = "", user=Depends(current_user), c=Depends(conn)):
    where, args = [VISIBLE], []
    if status == "queue":
        where.append(QUEUE)
    elif status:
        where.append("c.status = ?")
        args.append(status)
    if violation:
        where.append("c.violation = ?")
        args.append(violation)
    if camera.isdigit():
        where.append("c.camera_id = ?")
        args.append(int(camera))
    if day:
        where.append("substr(c.occurred_at, 1, 10) = ?")
        args.append(day)
    rows = c.execute(
        "SELECT c.*, cam.name AS camera FROM cases c JOIN jobs j ON j.id = c.job_id "
        f"JOIN cameras cam ON cam.id = c.camera_id WHERE {' AND '.join(where)} "
        "ORDER BY c.id DESC LIMIT 200", args).fetchall()
    cameras = c.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    return page(request, "cases.html", user, cases=rows, cameras=cameras,
                filters={"status": status, "violation": violation, "camera": camera, "day": day})


def load_case(c, case_id):
    row = c.execute(
        "SELECT c.*, cam.name AS camera, cam.location, u.username AS decider "
        "FROM cases c JOIN jobs j ON j.id = c.job_id JOIN cameras cam ON cam.id = c.camera_id "
        f"LEFT JOIN users u ON u.id = c.decided_by WHERE c.id = ? AND {VISIBLE}",
        (case_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Case not found")
    return row


def case_page(request, user, c, case, error=None):
    evidence = {row["kind"]: row for row in
                c.execute("SELECT * FROM evidence WHERE case_id = ?", (case["id"],))}
    open_case = case["decided_by"] is None and (
        case["status"] == "pending" or (case["status"] == "auto_issued" and case["sampled"]))
    return page(request, "case.html", user, case=case, evidence=evidence, open_case=open_case,
                votes=json.loads(case["votes_json"]),
                versions=json.loads(case["model_versions_json"]), error=error)


@app.get("/cases/{case_id}")
def case_detail(request: Request, case_id: int, user=Depends(current_user), c=Depends(conn)):
    return case_page(request, user, c, load_case(c, case_id))


@app.post("/cases/{case_id}/decide", dependencies=[Depends(csrf)])
def decide(request: Request, case_id: int, action: str = Form(...), plate: str = Form(""),
           reason: str = Form(""), user=Depends(current_user), c=Depends(conn)):
    case = load_case(c, case_id)
    if action not in ("approve", "reject"):
        raise HTTPException(400, "Unknown action")
    parsed = plates.parse(plate)
    reason = reason.strip()
    if action == "approve" and not parsed.valid:
        return case_page(request, user, c, case,
                         "The plate is not a valid Indian number. Correct it or reject the case.")
    if action == "reject" and not reason:
        return case_page(request, user, c, case, "Give a reason for rejecting.")
    approved = action == "approve"
    changed = c.execute(
        "UPDATE cases SET status = CASE WHEN ? = 0 THEN 'rejected' "
        "WHEN status = 'auto_issued' THEN 'auto_issued' ELSE 'approved' END, "
        "verdict = ?, decided_by = ?, decided_at = ?, reason = ?, plate = ?, plate_valid = ? "
        "WHERE id = ? AND decided_by IS NULL AND status IN ('pending', 'auto_issued')",
        (int(approved), int(approved), user["id"], db.now(), reason, parsed.text,
         int(parsed.valid), case_id)).rowcount
    if not changed:
        fresh = load_case(c, case_id)
        return case_page(request, user, c, fresh,
                         f"Already decided by {fresh['decider'] or 'the system'}.")
    audit.append(c, user["id"], action, f"case:{case_id}",
                 {"plate": parsed.text, "plate_before": case["plate"], "reason": reason})
    following = c.execute(
        f"SELECT c.id FROM cases c JOIN jobs j ON j.id = c.job_id WHERE {QUEUE} AND {VISIBLE} "
        "AND c.id != ? ORDER BY c.id LIMIT 1", (case_id,)).fetchone()
    return RedirectResponse(f"/cases/{following['id']}" if following else "/cases", 303)


@app.get("/evidence/{evidence_id}")
def evidence_file(evidence_id: int, user=Depends(current_user), c=Depends(conn)):
    row = c.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
    path = config.DATA / row["path"] if row else None
    if path is None or not path.is_file():
        raise HTTPException(404, "Evidence not found")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "private, no-store"})


@app.get("/cases/{case_id}/evidence.zip")
def case_zip(case_id: int, user=Depends(current_user), c=Depends(conn)):
    case = load_case(c, case_id)
    buffer = io.BytesIO()
    files = c.execute("SELECT * FROM evidence WHERE case_id = ?", (case_id,)).fetchall()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        summary = {key: case[key] for key in case.keys()}
        summary["evidence"] = [{"file": f"{f['kind']}.jpg", "sha256": f["sha256"]} for f in files]
        archive.writestr("case.json", json.dumps(summary, indent=2))
        for item in files:
            path = config.DATA / item["path"]
            if path.is_file():
                archive.write(path, f"{item['kind']}.jpg")
    audit.append(c, user["id"], "export_evidence", f"case:{case_id}")
    return Response(buffer.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="case-{case_id}.zip"'})


@app.get("/export.csv")
def export(user=Depends(current_user), c=Depends(conn)):
    rows = c.execute(
        "SELECT c.id, c.violation, c.occurred_at, c.plate, c.confidence, c.status, "
        "cam.name AS camera, cam.location, u.username AS decided_by, c.decided_at "
        "FROM cases c JOIN jobs j ON j.id = c.job_id JOIN cameras cam ON cam.id = c.camera_id "
        f"LEFT JOIN users u ON u.id = c.decided_by WHERE {VISIBLE} "
        "AND c.status IN ('approved', 'auto_issued') ORDER BY c.id").fetchall()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["case", "violation", "occurred_at", "plate", "confidence", "status",
                     "camera", "location", "decided_by", "decided_at"])
    writer.writerows(tuple(row) for row in rows)
    audit.append(c, user["id"], "export_csv", "cases", {"rows": len(rows)})
    return Response(buffer.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="cases.csv"'})


def calibration_state(c):
    versions = model_versions()
    checks = []
    for check, model in rules.CHECK_MODEL.items():
        version = versions.get(model)
        samples = [(row["confidence"], row["verdict"]) for row in c.execute(
            "SELECT confidence, verdict FROM cases WHERE violation = ? AND model_version = ? "
            "AND verdict IS NOT NULL", (check, version or ""))]
        current = c.execute(
            "SELECT * FROM calibration WHERE check_name = ? AND model_version = ?",
            (check, version or "")).fetchone()
        checks.append({
            "check": check, "model": model, "version": version, "reviewed": len(samples),
            "best": calibration.best_threshold(samples, config.PRECISION_TARGET,
                                               config.MIN_REVIEWED) if version else None,
            "table": [calibration.at_threshold(samples, t / 100) for t in range(50, 100, 5)],
            "enabled": bool(current and current["enabled"]),
            "threshold": current["threshold"] if current else None,
        })
    return checks


@app.get("/calibration")
def calibration_page(request: Request, user=Depends(admin), c=Depends(conn)):
    return page(request, "calibration.html", user, checks=calibration_state(c),
                target=config.PRECISION_TARGET, min_reviewed=config.MIN_REVIEWED,
                sample_rate=config.SAMPLE_RATE)


@app.post("/calibration/{check}", dependencies=[Depends(csrf)])
def set_auto_issue(check: str, enable: int = Form(...), user=Depends(admin), c=Depends(conn)):
    state = next((item for item in calibration_state(c) if item["check"] == check), None)
    if state is None:
        raise HTTPException(404, "Unknown check")
    if enable:
        best = state["best"]
        if best is None:
            raise HTTPException(400, "This check has not met the bar for automatic issuing")
        c.execute(
            "INSERT INTO calibration (check_name, model_version, threshold, precision_lb, n, "
            "enabled, changed_by, changed_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?) "
            "ON CONFLICT (check_name, model_version) DO UPDATE SET threshold = excluded.threshold, "
            "precision_lb = excluded.precision_lb, n = excluded.n, enabled = 1, "
            "changed_by = excluded.changed_by, changed_at = excluded.changed_at",
            (check, state["version"], best["threshold"], best["lower"], best["n"], user["id"],
             db.now()))
        audit.append(c, user["id"], "auto_issue_on", f"check:{check}",
                     {"threshold": best["threshold"], "precision_lb": best["lower"],
                      "n": best["n"], "model_version": state["version"]})
    else:
        c.execute("UPDATE calibration SET enabled = 0, changed_by = ?, changed_at = ? "
                  "WHERE check_name = ? AND model_version = ?",
                  (user["id"], db.now(), check, state["version"] or ""))
        audit.append(c, user["id"], "auto_issue_off", f"check:{check}")
    return RedirectResponse("/calibration", 303)


@app.get("/audit")
def audit_page(request: Request, user=Depends(admin), c=Depends(conn)):
    rows = c.execute(
        "SELECT a.*, u.username FROM audit_log a LEFT JOIN users u ON u.id = a.user_id "
        "ORDER BY a.id DESC LIMIT 500").fetchall()
    return page(request, "audit.html", user, entries=rows, broken=audit.verify(c))


@app.get("/admin")
def admin_page(request: Request, user=Depends(admin), c=Depends(conn), error: str = ""):
    return page(request, "admin.html", user, error=error,
                users=c.execute("SELECT * FROM users ORDER BY username").fetchall(),
                cameras=c.execute("SELECT * FROM cameras ORDER BY name").fetchall())


@app.post("/admin/users", dependencies=[Depends(csrf)])
def add_user(username: str = Form(...), password: str = Form(...), role: str = Form(...),
             user=Depends(admin), c=Depends(conn)):
    username = username.strip()
    if role not in ("officer", "admin") or not username or len(password) < 10:
        return RedirectResponse("/admin?error=Passwords+need+at+least+10+characters.", 303)
    if c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        return RedirectResponse("/admin?error=That+username+is+taken.", 303)
    c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
              (username, hasher.hash(password), role))
    audit.append(c, user["id"], "add_user", f"user:{username}", {"role": role})
    return RedirectResponse("/admin", 303)


@app.post("/admin/users/{user_id}/active", dependencies=[Depends(csrf)])
def set_active(user_id: int, active: int = Form(...), user=Depends(admin), c=Depends(conn)):
    if user_id == user["id"]:
        return RedirectResponse("/admin?error=You+cannot+deactivate+yourself.", 303)
    c.execute("UPDATE users SET active = ? WHERE id = ?", (int(bool(active)), user_id))
    audit.append(c, user["id"], "set_user_active", f"user_id:{user_id}", {"active": bool(active)})
    return RedirectResponse("/admin", 303)


@app.post("/admin/cameras", dependencies=[Depends(csrf)])
def add_camera(name: str = Form(...), location: str = Form(""), lat: str = Form(""),
               lon: str = Form(""), source_url: str = Form(""), user=Depends(admin),
               c=Depends(conn)):
    name, source_url = name.strip(), source_url.strip()
    if not name or c.execute("SELECT 1 FROM cameras WHERE name = ?", (name,)).fetchone():
        return RedirectResponse("/admin?error=Camera+names+must+be+unique.", 303)
    if source_url and not is_live(source_url):
        return RedirectResponse("/admin?error=Stream+addresses+start+with+rtsp+or+http.", 303)
    try:
        position = (float(lat) if lat else None, float(lon) if lon else None)
    except ValueError:
        return RedirectResponse("/admin?error=Latitude+and+longitude+must+be+numbers.", 303)
    c.execute("INSERT INTO cameras (name, location, lat, lon, source_url) VALUES (?, ?, ?, ?, ?)",
              (name, location.strip(), *position, source_url or None))
    audit.append(c, user["id"], "add_camera", f"camera:{name}")
    return RedirectResponse("/admin", 303)


@app.post("/admin/cameras/{camera_id}/start", dependencies=[Depends(csrf)])
def start_live(camera_id: int, user=Depends(admin), c=Depends(conn)):
    camera = c.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    if camera is None or not camera["source_url"]:
        raise HTTPException(404, "Camera has no stream address")
    busy = c.execute("SELECT 1 FROM jobs WHERE camera_id = ? AND live = 1 "
                     "AND status IN ('queued', 'running', 'stopping')", (camera_id,)).fetchone()
    if not busy:
        job_id = c.execute(
            "INSERT INTO jobs (camera_id, source, live, created_by, created_at) "
            "VALUES (?, ?, 1, ?, ?)",
            (camera_id, camera["source_url"], user["id"], db.now())).lastrowid
        audit.append(c, user["id"], "start_live", f"job:{job_id}", {"camera": camera_id})
    return RedirectResponse("/jobs", 303)
