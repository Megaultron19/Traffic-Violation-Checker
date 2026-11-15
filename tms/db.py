import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('officer', 'admin')),
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    location TEXT NOT NULL DEFAULT '',
    lat REAL,
    lon REAL,
    source_url TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id),
    source TEXT NOT NULL,
    live INTEGER NOT NULL DEFAULT 0,
    recorded_start TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'stopping', 'done', 'failed')),
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    heartbeat TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    camera_id INTEGER NOT NULL REFERENCES cameras(id),
    track_id INTEGER NOT NULL,
    violation TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    plate_raw TEXT NOT NULL DEFAULT '',
    plate TEXT NOT NULL DEFAULT '',
    plate_valid INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL,
    votes_json TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_versions_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'auto_issued')),
    sampled INTEGER NOT NULL DEFAULT 0,
    verdict INTEGER,
    decided_by INTEGER REFERENCES users(id),
    decided_at TEXT,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS cases_job ON cases(job_id);
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    kind TEXT NOT NULL CHECK (kind IN ('frame', 'crop', 'plate')),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration (
    check_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    threshold REAL NOT NULL,
    precision_lb REAL NOT NULL,
    n INTEGER NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    changed_by INTEGER REFERENCES users(id),
    changed_at TEXT NOT NULL,
    PRIMARY KEY (check_name, model_version)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    user_id INTEGER,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    details_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    config.DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.db_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init():
    conn = connect()
    conn.executescript(SCHEMA)
    conn.close()
