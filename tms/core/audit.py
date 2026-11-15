import hashlib
import json

from ..db import now

GENESIS = "0" * 64


def _digest(prev, at, user_id, action, target, details_json):
    payload = json.dumps([prev, at, user_id, action, target, details_json])
    return hashlib.sha256(payload.encode()).hexdigest()


def append(conn, user_id, action, target, details=None):
    details_json = json.dumps(details or {}, sort_keys=True)
    row = conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else GENESIS
    at = now()
    conn.execute(
        "INSERT INTO audit_log (at, user_id, action, target, details_json, prev_hash, hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (at, user_id, action, str(target), details_json, prev,
         _digest(prev, at, user_id, action, str(target), details_json)),
    )


def verify(conn):
    prev = GENESIS
    for row in conn.execute("SELECT * FROM audit_log ORDER BY id"):
        expected = _digest(prev, row["at"], row["user_id"], row["action"], row["target"],
                           row["details_json"])
        if row["prev_hash"] != prev or row["hash"] != expected:
            return row["id"]
        prev = row["hash"]
    return None
