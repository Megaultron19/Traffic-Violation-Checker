import argparse
import getpass
import sys

from argon2 import PasswordHasher

from . import db
from .core import audit


def create_user(args):
    password = args.password or getpass.getpass("Password: ")
    if len(password) < 10:
        sys.exit("Passwords need at least 10 characters.")
    conn = db.connect()
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                 (args.username, PasswordHasher().hash(password), args.role))
    audit.append(conn, None, "add_user", f"user:{args.username}", {"role": args.role, "via": "cli"})
    conn.commit()
    print(f"created {args.role} {args.username}")


def verify_audit(args):
    broken = audit.verify(db.connect())
    if broken is not None:
        sys.exit(f"audit log is broken at entry {broken}")
    print("audit log is intact")


def run_retention(args):
    from .worker import retention

    print(f"removed evidence for {retention(db.connect())} rejected cases")


def scan(args):
    import cv2

    from . import models
    from .core import modelreg
    from .pipeline import Pipeline

    parts = models.build(modelreg.resolve())
    pipe = Pipeline(parts["detector"], parts.get("helmet"), parts.get("seatbelt"),
                    parts.get("plates"))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    findings, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        findings += pipe.step(idx, frame)
    findings += pipe.flush()
    print(f"{idx} frames, {len(findings)} violations")
    for f in findings:
        plate = f.plate.text if f.plate else "-"
        print(f"track {f.track_id:>4}  {f.violation:<14} conf {f.decision.confidence:.2f}  "
              f"frames {f.decision.hits}/{f.decision.usable}  plate {plate}")


def main():
    parser = argparse.ArgumentParser(prog="python -m tms.manage")
    commands = parser.add_subparsers(required=True)
    user = commands.add_parser("create-user")
    user.add_argument("username")
    user.add_argument("--role", choices=("officer", "admin"), default="officer")
    user.add_argument("--password")
    user.set_defaults(run=create_user)
    commands.add_parser("verify-audit").set_defaults(run=verify_audit)
    commands.add_parser("retention").set_defaults(run=run_retention)
    video = commands.add_parser("scan")
    video.add_argument("video")
    video.set_defaults(run=scan)
    args = parser.parse_args()
    db.init()
    args.run(args)


if __name__ == "__main__":
    main()
