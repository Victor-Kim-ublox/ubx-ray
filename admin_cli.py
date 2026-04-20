#!/usr/bin/env python3
"""
ubX-ray admin CLI.

Operates directly on the on-disk state (SQLite DB + uploads/ + outputs/) so it
can be run from the server shell without the web app being online. The web
app's cleanup code path is not imported -- we duplicate the minimal delete
semantics here to avoid pulling FastAPI and friends into a maintenance tool.

All destructive operations require --yes to proceed (dry-run otherwise).

Examples
--------
    # List everything, newest first
    python admin_cli.py list

    # List only one user's uploads
    python admin_cli.py list --user 3fe08a...

    # Show one result's on-disk footprint
    python admin_cli.py show <rid>

    # Delete by rid (dry-run first, then for real)
    python admin_cli.py delete <rid1> <rid2>
    python admin_cli.py delete <rid1> <rid2> --yes

    # Delete everything older than 7 days
    python admin_cli.py purge --older-than 7 --yes

    # Delete every result belonging to a specific user
    python admin_cli.py purge --user <uid> --yes

    # Delete EVERYTHING (requires --yes AND --i-mean-it)
    python admin_cli.py purge --all --yes --i-mean-it

    # Prune orphan files in uploads/ and outputs/ that have no matching DB row
    python admin_cli.py prune-orphans --yes
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH    = DATA_DIR / "ubxray.sqlite3"

RUNNING_STATUSES = ("queued", "running")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connect_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        die(f"DB not found at {DB_PATH}. Is this the right project root?")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                pass
    return total


def rid_footprint_bytes(rid: str) -> int:
    total = 0
    for p in UPLOAD_DIR.glob(f"{rid}_*"):
        total += dir_size(p)
    total += dir_size(OUTPUT_DIR / rid)
    return total


def parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Delete primitive
# ---------------------------------------------------------------------------

def delete_rid(conn: sqlite3.Connection, rid: str, *, dry_run: bool) -> int:
    """
    Remove uploads/{rid}_*, outputs/{rid}/, and the DB row. Returns bytes freed.
    Mirrors app.py::cln_delete_by_rid.
    """
    freed = 0
    for p in UPLOAD_DIR.glob(f"{rid}_*"):
        sz = dir_size(p)
        freed += sz
        print(f"  - {p}  ({fmt_bytes(sz)})")
        if not dry_run:
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            except Exception as e:
                print(f"    ! failed: {e}")

    out_dir = OUTPUT_DIR / rid
    if out_dir.exists():
        sz = dir_size(out_dir)
        freed += sz
        print(f"  - {out_dir}  ({fmt_bytes(sz)})")
        if not dry_run:
            shutil.rmtree(out_dir, ignore_errors=True)

    if not dry_run:
        conn.execute("DELETE FROM results WHERE id=?", (rid,))
    return freed


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    conn = connect_db()
    cur = conn.cursor()

    sql = "SELECT id, user_id, filename, uploaded_at, status, epoch_total FROM results"
    where, params = [], []
    if args.user:
        where.append("user_id = ?")
        params.append(args.user)
    if args.status:
        where.append("status = ?")
        params.append(args.status)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY uploaded_at DESC"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    rows = cur.execute(sql, params).fetchall()

    if not rows:
        print("(no results)")
        return

    print(f"{'rid':36}  {'user':32}  {'uploaded_at':20}  {'status':8}  {'size':>10}  filename")
    print("-" * 140)
    total = 0
    for r in rows:
        sz = rid_footprint_bytes(r["id"])
        total += sz
        print(
            f"{r['id']:36}  "
            f"{(r['user_id'] or '-'):32.32}  "
            f"{(r['uploaded_at'] or '-'):20.20}  "
            f"{(r['status'] or '-'):8.8}  "
            f"{fmt_bytes(sz):>10}  "
            f"{r['filename'] or '-'}"
        )
    print("-" * 140)
    print(f"{len(rows)} row(s), on-disk total: {fmt_bytes(total)}")


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> None:
    conn = connect_db()
    r = conn.execute("SELECT * FROM results WHERE id=?", (args.rid,)).fetchone()
    if not r:
        die(f"rid {args.rid} not in DB")

    print("DB row:")
    for k in r.keys():
        print(f"  {k}: {r[k]}")

    print("\nOn-disk files:")
    any_found = False
    for p in UPLOAD_DIR.glob(f"{args.rid}_*"):
        any_found = True
        print(f"  {p}  ({fmt_bytes(dir_size(p))})")
    out_dir = OUTPUT_DIR / args.rid
    if out_dir.exists():
        any_found = True
        print(f"  {out_dir}  ({fmt_bytes(dir_size(out_dir))})")
    if not any_found:
        print("  (none)")


# ---------------------------------------------------------------------------
# Subcommand: delete
# ---------------------------------------------------------------------------

def cmd_delete(args: argparse.Namespace) -> None:
    conn = connect_db()
    dry = not args.yes
    total_freed = 0

    for rid in args.rids:
        row = conn.execute(
            "SELECT id, status, filename FROM results WHERE id=?", (rid,)
        ).fetchone()
        if not row:
            # Allow deleting orphan files even if the DB row is already gone.
            any_files = any(UPLOAD_DIR.glob(f"{rid}_*")) or (OUTPUT_DIR / rid).exists()
            if not any_files:
                print(f"[skip] {rid}: not in DB and no files on disk")
                continue
            print(f"[orphan] {rid}: no DB row, removing files only")
        else:
            if row["status"] in RUNNING_STATUSES and not args.force:
                print(f"[skip] {rid}: status={row['status']} (use --force to override)")
                continue
            print(f"[{'DRY' if dry else 'DEL'}] {rid}  ({row['filename']})")

        total_freed += delete_rid(conn, rid, dry_run=dry)

    if not dry:
        conn.commit()

    print(f"\n{'Would free' if dry else 'Freed'}: {fmt_bytes(total_freed)}")
    if dry:
        print("(dry-run -- re-run with --yes to actually delete)")


# ---------------------------------------------------------------------------
# Subcommand: purge
# ---------------------------------------------------------------------------

def _collect_purge_targets(
    conn: sqlite3.Connection, args: argparse.Namespace
) -> list[sqlite3.Row]:
    sql = "SELECT id, user_id, filename, uploaded_at, status FROM results WHERE 1=1"
    params: list = []

    if not args.include_running:
        sql += f" AND (status IS NULL OR status NOT IN ({','.join('?'*len(RUNNING_STATUSES))}))"
        params.extend(RUNNING_STATUSES)

    if args.user:
        sql += " AND user_id = ?"
        params.append(args.user)

    if args.older_than is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than)
        sql += " AND uploaded_at < ?"
        # Stored as ISO string, lexicographic compare works for UTC ISO timestamps.
        params.append(cutoff.isoformat().replace("+00:00", "Z"))

    rows = conn.execute(sql, params).fetchall()

    if args.older_than is not None:
        # Belt-and-braces: re-check parsed timestamps, in case a row has a
        # non-ISO value that sneaked through the lexicographic filter.
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than)
        kept = []
        for r in rows:
            dt = parse_iso_utc(r["uploaded_at"]) or datetime(1970, 1, 1, tzinfo=timezone.utc)
            if dt < cutoff:
                kept.append(r)
        rows = kept

    return rows


def cmd_purge(args: argparse.Namespace) -> None:
    # At least one filter required -- otherwise require the extra confirmation.
    has_filter = args.user or args.older_than is not None
    if not has_filter and not args.all:
        die("purge needs one of --user, --older-than, or --all")
    if args.all and not args.i_mean_it:
        die("--all also requires --i-mean-it (this wipes EVERYTHING)")

    conn = connect_db()
    targets = _collect_purge_targets(conn, args)

    if not targets:
        print("(nothing matches the filter)")
        return

    dry = not args.yes
    total_freed = 0
    for r in targets:
        print(f"[{'DRY' if dry else 'DEL'}] {r['id']}  user={r['user_id'] or '-'}  "
              f"uploaded={r['uploaded_at']}  status={r['status'] or '-'}  ({r['filename']})")
        total_freed += delete_rid(conn, r["id"], dry_run=dry)

    if not dry:
        conn.commit()

    print(f"\n{len(targets)} row(s) matched. "
          f"{'Would free' if dry else 'Freed'}: {fmt_bytes(total_freed)}")
    if dry:
        print("(dry-run -- re-run with --yes to actually delete)")


# ---------------------------------------------------------------------------
# Subcommand: prune-orphans
# ---------------------------------------------------------------------------

def _rid_from_upload_name(name: str) -> Optional[str]:
    # Uploads are named "{rid}_{original_filename}". rid is the part before
    # the first underscore. We treat anything that doesn't match the pattern
    # as an orphan candidate to inspect.
    if "_" not in name:
        return None
    return name.split("_", 1)[0]


def cmd_prune_orphans(args: argparse.Namespace) -> None:
    conn = connect_db()
    db_rids = {r[0] for r in conn.execute("SELECT id FROM results").fetchall()}

    dry = not args.yes
    total_freed = 0

    # uploads/
    for p in UPLOAD_DIR.iterdir():
        if not p.is_file():
            continue
        rid = _rid_from_upload_name(p.name)
        if rid is None or rid in db_rids:
            continue
        sz = dir_size(p)
        total_freed += sz
        print(f"[{'DRY' if dry else 'DEL'}] uploads/{p.name}  ({fmt_bytes(sz)})")
        if not dry:
            p.unlink(missing_ok=True)

    # outputs/
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if not d.is_dir():
                continue
            if d.name in db_rids:
                continue
            sz = dir_size(d)
            total_freed += sz
            print(f"[{'DRY' if dry else 'DEL'}] outputs/{d.name}/  ({fmt_bytes(sz)})")
            if not dry:
                shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'Would free' if dry else 'Freed'}: {fmt_bytes(total_freed)}")
    if dry:
        print("(dry-run -- re-run with --yes to actually delete)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ubX-ray admin CLI: inspect and delete stored upload data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    pl = sub.add_parser("list", help="List uploads (newest first).")
    pl.add_argument("--user", help="Filter by user_id.")
    pl.add_argument("--status", help="Filter by status (done/error/queued/running).")
    pl.add_argument("--limit", type=int, help="Max rows to show.")
    pl.set_defaults(func=cmd_list)

    # show
    ps = sub.add_parser("show", help="Show DB row and on-disk files for one rid.")
    ps.add_argument("rid")
    ps.set_defaults(func=cmd_show)

    # delete
    pd = sub.add_parser("delete", help="Delete one or more rids.")
    pd.add_argument("rids", nargs="+")
    pd.add_argument("--yes", action="store_true", help="Actually delete (default: dry-run).")
    pd.add_argument("--force", action="store_true",
                    help="Also delete queued/running rows (default: skip).")
    pd.set_defaults(func=cmd_delete)

    # purge
    pp = sub.add_parser("purge", help="Bulk delete by filter (--user / --older-than / --all).")
    pp.add_argument("--user", help="Delete every result belonging to this user.")
    pp.add_argument("--older-than", type=int, metavar="DAYS",
                    help="Delete results uploaded more than N days ago.")
    pp.add_argument("--all", action="store_true",
                    help="Delete EVERY non-running result (requires --i-mean-it).")
    pp.add_argument("--i-mean-it", action="store_true",
                    help="Required safety toggle for --all.")
    pp.add_argument("--include-running", action="store_true",
                    help="Also match queued/running rows (default: skip).")
    pp.add_argument("--yes", action="store_true", help="Actually delete (default: dry-run).")
    pp.set_defaults(func=cmd_purge)

    # prune-orphans
    po = sub.add_parser("prune-orphans",
                        help="Remove files in uploads/ and outputs/ with no matching DB row.")
    po.add_argument("--yes", action="store_true", help="Actually delete (default: dry-run).")
    po.set_defaults(func=cmd_prune_orphans)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
