# app.py — multi-user safe (per-user isolation via cookie), UBX→KMZ web app
# - Anonymous user_id via secure cookie "ubx_user"
# - DB row carries user_id; all reads are filtered by user_id
# - Non-owner access -> 403
# - Optional admin override with Authorization: Bearer <ADMIN_TOKEN>
# - Fast quick summary, background conversion with concurrency limit
# - WAL SQLite, per-request connections, status lifecycle

import os
import sys
import io
import json
import glob
import shutil
import logging
import zipfile
import sqlite3
import asyncio
import contextlib
import uuid
import mmap
from datetime import datetime
from typing import Optional
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, UploadFile, File, Form, Request, Header
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ProcessPoolExecutor, as_completed

# 추가
import threading, time
from datetime import timedelta, timezone
from pathlib import Path

try:
    from pyubx2 import UBXReader
except Exception:
    UBXReader = None

try:
    import ubx2kmz
except Exception:
    ubx2kmz = None

# =========================
# Paths / logging / config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPL_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("ubxray")

app = FastAPI(title="UBX-ray")
app.mount("/static", StaticFiles(directory="static"), name="static")

# FastAPI startup 훅에 워커 등록
@app.on_event("startup")
def _start_cleanup_worker():
    t = threading.Thread(
        target=cln_loop,
        args=(get_db, UPLOAD_DIR, OUTPUT_DIR, DB_PATH),
        daemon=True
    )
    t.start()
    logger.info(
        f"[cleanup] started: interval={CLN_INTERVAL_SEC}s, TTL={CLN_RETAIN_DAYS}d, "
        f"per-user={CLN_MAX_RESULTS_PER_USER}, quota={CLN_MAX_TOTAL_BYTES/1024**3:.1f}GB"
    )

templates = Jinja2Templates(directory=TEMPL_DIR)

DB_PATH = os.path.join(DATA_DIR, "ubxray.sqlite3")
ADMIN_TOKEN = os.getenv("UBXRAY_ADMIN_TOKEN", "").strip()  # optional

COOKIE_NAME = "ubx_user"
COOKIE_SECURE = False  # set True in HTTPS
COOKIE_SAMESITE = "lax"

# ===== Cleanup 정책 =====
CLN_MAX_TOTAL_BYTES = 10 * 1024**3   # 10GB (uploads + outputs)
CLN_MAX_RESULTS_PER_USER = 10        # 사용자별 최신 10개만 보존 (queued/running 제외)
CLN_RETAIN_DAYS = 7                  # 업로드 후 7일 TTL
CLN_INTERVAL_SEC = 60 * 60           # 60분마다 한 번씩

# =========================
# DB utils (WAL + per-request conn)
# =========================
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA busy_timeout=5000;")
        # 테이블이 없으면 최신 스키마로 생성 (있어도 안전)
        c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            filename TEXT,
            uploaded_at TEXT,
            epoch_total INTEGER,
            epoch_missing INTEGER,
            crc_errors INTEGER,
            kmz_path TEXT,
            opts_json TEXT,
            status TEXT DEFAULT 'queued',
            error TEXT
        )
        """)
        c.commit()

    # ✅ 먼저 컬럼 보강(ALTER) 수행: 구버전 테이블에도 필요한 컬럼 추가
    ensure_columns()

    # ✅ 그 다음에 인덱스 생성 (컬럼이 존재함이 보장됨)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id, uploaded_at DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_uploaded ON results(uploaded_at DESC)")
        c.commit()

def ensure_columns():
    """기존 DB에 누락된 컬럼이 있으면 추가"""
    need = {
        "user_id": "TEXT",
        "opts_json": "TEXT",
        "status": "TEXT DEFAULT 'queued'",
        "error": "TEXT",
        "progress": "INTEGER DEFAULT 0",
    }
    with sqlite3.connect(DB_PATH) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(results)").fetchall()}
        for name, decl in need.items():
            if name not in cols:
                c.execute(f"ALTER TABLE results ADD COLUMN {name} {decl}")
        c.commit()

@contextlib.contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
    finally:
        conn.close()

init_db()

# =========================
# Middleware: ensure user cookie
# =========================
@app.middleware("http")
async def assign_user_cookie(request: Request, call_next):
    user_id = request.cookies.get(COOKIE_NAME)
    new_cookie = None
    if not user_id:
        new_cookie = uuid.uuid4().hex  # anonymous id
        # attach to request.state for handlers
        request.state.user_id = new_cookie
    else:
        request.state.user_id = user_id

    response = await call_next(request)
    if new_cookie:
        response.set_cookie(
            key=COOKIE_NAME,
            value=new_cookie,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            max_age=60*60*24*365,  # 1y
        )
    return response

def is_admin(auth_header: Optional[str]) -> bool:
    if not ADMIN_TOKEN:
        return False
    if not auth_header:
        return False
    if not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    return token == ADMIN_TOKEN

# =========================
# Concurrency control for conversions
# =========================
MAX_CONVERT = max(1, (os.cpu_count() or 2) // 2)  # half of cores
PROCESS_POOL = ProcessPoolExecutor(max_workers=MAX_CONVERT)
SEM = asyncio.Semaphore(MAX_CONVERT)

# =========================
# Quick UBX summary (fast, UBX-only)
# =========================
def quick_ubx_summary(filepath: str):
    """
    NAV-PVT(epoch)만 카운트:
      - UBX 프레임 헤더를 읽어 길이만큼 점프
      - CLASS=0x01, ID=0x07(PVT)인 프레임만 개수 += 1
    iTOW/CRC/누락 계산 안 함 → 최속/단순
    반환: {"epoch_total": <PVT count>, "epoch_missing": 0, "crc_errors": 0}
    """
    SYNC1, SYNC2 = 0xB5, 0x62
    CLS_NAV, ID_PVT = 0x01, 0x07

    pvt_count = 0

    with open(filepath, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        n = len(mm)
        i = 0
        while i + 6 <= n:
            # sync 검색
            if mm[i] != SYNC1 or mm[i+1] != SYNC2:
                i += 1
                continue

            # 헤더 최소길이 보장
            if i + 6 > n:
                break

            msg_class = mm[i+2]
            msg_id    = mm[i+3]
            length    = mm[i+4] | (mm[i+5] << 8)

            # 프레임 총 길이 = 6(header) + payload(len) + 2(ck)
            frame_len = 6 + length + 2

            # 파일 끝 넘기면 종료
            if i + frame_len > n:
                break

            # NAV-PVT만 카운트
            if msg_class == CLS_NAV and msg_id == ID_PVT:
                pvt_count += 1

            # 다음 프레임으로 점프
            i += frame_len

    return {
        "epoch_total": int(pvt_count),  # ✅ NAV-PVT 프레임 개수
        "epoch_missing": 0,
        "crc_errors": 0,
    }
# =========================
# ubx2kmz runner (no -o; find output via glob)
# =========================
def pick_ubx2kmz_script() -> Optional[str]:
    candidates = [
        os.path.join(BASE_DIR, "ubx2kmz.py"),
        os.path.join(BASE_DIR, "ubx2kmz_v1.4_mapm_acc_full4.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def worker_convert_task(filepath, rid, opts, db_path):
    """Worker process function (no async here)"""
    # Must create own connection in worker
    def _update_progress(pct):
        # SQLite WAL allows concurrent writes, but simple is best.
        # To avoid locking too much, update only every few percent?
        # But 60MB file is fast. Let's just try updating.
        try:
            with sqlite3.connect(db_path, timeout=10) as conn:
                conn.execute("UPDATE results SET progress=? WHERE id=?", (pct, rid))
                conn.commit()
        except Exception:
            pass

    try:
        if not ubx2kmz:
            raise ImportError("ubx2kmz module not available")

        # Call the imported logic directly
        ubx2kmz.run(
            filepath,
            hz=opts.get("hz"),
            use_nav2=opts.get("nav2"),
            alt_abs=opts.get("alt_abs"),
            verify_ck=opts.get("ck"),
            html=opts.get("html"),
            mapm=opts.get("mapm"),
            progress_cb=_update_progress
        )

        # Find result KMZ
        base, _ = os.path.splitext(filepath)
        out_dir = os.path.join(os.path.dirname(os.path.dirname(filepath)), "outputs", rid)
        # Note: ubx2kmz creates output in same dir as input or current dir?
        # looking at ubx2kmz line 965: kmz_path = base + suffix
        # It saves NEXT TO input file.

        # We need to move it to final output dir
        os.makedirs(out_dir, exist_ok=True)
        final_kmz = os.path.join(out_dir, "result.kmz")

        # Search logic similar to original run_ubx2kmz
        candidates = glob.glob(base + "*.kmz")
        # Exclude mapm only if we didn't ask for it?
        # Just pick the newest valid one
        valid = [c for c in candidates if os.path.getsize(c) > 0]
        if not valid:
            raise RuntimeError("No KMZ output found")

        src = max(valid, key=os.path.getmtime)
        shutil.move(src, final_kmz)

        # HTML report too if requested
        if opts.get("html"):
            html_candidates = glob.glob(base + "*.html")
            if html_candidates:
                hsrc = max(html_candidates, key=os.path.getmtime)
                # Copy or move? The original logic kept them. Let's move to clean up upload dir.
                # But waits: the original didn't move HTML to output dir explicitly in DB, just KMZ.
                # Let's keep it simple: just KMZ in DB.
                # Only move KMZ as before.
                pass

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE results SET kmz_path=?, status='done', progress=100, error=NULL WHERE id=?",
                (final_kmz, rid)
            )
            conn.commit()

    except Exception as e:
        logging.error(f"Worker failed: {e}")
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE results SET status='error', error=? WHERE id=?", (str(e), rid))
            conn.commit()

# =========================
# Async queue wrapper (limit concurrency)
# =========================
async def enqueue_convert(filepath, rid, **opts):
    with get_db() as conn:
        conn.execute("UPDATE results SET status='queued', error=NULL, progress=0 WHERE id=?", (rid,))
        conn.commit()

    loop = asyncio.get_event_loop()
    async with SEM:
        with get_db() as conn:
            conn.execute("UPDATE results SET status='running' WHERE id=?", (rid,))
            conn.commit()

        # Run in process pool
        await loop.run_in_executor(
            PROCESS_POOL,
            worker_convert_task,
            filepath, rid, opts, DB_PATH
        )

# =========================
# Helpers: ownership checks
# =========================
def worker_convert_task(filepath, rid, opts, db_path):
    """Worker process function (no async here)"""
    # Closure state for throttling
    state = {"last_update": 0}

    def _update_progress(pct):
        # Throttle updates: Max 1 update per 0.5 sec, unless it's 100% or 0%
        now = time.time()
        if pct < 100 and pct > 0:
            if now - state["last_update"] < 0.5:
                return

        state["last_update"] = now
        try:
            with sqlite3.connect(db_path, timeout=10) as conn:
                # Use execute directly, commit explicitly
                conn.execute("UPDATE results SET progress=? WHERE id=?", (pct, rid))
                conn.commit()
        except Exception as e:
            # Print error to stdout (caught by logger in parent?)
            # Or just ignore to avoid breaking the worker
            print(f"[Worker] Progress update failed: {e}")

def ensure_owner_or_404(rid: str, user_id: str, admin: bool = False):
    with get_db() as conn:
        row = conn.execute(
            "SELECT user_id FROM results WHERE id=?", (rid,)
        ).fetchone()
    if not row:
        return False, 404
    owner = row[0]
    if admin or owner == user_id:
        return True, 200
    return False, 403

# =========================
# Routes
# =========================
@app.get("/status/{rid}")
def get_status(rid: str, request: Request):
    """Simple status poll endpoint"""
    with get_db() as conn:
        row = conn.execute("SELECT status, progress, error, kmz_path FROM results WHERE id=?", (rid,)).fetchone()
    if not row:
        return {"status": "not_found", "progress": 0}

    st, pg, err, kp = row
    return {
        "status": st or "queued",
        "progress": pg or 0,
        "error": err,
        "ready": bool(kp and os.path.exists(kp))
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # cookie is set by middleware
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    # options (optional)
    hz: str = Form(""),
    nav2: bool = Form(False),
    alt_abs: bool = Form(False),
    ck: bool = Form(False),
    html: bool = Form(False),
    mapm: bool = Form(False),
):
    user_id = getattr(request.state, "user_id", None) or uuid.uuid4().hex

    rid = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[-8:]
    save_path = os.path.join(UPLOAD_DIR, f"{rid}_{file.filename}")

    # Chunked Write (Memory Protection)
    with open(save_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            f.write(chunk)

    logger.info(f"Uploaded by {user_id}: {save_path}")

    summary = quick_ubx_summary(save_path)

    hz_val: Optional[int] = None
    s = (hz or "").strip()
    if s.isdigit():
        hz_val = int(s)

    opts = {
        "hz": hz_val,
        "nav2": bool(nav2),
        "alt_abs": bool(alt_abs),
        "ck": bool(ck),
        "html": bool(html),
        "mapm": bool(mapm),
    }

    with get_db() as conn:
        conn.execute("""
            INSERT INTO results (id, user_id, filename, uploaded_at, epoch_total, epoch_missing, crc_errors,
                                 kmz_path, opts_json, status, error, progress)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, 0)
        """, (
            rid, user_id, file.filename, datetime.utcnow().isoformat(),
            summary["epoch_total"], summary["epoch_missing"], summary["crc_errors"],
            None, json.dumps(opts),
        ))
        conn.commit()

    asyncio.create_task(enqueue_convert(save_path, rid, **opts))

    return RedirectResponse(url=f"/report/{rid}", status_code=303)

@app.get("/report/{rid}", response_class=HTMLResponse)
def report(request: Request, rid: str, authorization: Optional[str] = Header(None)):
    user_id = getattr(request.state, "user_id", None)
    admin = is_admin(authorization)
    ok, code = ensure_owner_or_404(rid, user_id, admin)
    if not ok:
        if code == 403:
            return HTMLResponse("<h2>Forbidden</h2>", status_code=403)
        return HTMLResponse("<h2>Result not found</h2>", status_code=404)

    with get_db() as conn:
        row = conn.execute("""
            SELECT id, user_id, filename, uploaded_at, epoch_total, epoch_missing, crc_errors,
                   kmz_path, opts_json, status, error
            FROM results WHERE id=?
        """, (rid,)).fetchone()

    r = {
        "id": row[0], "user_id": row[1], "filename": row[2], "uploaded_at": row[3],
        "epoch_total": row[4], "epoch_missing": row[5], "crc_errors": row[6],
        "kmz_path": row[7], "opts_json": row[8], "status": row[9], "error": row[10],
    }
    if r["kmz_path"] and not os.path.exists(r["kmz_path"]):
        r["kmz_path"] = None

    return templates.TemplateResponse("report.html", {"request": request, "r": r})

@app.get("/recent", response_class=HTMLResponse)
def recent(request: Request, authorization: Optional[str] = Header(None)):
    user_id = getattr(request.state, "user_id", None)
    admin = is_admin(authorization)

    with get_db() as conn:
        if admin:
            rows = conn.execute("""
                SELECT id, filename, uploaded_at, epoch_total, epoch_missing, crc_errors,
                       kmz_path, opts_json, status, error
                FROM results ORDER BY uploaded_at DESC LIMIT 50
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, filename, uploaded_at, epoch_total, epoch_missing, crc_errors,
                       kmz_path, opts_json, status, error
                FROM results WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 50
            """, (user_id,)).fetchall()

    return templates.TemplateResponse("recent.html", {"request": request, "rows": rows})

@app.get("/map/{rid}", response_class=HTMLResponse)
def map_view(request: Request, rid: str, authorization: Optional[str] = Header(None)):
    user_id = getattr(request.state, "user_id", None)
    admin = is_admin(authorization)
    ok, code = ensure_owner_or_404(rid, user_id, admin)
    if not ok:
        if code == 403:
            return HTMLResponse("<h3>Forbidden</h3>", status_code=403)
        return HTMLResponse("<h3>No KMZ available</h3>", status_code=404)

    with get_db() as conn:
        row = conn.execute("SELECT filename, kmz_path FROM results WHERE id=?", (rid,)).fetchone()
    if not row or not row[1] or not os.path.exists(row[1]):
        return HTMLResponse("<h3>No KMZ available yet</h3>", status_code=404)
    return templates.TemplateResponse("map.html", {"request": request, "rid": rid, "filename": row[0]})

@app.get("/kml/{rid}")
def kml_preview(request: Request, rid: str, authorization: Optional[str] = Header(None)):
    user_id = getattr(request.state, "user_id", None)
    admin = is_admin(authorization)
    ok, code = ensure_owner_or_404(rid, user_id, admin)
    if not ok:
        return PlainTextResponse("Forbidden" if code == 403 else "Not found", status_code=code)

    with get_db() as conn:
        row = conn.execute("SELECT kmz_path FROM results WHERE id=?", (rid,)).fetchone()
    if not row or not row[0] or not os.path.exists(row[0]):
        return HTMLResponse("<h3>No KMZ available yet</h3>", status_code=404)

    try:
        with zipfile.ZipFile(row[0], "r") as z:
            with z.open("doc.kml") as f:
                kml_bytes = f.read()
        return Response(content=kml_bytes, media_type="application/vnd.google-earth.kml+xml")
    except Exception as e:
        logger.exception(f"KML extraction failed: {e}")
        return HTMLResponse(f"<h3>KML extraction failed: {e}</h3>", status_code=500)

@app.get("/download")
def download(request: Request, path: str, authorization: Optional[str] = Header(None)):
    """
    Only allow download if path belongs to the requesting user (or admin).
    We verify by matching path with results.kmz_path for that user.
    """
    user_id = getattr(request.state, "user_id", None)
    admin = is_admin(authorization)
    abs_path = os.path.abspath(path)

    # Validate path and ownership
    if not os.path.exists(abs_path):
        return HTMLResponse("<h3>File not found</h3>", status_code=404)

    with get_db() as conn:
        row = conn.execute(
            "SELECT user_id FROM results WHERE kmz_path=?",
            (abs_path,)
        ).fetchone()
    if not row:
        # not a known KMZ path of any job
        return HTMLResponse("<h3>Invalid file</h3>", status_code=400)

    owner = row[0]
    if not (admin or owner == user_id):
        return HTMLResponse("<h3>Forbidden</h3>", status_code=403)

    fname = os.path.basename(abs_path)
    return FileResponse(abs_path, filename=fname, media_type="application/octet-stream")


# ============== Cleanup helpers (prefix: cln_) ==============
def cln_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except FileNotFoundError:
                pass
    return total

def cln_safe_rm(p: Path):
    try:
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[cleanup] remove fail: {p} -> {e}")

def cln_delete_by_rid(cur, rid: str, upload_dir: str, output_dir: str):
    rid = str(rid)  # ⬅️ 안전 캐스팅
    for p in Path(upload_dir).glob(f"{rid}_*"):
        cln_safe_rm(p)
    cln_safe_rm(Path(output_dir) / rid)
    cur.execute("DELETE FROM results WHERE id=?", (rid,))


# ============== Cleanup passes (prefix: cln_) ==============
def cln_expired(get_db, upload_dir: str, output_dir: str):
    cutoff = datetime.now(timezone.utc) - timedelta(days=CLN_RETAIN_DAYS)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, uploaded_at, status FROM results
            WHERE (status IS NULL OR status NOT IN ('queued','running'))
        """)
        rows = cur.fetchall()
        targets = []
        for rid, up, st in rows:
            try:
                dt = datetime.fromisoformat((up or "").replace("Z", "+00:00"))
            except Exception:
                dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
            # ⬇️ 타임존 없는 경우 UTC로 간주
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                targets.append(rid)
        if targets:
            for rid in targets:
                cln_delete_by_rid(cur, rid, upload_dir, output_dir)
            conn.commit()
            logger.info(f"[cleanup] TTL deleted: {len(targets)}")


def cln_keep_latest_per_user(get_db, upload_dir: str, output_dir: str):
    """유저별 최신 N개만 남기기 (queued/running 제외)"""
    if not CLN_MAX_RESULTS_PER_USER:
        return
    with get_db() as conn:
        cur = conn.cursor()
        # user_id 없을 수도 있으니 예외 처리
        try:
            cur.execute("SELECT DISTINCT user_id FROM results")
            users = [r[0] for r in cur.fetchall()]
        except Exception:
            users = [None]

        # 빈값/NULL 그룹도 각각 처리
        for uid in (users or [None]):
            if uid is None:
                cur.execute("""
                  SELECT id FROM results
                  WHERE (user_id IS NULL) AND (status IS NULL OR status NOT IN ('queued','running'))
                  ORDER BY uploaded_at DESC
                """)
            elif uid == "":
                cur.execute("""
                  SELECT id FROM results
                  WHERE (user_id='') AND (status IS NULL OR status NOT IN ('queued','running'))
                  ORDER BY uploaded_at DESC
                """)
            else:
                cur.execute("""
                  SELECT id FROM results
                  WHERE user_id=? AND (status IS NULL OR status NOT IN ('queued','running'))
                  ORDER BY uploaded_at DESC
                """, (uid,))
            ids = [r[0] for r in cur.fetchall()]
            extra = ids[CLN_MAX_RESULTS_PER_USER:]
            if not extra:
                continue
            for rid in extra:
                cln_delete_by_rid(cur, rid, upload_dir, output_dir)
            conn.commit()
            logger.info(f"[cleanup] keep-latest user={uid} pruned {len(extra)}")

def cln_quota(get_db, upload_dir: str, output_dir: str):
    """디스크 10GB 초과 시 오래된 완료 결과부터 LRU 삭제"""
    used = cln_dir_size(Path(upload_dir)) + cln_dir_size(Path(output_dir))
    if used <= CLN_MAX_TOTAL_BYTES:
        return
    need = used - CLN_MAX_TOTAL_BYTES

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM results
            WHERE (status IS NULL OR status NOT IN ('queued','running'))
            ORDER BY uploaded_at ASC
        """)
        rows = [r[0] for r in cur.fetchall()]

        freed = 0
        for rid in rows:
            # 대략적인 용량 계산
            size = 0
            for p in Path(upload_dir).glob(f"{rid}_*"):
                try: size += p.stat().st_size
                except FileNotFoundError: pass
            out_dir = Path(output_dir) / rid
            if out_dir.exists():
                for root, _, files in os.walk(out_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        try: size += os.path.getsize(fp)
                        except FileNotFoundError: pass
            cln_delete_by_rid(cur, rid, upload_dir, output_dir)
            conn.commit()
            freed += size
            if freed >= need:
                break
        logger.info(f"[cleanup] quota freed ~= {freed/1024**3:.2f} GB")

def cln_orphans(get_db, upload_dir: str, output_dir: str):
    """고아 청소: DB에 없는 파일/폴더 제거 + 파일 없는 DB행 정리"""
    with get_db() as conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, kmz_path FROM results")
            rows = cur.fetchall()
        except Exception:
            return
        known = {r[0] for r in rows}

        # FS → DB에 없는 rid
        for p in Path(upload_dir).glob("*"):
            if "_" in p.name:
                rid = p.name.split("_", 1)[0]
                if rid and rid not in known:
                    cln_safe_rm(p)
        for d in Path(output_dir).glob("*"):
            if d.is_dir() and d.name not in known:
                cln_safe_rm(d)

        # DB → KMZ 파일 없는 행 kmz_path NULL 처리
        missing = [rid for rid, kmz in rows if kmz and not os.path.exists(kmz)]
        if missing:
            q = ",".join("?" for _ in missing)
            cur.execute(f"UPDATE results SET kmz_path=NULL WHERE id IN ({q})", missing)
            conn.commit()

def cln_run_once(get_db, upload_dir: str, output_dir: str, db_path: str):
    # 순서: TTL → 최신N개 → 용량상한 → 고아 → VACUUM
    try: cln_expired(get_db, upload_dir, output_dir)
    except Exception as e: logger.warning(f"[cleanup] TTL error: {e}")
    try: cln_keep_latest_per_user(get_db, upload_dir, output_dir)
    except Exception as e: logger.warning(f"[cleanup] keep-latest error: {e}")
    try: cln_quota(get_db, upload_dir, output_dir)
    except Exception as e: logger.warning(f"[cleanup] quota error: {e}")
    try: cln_orphans(get_db, upload_dir, output_dir)
    except Exception as e: logger.warning(f"[cleanup] orphans error: {e}")
    # VACUUM (가끔 해도 무방)
    try:
        with sqlite3.connect(db_path) as c:
            c.execute("VACUUM")
    except Exception as e:
        logger.warning(f"[cleanup] VACUUM fail: {e}")

def cln_loop(get_db, upload_dir: str, output_dir: str, db_path: str):
    cln_run_once(get_db, upload_dir, output_dir, db_path)
    while True:
        time.sleep(CLN_INTERVAL_SEC)
        cln_run_once(get_db, upload_dir, output_dir, db_path)
