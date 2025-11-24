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
import subprocess
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
from concurrent.futures import ThreadPoolExecutor

# 추가
import threading, time
from datetime import timedelta, timezone
from pathlib import Path

# ==== NMEA Analysis Module Import ====
try:
    from nmea_comparison import analyze_nmea_files
except ImportError:
    analyze_nmea_files = None
    logging.warning("nmea_comparison.py not found. NMEA analysis feature will be disabled.")

try:
    from pyubx2 import UBXReader
except Exception:
    UBXReader = None

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
MAX_UPLOAD_BYTES = int(os.getenv("UBXRAY_MAX_UPLOAD_MB", "300")) * 1024**2  # 300MB default
ALLOWED_UPLOAD_EXTS = {".ubx", ".bin"}

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
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONVERT)
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

def run_ubx2kmz(
    filepath: str,
    rid: str,
    hz: Optional[int] = None,
    nav2: bool = False,
    alt_abs: bool = False,
    ck: bool = False,
    mapm: bool = False,
    html: bool = False,
    timeout_sec: int = 1800,
):
    script_path = pick_ubx2kmz_script()
    if not script_path:
        raise FileNotFoundError("ubx2kmz script not found next to app.py")

    out_dir = os.path.join(OUTPUT_DIR, rid)
    os.makedirs(out_dir, exist_ok=True)
    final_kmz = os.path.join(out_dir, "result.kmz")

    in_path = os.path.abspath(filepath)
    args = [sys.executable, script_path, in_path]
    if hz is not None:
        args += ["--hz", str(hz)]
    if nav2:
        args += ["--nav2"]
    if alt_abs:
        args += ["--alt-abs"]  # adjust if your script uses '--abs'
    if ck:
        args += ["--ck"]
    if html:
        args += ["--html"]
    if mapm:
        args += ["--mapm"]

    logger.info("Running ubx2kmz: " + " ".join(args))
    try:
        p = subprocess.run(
            args, cwd=BASE_DIR, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
        if p.stdout:
            logger.info("[ubx2kmz stdout]\n" + p.stdout)
        if p.stderr:
            logger.warning("[ubx2kmz stderr]\n" + p.stderr)

        # Find auto-generated KMZ near input
        base, _ = os.path.splitext(in_path)
        candidates = []
        patterns = [
            base + "_mapm.kmz",
            base + "_nav2_abs_ck.kmz", base + "_nav2_abs_nock.kmz",
            base + "_nav2_ck.kmz",     base + "_nav2_nock.kmz",
            base + "_nav_abs_ck.kmz",  base + "_nav_abs_nock.kmz",
            base + "_nav_ck.kmz",      base + "_nav_nock.kmz",
            base + "*.kmz",
        ]
        for pat in patterns:
            candidates.extend(glob.glob(pat))
        candidates = [c for c in candidates if os.path.exists(c) and os.path.getsize(c) > 0]
        if not candidates:
            raise RuntimeError(f"No KMZ found after ubx2kmz (returncode={p.returncode})")

        src = max(candidates, key=os.path.getmtime)
        shutil.copyfile(src, final_kmz)

        with get_db() as conn:
            conn.execute(
                "UPDATE results SET kmz_path=?, status='done', error=NULL WHERE id=?",
                (final_kmz, rid),
            )
            conn.commit()
        logger.info(f"[✅] KMZ generated: {final_kmz}")

    except subprocess.TimeoutExpired:
        with get_db() as conn:
            conn.execute("UPDATE results SET status='error', error=? WHERE id=?", ("timeout", rid))
            conn.commit()
        logger.error(f"ubx2kmz timed out after {timeout_sec}s")
    except Exception as e:
        with get_db() as conn:
            conn.execute("UPDATE results SET status='error', error=? WHERE id=?", (str(e), rid))
            conn.commit()
        logger.exception(f"[❌] ubx2kmz failed: {e}")

# =========================
# Async queue wrapper (limit concurrency)
# =========================
async def enqueue_convert(filepath, rid, **opts):
    with get_db() as conn:
        conn.execute("UPDATE results SET status='queued', error=NULL WHERE id=?", (rid,))
        conn.commit()
    async with SEM:
        with get_db() as conn:
            conn.execute("UPDATE results SET status='running', error=NULL WHERE id=?", (rid,))
            conn.commit()
        await run_in_threadpool(run_ubx2kmz, filepath, rid, **opts)

# =========================
# Helpers: ownership checks
# =========================
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

    # ===== Upload validation (size + extension + path safety) =====
    clean_name = os.path.basename(file.filename or "")
    if not clean_name:
        clean_name = "upload.ubx"

    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return PlainTextResponse(
            f"Unsupported file type: {ext or '(none)'}; allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTS))}",
            status_code=400,
        )

    content_len = request.headers.get("content-length")
    if content_len:
        try:
            if int(content_len) > MAX_UPLOAD_BYTES:
                return PlainTextResponse("File too large", status_code=413)
        except ValueError:
            pass

    save_path = os.path.join(UPLOAD_DIR, f"{rid}_{clean_name}")
    total = 0
    chunk_size = 1024 * 1024  # 1MB
    with open(save_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                with contextlib.suppress(FileNotFoundError):
                    os.remove(save_path)
                return PlainTextResponse("File too large", status_code=413)
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
                                 kmz_path, opts_json, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL)
        """, (
            rid, user_id, clean_name, datetime.utcnow().isoformat(),
            summary["epoch_total"], summary["epoch_missing"], summary["crc_errors"],
            None, json.dumps(opts),
        ))
        conn.commit()

    asyncio.create_task(enqueue_convert(save_path, rid, **opts))

    return RedirectResponse(url=f"/report/{rid}", status_code=303)

# =========================
# NEW ROUTE: NMEA Analysis
# =========================
@app.post("/analyze_nmea", response_class=HTMLResponse)
async def analyze_nmea(
    request: Request,
    ref_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    if not analyze_nmea_files:
        return HTMLResponse("<h3>NMEA Analysis module not loaded.</h3>", status_code=500)

    user_id = getattr(request.state, "user_id", None) or uuid.uuid4().hex
    rid = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[-8:]

    # 1. Save files
    ref_name = os.path.basename(ref_file.filename or "ref.nmea")
    test_name = os.path.basename(test_file.filename or "test.nmea")
    
    ref_path = os.path.join(UPLOAD_DIR, f"{rid}_ref_{ref_name}")
    test_path = os.path.join(UPLOAD_DIR, f"{rid}_test_{test_name}")

    try:
        # Write files (chunked write for safety)
        with open(ref_path, "wb") as f:
            while True:
                chunk = await ref_file.read(1024 * 1024)
                if not chunk: break
                f.write(chunk)
        
        with open(test_path, "wb") as f:
            while True:
                chunk = await test_file.read(1024 * 1024)
                if not chunk: break
                f.write(chunk)
            
        logger.info(f"NMEA Analysis requested by {user_id}: {ref_name} vs {test_name}")

        # 2. Run Analysis (in threadpool to not block async loop)
        result = await run_in_threadpool(analyze_nmea_files, ref_path, test_path)

        if result['status'] == 'error':
            return HTMLResponse(f"<h3>Analysis Error: {result['message']}</h3>", status_code=400)

        # 3. Render Report
        return templates.TemplateResponse("report_nmea.html", {
            "request": request,
            "stats": result['statistics'],
            "graph_data": json.dumps(result['graph_data'])
        })

    except Exception as e:
        logger.exception(f"NMEA Analysis failed: {e}")
        return HTMLResponse(f"<h3>Internal Error: {str(e)}</h3>", status_code=500)


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
