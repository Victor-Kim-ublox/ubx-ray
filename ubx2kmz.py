#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UBX (NAV-PVT / NAV2-PVT) -> KMZ
- Default: NAV-PVT only (class=0x01,id=0x07)
- Option: --nav2          => NAV2-PVT only (class=0x29,id=0x07)
- Option: --hz {1,2,5,10} => keep only frames aligned to the given Hz (iTOW ms)
- Option: --alt-abs       => set altitudeMode=absolute with extrude=1 in <Point>
- Option: --ck            => enable UBX checksum verification (default: off)
"""

import argparse
import bisect
import os
import time
import mmap
import struct
import math
import zipfile
import json
from collections import defaultdict
from datetime import datetime, timedelta

UBX_SYNC1 = 0xB5
UBX_SYNC2 = 0x62
UBX_SYNC  = b"\xB5\x62"  # sync pattern for C-speed mmap.find() scanning
NAV_CLASS  = 0x01
NAV2_CLASS = 0x29
PVT_ID     = 0x07
SAT_ID     = 0x35
SEC_CLASS  = 0x27
SECSIG_ID  = 0x09

# UBX-INF (class 0x04): informational text messages emitted by the receiver.
# Payload is ASCII. Surfaced in the report so users can see what the receiver
# reported (errors/warnings/notices) and at which point in the log.
INF_CLASS  = 0x04
INF_LEVELS = {0x00: "ERROR", 0x01: "WARNING", 0x02: "NOTICE", 0x03: "TEST", 0x04: "DEBUG"}
INF_MAX    = 2000  # cap collected INF messages to keep the graph JSON bounded

# SEC-SIG jamming / spoofing state codes
JAM_STATE_UNKNOWN   = 0
JAM_STATE_OK        = 1
JAM_STATE_WARNING   = 2

SPF_STATE_UNKNOWN   = 0
SPF_STATE_OK        = 1
SPF_STATE_INDICATED = 2
SPF_STATE_AFFIRMED  = 3

VALID_LEN_SET = {92, 96}
PROGRESS_EVERY = 1000

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
"""
FOOTER = "  </Document>\n</kml>\n"

def pvt_placemark(ts, href, icon_heading, heading_true, heading_src,
                  lon, lat, alt, fix, itow, speed_m_s, speed_kmh, color,
                  pos_acc, alt_acc, speed_acc, speed_acc_kmh, head_acc,
                  point_block, gnssFixOK, flags_hex):
    """Render one NAV-PVT track Placemark.

    A single f-string (was a module-level `str.format` template): .format
    re-parses the template on every call, and this runs once per kept epoch —
    the f-string variant is compiled once and measurably faster.
    """
    return (
        "    <Placemark>\n"
        f"      <TimeStamp><when>{ts}</when></TimeStamp>\n"
        "      <Style>\n"
        "        <IconStyle>\n"
        f"          <color>{color}</color>\n"
        "          <colorMode>normal</colorMode>\n"
        "          <scale>0.5</scale>\n"
        f"          <heading>{icon_heading:.1f}</heading>\n"
        f"          <Icon><href>{href}</href></Icon>\n"
        "          <hotSpot x=\"0.5\" y=\"0.5\" xunits=\"fraction\" yunits=\"fraction\"/>\n"
        "        </IconStyle>\n"
        "      </Style>\n"
        "      <description><![CDATA[\n"
        f"        <b>UTC:</b> {ts}<br/>\n"
        f"        <b>iTOW:</b> {itow}<br/>\n"
        f"        <b>FixType:</b> {fix}<br/>\n"
        f"        <b>Fix flags:</b> {flags_hex} (gnssFixOK={gnssFixOK})<br/>\n"
        f"        <b>Heading:</b> {heading_true:.1f}° ({heading_src})<br/>\n"
        f"        <b>HeadAcc:</b> {head_acc:.2f}°<br/>\n"
        f"        <b>Speed:</b> {speed_m_s:.2f} m/s ({speed_kmh:.1f} km/h)<br/>\n"
        f"        <b>SpeedAcc:</b> {speed_acc:.2f} m/s ({speed_acc_kmh:.1f} km/h)<br/>\n"
        f"        <b>Lat:</b> {lat:.7f}<br/>\n"
        f"        <b>Lon:</b> {lon:.7f}<br/>\n"
        f"        <b>PosAcc2D:</b> {pos_acc:.2f} m<br/>\n"
        f"        <b>Alt:</b> {alt:.3f} m<br/>\n"
        f"        <b>AltAcc:</b> {alt_acc:.2f} m<br/>\n"
        "      ]]></description>\n"
        f"{point_block}"
        "    </Placemark>\n"
    )


# === AID-MAPM support (sky-blue arrows) ===
AID_CLASS = 0x0B
MAPM_ID   = 0x05
# Sky-blue (#87CEEB) in KML AABBGGRR order, so map-matching arrows stand out
# from the fix-type coloured NAV-PVT arrows.
MAPM_COLOR = "FFEBCE87"

MAPM_PLACEMARK_TEMPLATE = (
    "    <Placemark>\n"
    "      <name>AID-MAPM</name>\n"
    "      <Style>\n"
    "        <IconStyle>\n"
    "          <color>{color}</color>\n"
    "          <colorMode>normal</colorMode>\n"
    "          <scale>0.5</scale>\n"
    "          <heading>{icon_heading:.1f}</heading>\n"
    "          <Icon><href>https://maps.google.com/mapfiles/kml/shapes/arrow.png</href></Icon>\n"
    "          <hotSpot x=\"0.5\" y=\"0.5\" xunits=\"fraction\" yunits=\"fraction\"/>\n"
    "        </IconStyle>\n"
    "      </Style>\n"
    "      <description><![CDATA[\n"
    "        <b>UBX-AID-MAPM</b>{rel_note}<br/>\n"
    "        <b>iTOW:</b> {itow}<br/>\n"
    "{arrived_line}"
    "        <b>Heading:</b> {heading_true:.1f}°<br/>\n"
    "        <b>HeadAcc:</b> {head_acc:.2f}°<br/>\n"
    "{pos_detail}"
    "        <b>PosAcc2D:</b> {pos_acc:.2f} m<br/>\n"
    "        <b>Alt:</b> {alt:.3f} m<br/>\n"
    "        <b>AltAcc:</b> {alt_acc:.2f} m<br/>\n"
    "      ]]> </description>\n"
    "      <Point><coordinates>{lon:.7f},{lat:.7f},{alt:.3f}</coordinates></Point>\n"
    "    </Placemark>\n"
)

def parse_aid_mapm(payload: memoryview):
    """Parse UBX-AID-MAPM (length 28 bytes).

    Layout: itowMM(U4,0), flags(X2,4), headMM(U2,6), latMM(I4,8), lonMM(I4,12),
    altMM(I4,16), posHAccMM(U2,20), altAccMM(U2,22), headAccMM(U2,24), reserved0(2,26).

    flags bits: 0 latLonValid, 1 altValid, 2 headValid, 3 hmsl, 6 relativePos.
    When `relativePos` is set, lat/lon are **deltas** (same 1e-7 deg units) to the
    NAV-PVT fix at the **same iTOW** — the caller must add them to that fix's
    position.
    """
    if len(payload) < 28:
        return None
    itow  = int.from_bytes(payload[0:4], 'little', signed=False)
    flags = int.from_bytes(payload[4:6], 'little', signed=False)
    headMM = int.from_bytes(payload[6:8], 'little', signed=False)
    lat = int.from_bytes(payload[8:12], 'little', signed=True)
    lon = int.from_bytes(payload[12:16], 'little', signed=True)
    alt = int.from_bytes(payload[16:20], 'little', signed=True)
    pos_acc = int.from_bytes(payload[20:22], 'little', signed=False) * 1e-1
    alt_acc = int.from_bytes(payload[22:24], 'little', signed=False) * 1e-1
    head_acc = int.from_bytes(payload[24:26], 'little', signed=False) * 1e-2
    return {
        "iTOW": itow,
        "flags": flags,
        "latLonValid": bool(flags & 0x01),
        "altValid":    bool(flags & 0x02),
        "headValid":   bool(flags & 0x04),
        "hmsl":        bool(flags & 0x08),
        "relativePos": bool(flags & 0x40),   # bit 6
        "heading": (headMM * 1e-2) % 360.0,
        "lat": lat * 1e-7,   # absolute deg, or delta deg when relativePos
        "lon": lon * 1e-7,
        "alt": alt * 1e-3,
        "pos_acc": pos_acc,
        "alt_acc": alt_acc,
        "head_acc": head_acc,
    }

def mapm_placemark(rec, lat, lon, alt, relative=False, arrived_itow=None):
    """Render one sky-blue AID-MAPM Placemark.

    `lat`/`lon`/`alt` are the resolved *absolute* coordinates used for the map
    point. `arrived_itow` is the iTOW of the primary NAV-PVT that *preceded* this
    AID-MAPM in the stream — shown for *every* point (relative or absolute) as
    `arrived iTOW`, right under the message's own iTOW, so it is clear which
    NAV-PVT the message arrived after (its own itowMM is the map-matching
    solution time, which lags the arrival point). When `relative` is set,
    `rec["lat"]`/`rec["lon"]` hold the raw deltas that were added to the NAV-PVT
    fix at the *same iTOW* as the message (not the arrival-time fix); the popup
    then shows **both** the raw delta and the computed absolute position. For
    absolute points it shows a plain Lat/Lon.
    """
    heading_true = normalize_heading(rec["heading"])
    icon_heading = normalize_heading(heading_true + 180.0)
    # Nearest PVT iTOW (arrival reference) for both relative and absolute points.
    arrived_line = (f"        <b>arrived iTOW:</b> {arrived_itow}<br/>\n"
                    if arrived_itow is not None else "")
    if relative:
        pos_detail = (
            f"        <b>&#916;Lat (raw):</b> {rec['lat']:+.7f}&deg;<br/>\n"
            f"        <b>&#916;Lon (raw):</b> {rec['lon']:+.7f}&deg;<br/>\n"
            f"        <b>Lat (computed):</b> {lat:.7f}<br/>\n"
            f"        <b>Lon (computed):</b> {lon:.7f}<br/>\n"
        )
    else:
        pos_detail = (
            f"        <b>Lat:</b> {lat:.7f}<br/>\n"
            f"        <b>Lon:</b> {lon:.7f}<br/>\n"
        )
    return MAPM_PLACEMARK_TEMPLATE.format(
        color=MAPM_COLOR,
        rel_note=" (relativePos)" if relative else "",
        icon_heading=icon_heading,
        heading_true=heading_true,
        lon=lon, lat=lat, alt=alt,
        arrived_line=arrived_line,
        pos_detail=pos_detail,
        itow=rec["iTOW"],
        pos_acc=rec.get("pos_acc", 0.0),
        alt_acc=rec.get("alt_acc", 0.0),
        head_acc=rec.get("head_acc", 0.0),
    )

def build_kml_mapm_only(ubx_path: str, alt_abs: bool = False, verify_ck: bool = False):
    """Scan UBX and emit KML with ONLY AID-MAPM placemarks (sky-blue arrows).

    Absolute-position points only. `relativePos` points carry deltas to the
    latest navigation fix, which this NAV-less mode has no reference for, so
    they are skipped here (they are resolved in the main `build_kml` path).
    """
    buf = []
    total_frames = 0
    mapm_points = 0

    buf.append(HEADER)
    with open(ubx_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        n = len(mv)
        i = 0
        while True:
            # C-speed sync search (see build_kml for rationale).
            i = mm.find(UBX_SYNC, i)
            if i < 0 or i + 8 > n:
                break
            cls_ = mv[i+2]
            id_  = mv[i+3]
            length = mv[i+4] | (mv[i+5] << 8)
            frame_end = i + 6 + length + 2
            if frame_end > n:
                break

            payload = mv[i+6:i+6+length]
            ck_a, ck_b = mv[i+6+length], mv[i+6+length+1]
            total_frames += 1

            if verify_ck:
                ca, cb = fletcher_ck(mv[i+2:frame_end-2])
                if ca != ck_a or cb != ck_b:
                    i += 2
                    continue

            if cls_ == AID_CLASS and id_ == MAPM_ID:
                rec = parse_aid_mapm(payload)
                if rec and rec["latLonValid"] and not rec["relativePos"]:
                    buf.append(mapm_placemark(rec, rec["lat"], rec["lon"], rec["alt"]))
                    mapm_points += 1

            i = frame_end

    buf.append(FOOTER)
    print(f"{now_str()} | Finished doc.kml (frames scanned: {total_frames}, MAPM points: {mapm_points})")
    return ''.join(buf)

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def fletcher_ck(data: memoryview):
    a = 0
    b = 0
    for x in data:
        a = (a + x) & 0xFF
        b = (b + a) & 0xFF
    return a, b

# Precompiled unpackers for NAV-PVT (payload is always 92 or 96 bytes, so all
# fixed offsets below are guaranteed to exist). Precompiling avoids re-parsing
# the format string on every frame — parse_nav_pvt runs once per epoch and is
# one of the hottest paths in the scan.
_PVT_HEAD = struct.Struct("<IHBBBBBBIiBBBBiiiiII")  # 0..48: iTOW..hAcc,vAcc
_PVT_DYN  = struct.Struct("<iiII")                  # 60..76: gSpeed,headMot,sAcc,headAcc
_PVT_HEADVEH = struct.Struct("<i")                  # 84: headVeh

def parse_nav_pvt(payload: memoryview):
    L = len(payload)
    if L not in VALID_LEN_SET:
        return None
    iTOW, year, month, day, hour, minute, sec, valid, tAcc, nano, fixType, flags, \
        flags2, numSV, lon, lat, height, hMSL, hAcc_mm, vAcc_mm = \
        _PVT_HEAD.unpack_from(payload, 0)
    validDate = (valid & 0x01) != 0
    validTime = (valid & 0x02) != 0
    gSpeed, headMot, sAcc_mm, headAcc_raw = _PVT_DYN.unpack_from(payload, 60)
    headVeh = _PVT_HEADVEH.unpack_from(payload, 84)[0]
    speed_m_s = gSpeed / 1000.0  # mm/s -> m/s
    speed_kmh = speed_m_s * 3.6
    hAcc = hAcc_mm * 1e-3
    vAcc = vAcc_mm * 1e-3
    sAcc = sAcc_mm * 1e-3
    sAcc_kmh = sAcc * 3.6
    headAcc = headAcc_raw * 1e-5

    return {
        "iTOW": iTOW,
        "year": year, "month": month, "day": day,
        "hour": hour, "min": minute, "sec": sec,
        "nano": nano,   # signed ns offset of the second (for sub-second timestamps)
        "validDate": validDate, "validTime": validTime,
        "fixType": fixType,
        "flags": flags,
        "numSV": numSV,
        "lat": lat * 1e-7,
        "lon": lon * 1e-7,
        "hMSL": hMSL / 1000.0,
        "speed_m_s": abs(speed_m_s),
        "speed_kmh": abs(speed_kmh),
        "headMot": (headMot * 1e-5),
        "headVeh": (headVeh * 1e-5) if headVeh is not None else None,
        "pos_acc": hAcc, "alt_acc": vAcc, "speed_acc": sAcc, "speed_acc_kmh": sAcc_kmh, "head_acc": headAcc, 
    }

def parse_nav_sat(payload: memoryview):
    """
    Parse UBX-NAV-SAT (0x01 0x35)
    Header: 8 bytes [iTOW(4), version(1), numSvs(1), reserved(2)]
    Block: 12 bytes per SV
    """
    if len(payload) < 8:
        return None
    iTOW = int.from_bytes(payload[0:4], 'little', signed=False)
    numSvs = payload[5]

    # u-blox receivers track at most ~60 SVs; larger numSvs combined with a
    # matching length almost certainly means the sync pattern matched garbage.
    if numSvs > 128:
        return None
    if len(payload) < 8 + 12 * numSvs:
        return None

    # Valid CN0 range per u-blox spec. Anything outside this is treated as a
    # mis-parsed block (e.g. from a false sync match with --ck disabled).
    CNO_MAX = 63

    # Minimum qualityInd to count a block as a real tracked signal.
    # Per u-blox NAV-SAT spec, flags bits 2..0 (qualityInd):
    #   0 = no signal, 1 = searching, 2 = acquired, 3 = detected but unusable,
    #   4 = code locked, 5 = code+carrier locked,
    #   6..7 = code+carrier locked + nav data.
    # Blocks with qualityInd < 4 are not real tracked signals — their `cno`
    # field can carry stale/placeholder values (we've seen bogus rows with
    # cno=60 but qualityInd=0, which were inflating the top-5 average).
    QI_MIN = 4

    cnos = []
    offset = 8
    for _ in range(numSvs):
        # Block structure: gnssId(1), svId(1), cno(1), elev(1), azim(2), prRes(2), flags(4)
        cno = payload[offset + 2]  # dBHz
        flags = int.from_bytes(payload[offset + 8:offset + 12], 'little', signed=False)
        quality_ind = flags & 0x07
        if quality_ind >= QI_MIN and 0 < cno <= CNO_MAX:
            cnos.append(cno)
        offset += 12

    return {"iTOW": iTOW, "cnos": cnos}

def parse_sec_sig(payload: memoryview):
    """
    Parse UBX-SEC-SIG (0x27 0x09) version 0x02.
    Layout:
      0: version (U1) — expected 0x02
      1: sigSecFlags (X1)
           bit 0    : jamDetEnabled
           bits 2..1: jamState   (0=Unknown, 1=OK, 2=Warning)
           bit 3    : spfDetEnabled
           bits 5..4: spfState   (0=Unknown, 1=OK, 2=Indicated, 3=Affirmed)
      2: reserved0 (U1)
      3: jamNumCentFreqs (U1)
      4..: jamStateCentFreq (X4) repeated jamNumCentFreqs times
             bits 23..0: centFreq (kHz)
             bit 24    : jammed
    """
    L = len(payload)
    if L < 4:
        return None
    version = payload[0]
    if version != 0x02:
        return None
    flags = payload[1]
    jam_det_enabled = (flags & 0x01) != 0
    jam_state       = (flags >> 1) & 0x03
    spf_det_enabled = (flags & 0x08) != 0
    spf_state       = (flags >> 4) & 0x03
    num_cf = payload[3]
    if L < 4 + 4 * num_cf:
        return None
    freqs = []
    for n in range(num_cf):
        word = int.from_bytes(payload[4 + 4*n : 8 + 4*n], 'little', signed=False)
        cent_khz = word & 0x00FFFFFF
        jammed   = (word >> 24) & 0x01
        freqs.append({
            "freq_mhz": round(cent_khz / 1000.0, 3),
            "jammed": bool(jammed),
        })
    return {
        "version": version,
        "jamDetEnabled": jam_det_enabled,
        "jamState": jam_state,
        "spfDetEnabled": spf_det_enabled,
        "spfState": spf_state,
        "centFreqs": freqs,
    }


def normalize_heading(deg: float) -> float:
    h = math.fmod(deg if deg is not None else 0.0, 360.0)
    if h < 0:
        h += 360.0
    return h

def pvt_utc_str(rec) -> str:
    """UTC timestamp for a NAV-PVT record with two decimal places on the
    seconds (uses the signed `nano` offset). Format: YYYY-MM-DDTHH:MM:SS.ccZ.

    Fast path: for the overwhelmingly common case (0 <= nano, no roll-over to
    the next second) the string is built with pure integer arithmetic — the
    datetime+strftime construction it replaces was one of the hottest spots in
    the per-epoch profile. Negative nano (borrow into the previous second) and
    near-1s values fall back to datetime, which handles the carry across
    minute/day boundaries.
    """
    nano = rec.get('nano', 0)
    if 0 <= nano < 999_999_500:
        # Centiseconds, matching timedelta's round-to-microsecond behaviour.
        cs = (nano + 500) // 10_000_000
        return (f"{rec['year']:04d}-{rec['month']:02d}-{rec['day']:02d}"
                f"T{rec['hour']:02d}:{rec['min']:02d}:{rec['sec']:02d}.{cs:02d}Z")
    try:
        t = datetime(rec['year'], rec['month'], rec['day'],
                     rec['hour'], rec['min'], rec['sec']) \
            + timedelta(seconds=nano * 1e-9)
        return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{t.microsecond // 10000:02d}Z"
    except ValueError:
        return (f"{rec['year']:04d}-{rec['month']:02d}-{rec['day']:02d}"
                f"T{rec['hour']:02d}:{rec['min']:02d}:{rec['sec']:02d}.00Z")

def build_kml(ubx_path: str, hz: int = None, use_nav2: bool = False,
              alt_abs: bool = False, verify_ck: bool = False):
    buf = []
    total_msgs = 0
    valid_msgs = 0
    kept = 0

    # [수정] 누락 감지를 위한 변수 초기화
    missing_epochs = 0
    last_itow = None
    # Hz 옵션이 있으면 예상 간격(ms)을 미리 설정, 없으면 자동 추정
    expected_interval = (1000 // hz) if hz else None

    # Graph data structures
    graph_data = {
        "labels": [],
        "acc2d": [],
        "acc3d": [],
        "fix_type": [],
        "speed": [],        # km/h
        "altitude": [],     # m (hMSL)
        "num_sv": [],       # satellite count
        "lat": [],          # deg, per kept epoch (for map overlays)
        "lon": [],          # deg, per kept epoch (for map overlays)
        "cno_labels": [],
        "cno_top_avg": [],
        # Accuracy series of the *other* PVT class (NAV2-PVT when the primary
        # is NAV-PVT, and vice versa). Populated only when such frames exist,
        # so the report can show both accuracy charts side by side.
        "primary_pvt": "NAV2-PVT" if use_nav2 else "NAV-PVT",
        "alt_pvt":     "NAV-PVT" if use_nav2 else "NAV2-PVT",
        "alt_labels": [],
        "alt_acc2d": [],
        "alt_acc3d": [],
        # UBX-SEC-SIG (jamming/spoofing). Populated only if SEC-SIG frames are found.
        "sec_labels": [],     # iTOW per sample, aligned with the last NAV-PVT iTOW
        "jam_state": [],      # 0=Unknown, 1=OK, 2=Warning
        "spf_state": [],      # 0=Unknown, 1=OK, 2=Indicated, 3=Affirmed
        "jam_det_enabled": [],
        "spf_det_enabled": [],
        "sec_freqs": [],      # list per sample: [{freq_mhz, jammed}, ...]
        # UBX-INF messages: [{itow, utc, level, text}, ...] in file order.
        # itow/utc are the most recent PVT time reference (None before the
        # first fix, since INF frames carry no time of their own).
        "inf_messages": [],
        # [추가] 통계 정보 저장용
        "stats": {
            "epoch_total": 0,
            "epoch_missing": 0
        }
    }
    
    kept_itows = set()
    itow_to_cno = {}
    itow_to_secsig = {}
    last_pvt_itow = None  # most recent PVT iTOW seen, used to timestamp SEC-SIG frames
    # Primary NAV-PVT fix positions keyed by iTOW: AID-MAPM relativePos deltas
    # are added to the fix at the *same iTOW* as the MAPM message. Frames are
    # buffered and resolved after the scan so a matching fix that appears later
    # in the byte stream is still found. Each buffered record also carries
    # `arrived_itow` — the primary PVT that preceded it in stream order —
    # captured at scan time, since it reflects arrival, not solution time.
    itow_to_fix = {}       # iTOW -> (lat, lon)
    mapm_records = []       # (rec, arrived_itow) tuples deferred for resolution
    mapm_points = 0
    # Time reference for INF messages: updated for ANY valid-time PVT epoch
    # (even no-fix), so INF frames get the closest available timestamp. The
    # UTC string is rendered lazily — only when an INF frame actually arrives —
    # since formatting it eagerly for every epoch was pure waste on the
    # (typical) logs that carry few or no INF messages.
    inf_ref_itow = None
    inf_ref_rec  = None   # PVT record backing the lazy UTC string
    inf_ref_utc  = None   # rendered-on-demand cache for inf_ref_rec

    target_class = NAV2_CLASS if use_nav2 else NAV_CLASS
    alt_class    = NAV_CLASS if use_nav2 else NAV2_CLASS  # same payload layout
    period = 1000 // hz if hz else None

    buf.append(HEADER)
    with open(ubx_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        n = len(mv)
        i = 0
        while True:
            # Locate the next sync pattern with mmap.find (C speed) instead of
            # advancing one byte at a time in Python — non-UBX content between
            # frames (interleaved NMEA text, headers, corrupt runs) is skipped
            # orders of magnitude faster.
            i = mm.find(UBX_SYNC, i)
            if i < 0 or i + 8 > n:
                break
            cls_ = mv[i+2]
            id_  = mv[i+3]
            length = mv[i+4] | (mv[i+5] << 8)
            frame_end = i + 6 + length + 2
            if frame_end > n: break

            payload = mv[i+6:i+6+length]
            ck_a, ck_b = mv[i+6+length], mv[i+6+length+1]

            if verify_ck:
                ca, cb = fletcher_ck(mv[i+2:frame_end-2])
                if ca != ck_a or cb != ck_b:
                    i += 2
                    continue

            # 1. NAV-PVT 처리
            if cls_ == target_class and id_ == PVT_ID:
                total_msgs += 1
                rec = parse_nav_pvt(payload)
                # Update the INF time reference for any epoch with a valid
                # date/time, regardless of fix (so INF messages during a no-fix
                # period still get the nearest timestamp).
                if rec and rec["validDate"] and rec["validTime"]:
                    inf_ref_itow = rec["iTOW"]
                    inf_ref_rec  = rec
                    inf_ref_utc  = None  # re-rendered on demand for the new epoch
                # fixType: 1=DR only, 2=2D, 3=3D, 4=GNSS+DR. 0 (no fix) and
                # 5 (time only) carry no usable position and are skipped.
                if rec and rec["validDate"] and rec["validTime"] and rec["fixType"] in (1, 2, 3, 4):
                    last_pvt_itow = rec["iTOW"]
                    # Anchor positions for AID-MAPM relativePos deltas, keyed by
                    # iTOW (recorded before Hz filtering so every fix is usable).
                    itow_to_fix[rec["iTOW"]] = (rec["lat"], rec["lon"])
                    # === [추가됨] Missing Epoch 계산 로직 ===
                    curr_itow = rec["iTOW"]
                    if last_itow is not None:
                        diff = curr_itow - last_itow
                        
                        # 주(Week) 변경으로 인한 iTOW 리셋 보정 (매우 드문 케이스)
                        if diff < -500000000: 
                            diff += 604800000
                        
                        if diff > 0:
                            # 예상 간격이 아직 없으면 첫 간격으로 추정 (단, 너무 작은 노이즈 제외)
                            if expected_interval is None and diff > 50:
                                # 일반적인 GNSS 주기에 맞춰 근사값 설정
                                if 80 <= diff <= 120: expected_interval = 100        # 10Hz
                                elif 180 <= diff <= 220: expected_interval = 200     # 5Hz
                                elif 900 <= diff <= 1100: expected_interval = 1000   # 1Hz
                                else: expected_interval = diff
                            
                            # 간격이 예상보다 1.5배 이상 벌어지면 누락으로 간주
                            if expected_interval and diff > (expected_interval * 1.5):
                                skipped = round(diff / expected_interval) - 1
                                if skipped > 0:
                                    missing_epochs += int(skipped)
                    
                    last_itow = curr_itow
                    # ========================================

                    if hz and (rec["iTOW"] % period) != 0:
                        i = frame_end
                        continue
                    
                    kept_itows.add(rec["iTOW"])

                    # Accuracy Graph Data
                    h_acc = rec.get("pos_acc", 0.0)
                    v_acc = rec.get("alt_acc", 0.0)
                    d3_acc = math.sqrt(h_acc**2 + v_acc**2)
                    
                    graph_data["labels"].append(rec["iTOW"])
                    graph_data["acc2d"].append(round(h_acc, 3))
                    graph_data["acc3d"].append(round(d3_acc, 3))
                    graph_data["fix_type"].append(rec["fixType"])
                    graph_data["speed"].append(round(rec["speed_kmh"], 2))
                    graph_data["altitude"].append(round(rec["hMSL"], 2))
                    graph_data["num_sv"].append(int(rec.get("numSV", 0)))
                    graph_data["lat"].append(round(rec["lat"], 7))
                    graph_data["lon"].append(round(rec["lon"], 7))

                    # KML Placemark 생성 (기존 유지)
                    gnssFixOK = (rec.get("flags", 0) & 0x01) != 0
                    if not gnssFixOK:
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FFFF0000"
                    elif rec["fixType"] == 2:
                        # 2D fix: yellow, heading from motion (vehicle heading
                        # is not meaningful without a 3D solution).
                        heading_raw = rec["headMot"]; heading_src = "headMot"; color = "FF00FFFF"
                    elif rec["fixType"] == 3:
                        heading_raw = rec["headMot"]; heading_src = "headMot"; color = "FF00C800"
                    elif rec["fixType"] == 4:
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FF00A5FF"
                    else:
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FF800080"

                    heading_true = normalize_heading(heading_raw)
                    icon_heading = normalize_heading(heading_true + 180.0)

                    # Timestamp with two decimal places on the seconds (uses the
                    # NAV-PVT `nano` field). Shown in the popup description and
                    # used as the KML <when> for playback ordering.
                    ts = pvt_utc_str(rec)
                    href = "https://maps.google.com/mapfiles/kml/shapes/arrow.png"
                    
                    if alt_abs:
                        point_block = f"      <Point>\n        <extrude>1</extrude>\n        <altitudeMode>absolute</altitudeMode>\n        <coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['hMSL']:.3f}</coordinates>\n      </Point>\n"
                    else:
                        point_block = f"      <Point><coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['hMSL']:.3f}</coordinates></Point>\n"

                    buf.append(pvt_placemark(
                        ts=ts, href=href,
                        icon_heading=icon_heading, heading_true=heading_true, heading_src=heading_src,
                        lon=rec["lon"], lat=rec["lat"], alt=rec["hMSL"],
                        fix=rec["fixType"], itow=rec["iTOW"],
                        speed_m_s=rec["speed_m_s"], speed_kmh=rec["speed_kmh"], color=color,
                        pos_acc=rec.get("pos_acc", 0.0), alt_acc=rec.get("alt_acc", 0.0), speed_acc=rec.get("speed_acc", 0.0), speed_acc_kmh=rec.get("speed_acc_kmh", 0.0), head_acc=rec.get("head_acc", 0.0),
                        point_block=point_block,
                        gnssFixOK=1 if gnssFixOK else 0,
                        flags_hex=f"0x{rec.get('flags',0):02X}",
                    ))
                    valid_msgs += 1
                    if hz: kept += 1

            # 1b. Secondary PVT class (NAV2-PVT alongside NAV-PVT, or vice
            # versa). Same payload layout; only the accuracy series is kept
            # so the report can chart both message streams.
            elif cls_ == alt_class and id_ == PVT_ID:
                rec = parse_nav_pvt(payload)
                if rec and rec["validDate"] and rec["validTime"] and rec["fixType"] in (1, 2, 3, 4):
                    if not (hz and (rec["iTOW"] % period) != 0):
                        a_h = rec.get("pos_acc", 0.0)
                        a_v = rec.get("alt_acc", 0.0)
                        graph_data["alt_labels"].append(rec["iTOW"])
                        graph_data["alt_acc2d"].append(round(a_h, 3))
                        graph_data["alt_acc3d"].append(round(math.sqrt(a_h**2 + a_v**2), 3))

            # 2. NAV-SAT 처리
            elif cls_ == NAV_CLASS and id_ == SAT_ID:
                sat_rec = parse_nav_sat(payload)
                if sat_rec:
                    itow_to_cno[sat_rec["iTOW"]] = sat_rec["cnos"]

            # 3. SEC-SIG (jamming / spoofing) 처리
            # SEC-SIG has no iTOW field; associate with the most recent NAV-PVT iTOW.
            elif cls_ == SEC_CLASS and id_ == SECSIG_ID:
                sec_rec = parse_sec_sig(payload)
                if sec_rec and last_pvt_itow is not None:
                    itow_to_secsig[last_pvt_itow] = sec_rec

            # 4. UBX-INF (ASCII text messages). Collected in file order with the
            # nearest PVT time reference so the report can show what the receiver
            # reported and when.
            elif cls_ == INF_CLASS and id_ in INF_LEVELS:
                if len(graph_data["inf_messages"]) < INF_MAX:
                    text = bytes(payload).decode("ascii", "ignore").strip()
                    # Guard against false sync matches (verify_ck off by default):
                    # a real INF payload is printable text.
                    printable = sum(1 for c in text if 32 <= ord(c) < 127)
                    if text and printable >= 0.8 * len(text):
                        if inf_ref_utc is None and inf_ref_rec is not None:
                            inf_ref_utc = pvt_utc_str(inf_ref_rec)
                        graph_data["inf_messages"].append({
                            "itow":  inf_ref_itow,
                            "utc":   inf_ref_utc,
                            "level": INF_LEVELS[id_],
                            "text":  text,
                        })

            # 5. AID-MAPM (map-matching points). Buffered and resolved after the
            # scan so `relativePos` deltas can be anchored to the NAV-PVT fix at
            # the *same iTOW* even when that fix appears later in the stream.
            # `arrived iTOW` — the primary PVT that preceded this frame in
            # stream order (its own itowMM is the map-matching *solution* time,
            # which lags the actual arrival point) — must be captured NOW.
            elif cls_ == AID_CLASS and id_ == MAPM_ID:
                rec = parse_aid_mapm(payload)
                if rec and rec["latLonValid"]:
                    mapm_records.append((rec, inf_ref_itow))

            i = frame_end

    # Merge PVT time and NAV-SAT CN0. Only the Top-5 line is rendered on the
    # report, so we no longer emit a per-satellite scatter dataset (saves a
    # large amount of JSON and cuts Chart.js render cost).
    sorted_itows = sorted(list(kept_itows))

    for itow in sorted_itows:
        if itow in itow_to_cno:
            cnos = itow_to_cno[itow]

            # Empty cnos → 0.0 so the line stays continuous instead of gapping.
            avg_cno = 0.0
            if cnos:
                sorted_cnos = sorted(cnos, reverse=True)
                top_k = sorted_cnos[:5]
                # Divide by the actual top-k count so epochs with fewer than
                # 5 tracked signals are not artificially pulled down.
                avg_cno = sum(top_k) / len(top_k)

            graph_data["cno_labels"].append(itow)
            graph_data["cno_top_avg"].append(round(avg_cno, 1))

    # Merge SEC-SIG samples, aligned to kept PVT epochs.
    if itow_to_secsig:
        for itow in sorted_itows:
            rec = itow_to_secsig.get(itow)
            if rec is None:
                continue
            graph_data["sec_labels"].append(itow)
            graph_data["jam_state"].append(rec["jamState"])
            graph_data["spf_state"].append(rec["spfState"])
            graph_data["jam_det_enabled"].append(1 if rec["jamDetEnabled"] else 0)
            graph_data["spf_det_enabled"].append(1 if rec["spfDetEnabled"] else 0)
            graph_data["sec_freqs"].append(rec["centFreqs"])

    # Resolve buffered AID-MAPM points now that every NAV-PVT fix is known.
    # Resolve buffered AID-MAPM points now that every NAV-PVT fix is known.
    # relativePos deltas are added to the fix at the same iTOW as the MAPM
    # message (nearest iTOW as a fallback so a point is still placed when that
    # exact epoch is missing); a relativePos point with no fix at all is
    # skipped. The popup's `arrived iTOW` uses the stream-order value captured
    # during the scan, independent of the delta anchor.
    fix_itows = sorted(itow_to_fix.keys())
    for rec, arrived_itow in mapm_records:
        if rec["relativePos"]:
            if not fix_itows:
                continue  # no navigation fix to anchor the delta against
            j = bisect.bisect_left(fix_itows, rec["iTOW"])
            cand = []
            if j < len(fix_itows):
                cand.append(fix_itows[j])
            if j > 0:
                cand.append(fix_itows[j - 1])
            ref_itow = min(cand, key=lambda k: abs(k - rec["iTOW"]))
            ref_lat, ref_lon = itow_to_fix[ref_itow]
            m_lat = ref_lat + rec["lat"]
            m_lon = ref_lon + rec["lon"]
            buf.append(mapm_placemark(rec, m_lat, m_lon, rec["alt"],
                                      relative=True, arrived_itow=arrived_itow))
        else:
            buf.append(mapm_placemark(rec, rec["lat"], rec["lon"], rec["alt"],
                                      arrived_itow=arrived_itow))
        mapm_points += 1

    # [수정] 통계 정보 최종 저장
    graph_data["stats"]["epoch_total"] = valid_msgs
    graph_data["stats"]["epoch_missing"] = missing_epochs

    buf.append(FOOTER)
    print(f"{now_str()} | Finished doc.kml (Total: {valid_msgs}, Missing: {missing_epochs}, kept: {kept if hz else 'All'}, MAPM: {mapm_points})")

    kml_text = ''.join(buf)
    return kml_text, graph_data

def write_kmz(kml_text: str, kmz_path: str) -> None:
    os.makedirs(os.path.dirname(kmz_path) or '.', exist_ok=True)
    with zipfile.ZipFile(kmz_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', kml_text.encode('utf-8'))
    print(f"{now_str()} | KMZ saved -> {kmz_path}")

def run(ubx_path: str, hz: int = None, use_nav2: bool = False,
        alt_abs: bool = False, verify_ck: bool = False, mapm: bool = False):
    base, _ = os.path.splitext(ubx_path)
    if mapm:
        kmz_path = base + "_mapm.kmz"
        print(f"{now_str()} | MAPM-only mode: writing {kmz_path}")
        kml_text = build_kml_mapm_only(ubx_path, alt_abs=alt_abs, verify_ck=verify_ck)
        write_kmz(kml_text, kmz_path)
        return

    suffix = "_nav2" if use_nav2 else "_nav"
    if hz:
        suffix = f"_{hz}hz" + suffix
    if alt_abs:
        suffix = suffix + "_abs"
    suffix = suffix + ("_ck" if verify_ck else "_nock")
    kmz_path = base + suffix + ".kmz"

    # JSON 경로: KMZ와 같은 base (API 엔드포인트에서 .kmz → _graph.json 치환)
    json_path = kmz_path.replace(".kmz", "_graph.json")

    print(f"{now_str()} | KMZ mode: {kmz_path}")

    # build_kml에서 graph_data도 함께 받아옴 (HTML 관련 인자 제거)
    kml_text, graph_data = build_kml(
        ubx_path, hz=hz, use_nav2=use_nav2,
        alt_abs=alt_abs, verify_ck=verify_ck
    )

    write_kmz(kml_text, kmz_path)

    # JSON 파일 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f)
    print(f"{now_str()} | Graph JSON saved -> {json_path}")

def main():
    ap = argparse.ArgumentParser(description="UBX NAV/NAV2-PVT -> KMZ + AID-MAPM (--mapm)")
    ap.add_argument("ubx", help="Input UBX file path")
    ap.add_argument("--hz", type=int, choices=[1, 2, 5, 10],
                    help="Downsample to given rate (Hz). If omitted, keep all frames.")
    ap.add_argument("--nav2", action="store_true",
                    help="Use NAV2-PVT instead of NAV-PVT")
    ap.add_argument("--alt-abs", action="store_true",
                    help="Use altitudeMode=absolute with extrude=1 in <Point>")
    ap.add_argument("--ck", action="store_true",
                    help="Enable UBX checksum verification (default: off)")
    ap.add_argument("--mapm", action="store_true",
                    help="Parse only UBX-AID-MAPM and overlay white arrows in KMZ; ignores NAV/NAV2")
    args = ap.parse_args()
    
    run(args.ubx, hz=args.hz, use_nav2=args.nav2,
        alt_abs=args.alt_abs, verify_ck=args.ck, mapm=args.mapm)

if __name__ == "__main__":
    main()