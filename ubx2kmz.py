#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UBX (NAV-PVT / NAV2-PVT) -> KMZ + HTML report (with gnssFixOK)
- Default: NAV-PVT only (class=0x01,id=0x07)
- Option: --nav2          => NAV2-PVT only (class=0x29,id=0x07)
- Option: --hz {1,2,5,10} => keep only frames aligned to the given Hz (iTOW ms)
- Option: --alt-abs       => set altitudeMode=absolute with extrude=1 in <Point>
- Option: --ck            => enable UBX checksum verification (default: off)
- Option: --html          => write an offline HTML report with fixType/numSV/gSpeed/gnssFixOK
"""

import argparse
import os
import time
import mmap
import struct
import math
import zipfile
import json
from collections import defaultdict

UBX_SYNC1 = 0xB5
UBX_SYNC2 = 0x62
NAV_CLASS  = 0x01
NAV2_CLASS = 0x29
PVT_ID     = 0x07

VALID_LEN_SET = {92, 96}
PROGRESS_EVERY = 1000

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
"""
FOOTER = "  </Document>\n</kml>\n"

PLACEMARK_TEMPLATE = (
    "    <Placemark>\n"
    "      <TimeStamp><when>{ts}</when></TimeStamp>\n"
    "      <Style>\n"
    "        <IconStyle>\n"
    "          <color>{color}</color>\n"
    "          <colorMode>normal</colorMode>\n"
    "          <scale>0.5</scale>\n"
    "          <heading>{icon_heading:.1f}</heading>\n"
    "          <Icon><href>{href}</href></Icon>\n"
    "          <hotSpot x=\"0.5\" y=\"0.5\" xunits=\"fraction\" yunits=\"fraction\"/>\n"
    "        </IconStyle>\n"
    "      </Style>\n"
    "      <description><![CDATA[\n"
    "        <b>UTC:</b> {ts}<br/>\n"
    "        <b>iTOW:</b> {itow}<br/>\n"
    "        <b>FixType:</b> {fix}<br/>\n"
    "        <b>Fix flags:</b> {flags_hex} (gnssFixOK={gnssFixOK})<br/>\n"
    "        <b>Heading:</b> {heading_true:.1f}° ({heading_src})<br/>\n"
    "        <b>HeadAcc:</b> {head_acc:.2f}°<br/>\n"
    "        <b>Speed:</b> {speed_m_s:.2f} m/s ({speed_kmh:.1f} km/h)<br/>\n"
    "        <b>SpeedAcc:</b> {speed_acc:.2f} m/s ({speed_acc_kmh:.1f} km/h)<br/>\n"
    "        <b>Lat:</b> {lat:.7f}<br/>\n"
    "        <b>Lon:</b> {lon:.7f}<br/>\n"
    "        <b>PosAcc2D:</b> {pos_acc:.2f} m<br/>\n"
    "        <b>Alt:</b> {alt:.3f} m<br/>\n"
    "        <b>AltAcc:</b> {alt_acc:.2f} m<br/>\n"
    "      ]]></description>\n"
    "{point_block}"
    "    </Placemark>\n"
)


# === AID-MAPM support (white arrows) ===
AID_CLASS = 0x0B
MAPM_ID   = 0x05

MAPM_PLACEMARK_TEMPLATE = (
    "    <Placemark>\n"
    "      <Style>\n"
    "        <IconStyle>\n"
    "          <color>FFFFFFFF</color>\n"
    "          <colorMode>normal</colorMode>\n"
    "          <scale>0.5</scale>\n"
    "          <heading>{icon_heading:.1f}</heading>\n"
    "          <Icon><href>https://maps.google.com/mapfiles/kml/shapes/arrow.png</href></Icon>\n"
    "          <hotSpot x=\"0.5\" y=\"0.5\" xunits=\"fraction\" yunits=\"fraction\"/>\n"
    "        </IconStyle>\n"
    "      </Style>\n"
    "      <description><![CDATA[\n"
    "        <b>UBX-AID-MAPM</b><br/>\n"
    "        <b>iTOW:</b> {itow}<br/>\n"
    "        <b>Heading:</b> {heading_true:.1f}°<br/>\n"
    "        <b>HeadAcc:</b> {head_acc:.2f}°<br/>\n"
    "        <b>Lat:</b> {lat:.7f}<br/>\n"
    "        <b>Lon:</b> {lon:.7f}<br/>\n"
    "        <b>PosAcc2D:</b> {pos_acc:.2f} m<br/>\n"
    "        <b>Alt:</b> {alt:.3f} m<br/>\n"
    "        <b>AltAcc:</b> {alt_acc:.2f} m<br/>\n"
    "      ]]> </description>\n"
    "      <Point><coordinates>{lon:.7f},{lat:.7f},{alt:.3f}</coordinates></Point>\n"
    "    </Placemark>\n"
)

def parse_aid_mapm(payload: memoryview):
    """Parse UBX-AID-MAPM (length typically 28 bytes).
    0  U4 itowMM
    4  U2 flags (ignored)
    6  U2 headMM  (0.01 deg)
    8  I4 latMM   (1e-7 deg)
    12 I4 lonMM   (1e-7 deg)
    16 I4 altMM   (1e-3 m)
    20 U2 posHAccMM (ignored)
    22 U2 altAccMM  (ignored)
    24 U2 headAccMM (ignored)
    26 U1[2] reserved
    """
    if len(payload) < 28:
        return None
    itow = int.from_bytes(payload[0:4], 'little', signed=False)
    headMM = int.from_bytes(payload[6:8], 'little', signed=False)
    lat = int.from_bytes(payload[8:12], 'little', signed=True)
    lon = int.from_bytes(payload[12:16], 'little', signed=True)
    alt = int.from_bytes(payload[16:20], 'little', signed=True)
    # Accuracy fields for AID-MAPM
    pos_acc = int.from_bytes(payload[20:22], 'little', signed=False) * 1e-1
    alt_acc = int.from_bytes(payload[22:24], 'little', signed=False) * 1e-1
    head_acc = int.from_bytes(payload[24:26], 'little', signed=False) * 1e-2
    return {
        "iTOW": itow,
        "heading": (headMM * 1e-2) % 360.0,
        "lat": lat * 1e-7,
        "lon": lon * 1e-7,
        "alt": alt * 1e-3,
        "pos_acc": pos_acc,
        "alt_acc": alt_acc,
        "head_acc": head_acc,
    }

def build_kml_mapm_only(ubx_path: str, alt_abs: bool = False, verify_ck: bool = False, progress_cb=None):
    """Scan UBX and emit KML with ONLY AID-MAPM placemarks (white arrows)."""
    buf = []
    total_frames = 0
    mapm_points = 0

    buf.append(HEADER)
    with open(ubx_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        n = len(mv)
        i = 0
        last_pct = 0
        while i + 8 <= n:
            if progress_cb:
                pct = int(i * 100 / n)
                if pct > last_pct:
                    progress_cb(pct)
                    last_pct = pct

            if mv[i] != UBX_SYNC1 or mv[i+1] != UBX_SYNC2:
                i += 1
                continue
            if i + 6 > n:
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
                if rec:
                    heading_true = normalize_heading(rec["heading"])
                    icon_heading = normalize_heading(heading_true + 180.0)
                    if alt_abs:
                        point_block = (
                            f"      <Point>\n"
                            f"        <extrude>1</extrude>\n"
                            f"        <altitudeMode>absolute</altitudeMode>\n"
                            f"        <coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['alt']:.3f}</coordinates>\n"
                            f"      </Point>\n"
                        )
                    else:
                        point_block = (
                            f"      <Point><coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['alt']:.3f}</coordinates></Point>\n"
                        )
                    buf.append(MAPM_PLACEMARK_TEMPLATE.format(
                        icon_heading=icon_heading,
                        heading_true=heading_true,
                        lon=rec["lon"], lat=rec["lat"], alt=rec["alt"],
                        itow=rec["iTOW"],
                        pos_acc=rec.get("pos_acc", 0.0), alt_acc=rec.get("alt_acc", 0.0), head_acc=rec.get("head_acc", 0.0),
                        point_block=point_block
                    ))
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

def parse_nav_pvt(payload: memoryview):
    L = len(payload)
    if L not in VALID_LEN_SET:
        return None
    head = struct.unpack_from("<I H B B B B B B I i B B B B i i i i", payload, 0)
    iTOW, year, month, day, hour, minute, sec, valid, tAcc, nano, fixType, flags, flags2, numSV, \
        lon, lat, height, hMSL = head
    validDate = (valid & 0x01) != 0
    validTime = (valid & 0x02) != 0
    gSpeed = 0
    headMot = 0
    try:
        gSpeed, headMot = struct.unpack_from("<i i", payload, 60)
    except struct.error:
        pass
    headVeh = None
    for off in (84, 88):
        if off + 4 <= L:
            try:
                val = struct.unpack_from("<i", payload, off)[0]
                headVeh = val
                break
            except struct.error:
                pass
    speed_m_s = gSpeed / 1000.0  # mm/s -> m/s
    speed_kmh = speed_m_s * 3.6
    # Accuracy fields
    try:
        hAcc = int.from_bytes(payload[40:44], 'little', signed=False) * 1e-3
        vAcc = int.from_bytes(payload[44:48], 'little', signed=False) * 1e-3
        sAcc = int.from_bytes(payload[68:72], 'little', signed=False) * 1e-3
        sAcc_kmh = sAcc * 3.6
        headAcc = int.from_bytes(payload[72:76], 'little', signed=False) * 1e-5
    except Exception:
        hAcc = vAcc = sAcc = headAcc = 0.0

    return {
        "iTOW": iTOW,
        "year": year, "month": month, "day": day,
        "hour": hour, "min": minute, "sec": sec,
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

def normalize_heading(deg: float) -> float:
    h = math.fmod(deg if deg is not None else 0.0, 360.0)
    if h < 0:
        h += 360.0
    return h

def build_kml(ubx_path: str, hz: int = None, use_nav2: bool = False,
              alt_abs: bool = False, verify_ck: bool = False,
              want_html: bool = False, progress_cb=None):
    buf = []
    total_msgs = 0
    valid_msgs = 0
    kept = 0

    fix_counts = defaultdict(int)
    ok_counts  = defaultdict(int)
    fix_series = []     # [{t: iTOW_s, fixType: int}]
    numsv_series = []   # [{t: iTOW_s, v: int}]
    gspeed_series = []  # [{t: iTOW_s, v: float}]
    ok_series   = []    # [{t: iTOW_s, ok: 0|1}]

    target_class = NAV2_CLASS if use_nav2 else NAV_CLASS
    period = 1000 // hz if hz else None  # iTOW(ms) 간격

    buf.append(HEADER)
    with open(ubx_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        n = len(mv)
        i = 0
        last_pct = 0
        while i + 8 < n:
            if progress_cb:
                pct = int(i * 100 / n)
                if pct > last_pct:
                    progress_cb(pct)
                    last_pct = pct

            if mv[i] != UBX_SYNC1 or mv[i+1] != UBX_SYNC2:
                i += 1
                continue
            if i + 6 >= n:
                break
            cls_ = mv[i+2]
            id_  = mv[i+3]
            length = mv[i+4] | (mv[i+5] << 8)
            frame_end = i + 6 + length + 2
            if frame_end > n:
                break

            payload = mv[i+6:i+6+length]
            ck_a, ck_b = mv[i+6+length], mv[i+6+length+1]

            if verify_ck:
                ca, cb = fletcher_ck(mv[i+2:frame_end-2])  # class,id,len(2),payload
                if ca != ck_a or cb != ck_b:
                    i += 2
                    continue

            if cls_ == target_class and id_ == PVT_ID:
                total_msgs += 1
                rec = parse_nav_pvt(payload)
                if rec and rec["validDate"] and rec["validTime"] and rec["fixType"] in (1, 3, 4):
                    if hz and (rec["iTOW"] % period) != 0:
                        i = frame_end
                        continue

                    # gnssFixOK (Fix status flags bit0)
                    gnssFixOK = (rec.get("flags", 0) & 0x01) != 0

                    # Decide heading & KML color (AABBGGRR)
                    if not gnssFixOK:
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FFFF0000"  # blue
                    elif rec["fixType"] == 3:
                        heading_raw = rec["headMot"]; heading_src = "headMot"; color = "FF00C800"
                    elif rec["fixType"] == 4:
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FF00A5FF"
                    else:  # fixType == 1 (DR) or others
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FF800080"

                    heading_true = normalize_heading(heading_raw)
                    icon_heading = normalize_heading(heading_true + 180.0)

                    ts = f"{rec['year']:04d}-{rec['month']:02d}-{rec['day']:02d}T{rec['hour']:02d}:{rec['min']:02d}:{rec['sec']:02d}Z"
                    href = "https://maps.google.com/mapfiles/kml/shapes/arrow.png"

                    if alt_abs:
                        point_block = (
                            f"      <Point>\n"
                            f"        <extrude>1</extrude>\n"
                            f"        <altitudeMode>absolute</altitudeMode>\n"
                            f"        <coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['hMSL']:.3f}</coordinates>\n"
                            f"      </Point>\n"
                        )
                    else:
                        point_block = (
                            f"      <Point><coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['hMSL']:.3f}</coordinates></Point>\n"
                        )

                    buf.append(PLACEMARK_TEMPLATE.format(
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
                    if hz:
                        kept += 1

                    if want_html:
                        tsec = rec["iTOW"] / 1000.0
                        fix_counts[rec["fixType"]] += 1
                        ok_counts[1 if gnssFixOK else 0] += 1
                        fix_series.append({"t": tsec, "fixType": rec["fixType"]})
                        numsv_series.append({"t": tsec, "v": int(rec["numSV"])})
                        gspeed_series.append({"t": tsec, "v": float(rec["speed_m_s"])})
                        ok_series.append({"t": tsec, "ok": 1 if gnssFixOK else 0})

                    if (valid_msgs % PROGRESS_EVERY) == 0:
                        print(f"{now_str()} | Writing placemarks: {valid_msgs}"
                              f"{' / kept ' + str(kept) if hz else ''} "
                              f"(total frames: {total_msgs})")
            i = frame_end
    buf.append(FOOTER)
    print(f"{now_str()} | Finished doc.kml (frames: {total_msgs}, placemarks: {valid_msgs}"
          f"{', kept: ' + str(kept) if hz else ''})")

    kml_text = ''.join(buf)
    html_text = None
    if want_html:
        html_text = build_html_report(
            ubx_path=ubx_path,
            hz=hz,
            use_nav2=use_nav2,
            verify_ck=verify_ck,
            counts=dict(sorted(fix_counts.items())),
            series=fix_series,
            sv_series=numsv_series,
            gs_series=gspeed_series,
            ok_series=ok_series,
            ok_counts=dict(sorted(ok_counts.items()))
        )
    return kml_text, html_text

def write_kmz(kml_text: str, kmz_path: str) -> None:
    os.makedirs(os.path.dirname(kmz_path) or '.', exist_ok=True)
    with zipfile.ZipFile(kmz_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', kml_text.encode('utf-8'))
    print(f"{now_str()} | KMZ saved -> {kmz_path}")

def write_html(html_text: str, html_path: str) -> None:
    os.makedirs(os.path.dirname(html_path) or '.', exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print(f"{now_str()} | HTML saved -> {html_path}")

def build_html_report(ubx_path: str, hz, use_nav2, verify_ck, counts, series, sv_series, gs_series, ok_series, ok_counts):
    ok_total = sum(ok_counts.values()) or 0
    ok_good = ok_counts.get(1, 0)
    ok_bad  = ok_counts.get(0, 0)
    ok_pct = (ok_good / ok_total * 100.0) if ok_total else 0.0
    meta = {
        "file": os.path.basename(ubx_path),
        "hz": hz,
        "nav_mode": "NAV2-PVT" if use_nav2 else "NAV-PVT",
        "checksum": "on" if verify_ck else "off",
        "total_points": len(series),
        "ok_pct": ok_pct,
        "ok_good": ok_good,
        "ok_bad": ok_bad,
    }

    # Serialize data
    series_json    = json.dumps(series, ensure_ascii=False)
    sv_series_json = json.dumps(sv_series, ensure_ascii=False)
    gs_series_json = json.dumps(gs_series, ensure_ascii=False)
    ok_series_json = json.dumps(ok_series, ensure_ascii=False)

    # Build HTML safely: avoid f-string for JS blocks with { }
    head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UBX Fix Report - {meta["file"]}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; color:#222; }}
  h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
  .meta {{ font-size: 13px; color:#555; margin-bottom: 16px; }}
  .card {{ border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:0 1px 3px rgba(0,0,0,.04); margin-bottom:16px; }}
  .hint {{ font-size:12px; color:#666; margin-top:6px; }}
  .btn {{ display:inline-block; font-size:12px; padding:6px 8px; border:1px solid #ddd; border-radius:8px; background:#fafafa; cursor:pointer; margin-right:8px; }}
  .btn:active {{ transform: translateY(1px); }}
  canvas {{ width:100%; height:360px; border-radius:8px; background:#fff; cursor: crosshair; }}
.v-resizer {{ height:8px; margin-top:6px; border:1px solid #e5e7eb; border-radius:6px; background:#f1f5f9; cursor: row-resize; }}
.v-resizer:hover {{ background:#e2e8f0; }}
  .hidden {{ display: none; }}
  .toggles label {{ margin-right: 12px; user-select: none; }}
  footer {{ margin-top: 20px; font-size: 12px; color:#777; }}
</style>
</head>
<body>
  <h1>UBX Fix Report</h1>
  <div class="meta">
    <b>File:</b> {meta["file"]} &nbsp;•&nbsp;
    <b>Mode:</b> {meta["nav_mode"]} &nbsp;•&nbsp;
    <b>Downsample:</b> {meta["hz"] if meta["hz"] else "none"} &nbsp;•&nbsp;
    <b>Checksum:</b> {meta["checksum"]} &nbsp;•&nbsp;
    <b>Points:</b> {meta["total_points"]} &nbsp;•&nbsp;
    <b>FixOK:</b> {meta["ok_pct"]:.1f}% ({meta["ok_good"]}/{meta["ok_good"]+meta["ok_bad"]})
  </div>

  <div class="card toggles" id="controls">
    <b>Show/Hide:</b>
    <label><input type="checkbox" id="chkFix" checked> FixType</label>
    <label><input type="checkbox" id="chkSV" checked> numSV</label>
    <label><input type="checkbox" id="chkGS" checked> gSpeed</label>
    <label><input type="checkbox" id="chkOK" checked> gnssFixOK</label>
  </div>

  <div class="card" id="card-fix">
    <h3 style="margin:0 0 8px 0;">FixType Timeline (line + markers)</h3>
    <div style="font-size:12px; color:#666; margin-bottom:6px;">X = iTOW (seconds), Y = fixType</div>
    <div style="margin-bottom:8px;">
      <span class="btn" id="resetBtn">Reset View</span>
      <span class="hint">Drag = pan, Wheel = zoom. Double-click to zoom in, Shift+Double-click to zoom out.</span>
    </div>
    <canvas id="timeline" width="1200" height="360"></canvas>
    <div class="v-resizer" data-target="timeline" title="Drag to resize"></div>
  </div>

  <div class="card" id="card-sv">
    <h3 style="margin:0 0 8px 0;">numSV Timeline (line + markers)</h3>
    <div style="font-size:12px; color:#666; margin-bottom:6px;">X = iTOW (seconds), Y = numSV</div>
    <canvas id="svTimeline" width="1200" height="360"></canvas>
    <div class="v-resizer" data-target="svTimeline" title="Drag to resize"></div>
  </div>

  <div class="card" id="card-gs">
    <h3 style="margin:0 0 8px 0;">gSpeed Timeline (line + markers)</h3>
    <div style="font-size:12px; color:#666; margin-bottom:6px;">X = iTOW (seconds), Y = speed (m/s)</div>
    <canvas id="gsTimeline" width="1200" height="360"></canvas>
    <div class="v-resizer" data-target="gsTimeline" title="Drag to resize"></div>
  </div>

  <div class="card" id="card-ok">
    <h3 style="margin:0 0 8px 0;">gnssFixOK Timeline (0/1)</h3>
    <div style="font-size:12px; color:#666; margin-bottom:6px;">X = iTOW (seconds), Y = gnssFixOK (0=bad, 1=ok)</div>
    <canvas id="okTimeline" width="1200" height="220"></canvas>
    <div class="v-resizer" data-target="okTimeline" title="Drag to resize"></div>
  </div>

  <footer>Generated locally · No external libraries.</footer>

<script>
"""
    data_block = (
        "const SERIES = " + series_json + ";\n"
        "const SV_SER = " + sv_series_json + ";\n"
        "const GS_SER = " + gs_series_json + ";\n"
        "const OK_SER = " + ok_series_json + ";\n"
    )
    js_rest = r"""
// ===== 유틸 =====
function fixColor(ft) {
  if (ft === 3) return "#00C800";
  if (ft === 4) return "#00A5FF";
  if (ft === 1) return "#800080";
  return "#9E9E9E";
}
function drawPolylineWithMarkers(ctx, pts, color, r=2) {
  if (!pts.length) return;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
  ctx.fillStyle = color;
  for (const [x,y] of pts) { ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); }
}

// ===== 공통 뷰 상태 (타임라인 3개 공유) =====
let viewT = null;            // [minT, maxT]
const FIX_Y = [0,5];
let isDragging=false, dragStartX=0, dragStartT0=0;

// ===== FixType Timeline =====
function drawFixTimeline() {
  const cvs = document.getElementById('timeline');
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const padL=50, padR=10, padT=10, padB=40;
  const W=cvs.width, H=cvs.height;
  const N=SERIES.length; if (!N) return;

  const dataMinT=SERIES[0].t, dataMaxT=SERIES[N-1].t;
  let minT = viewT ? viewT[0] : dataMinT;
  let maxT = viewT ? viewT[1] : dataMaxT;
  if (maxT<=minT) maxT=minT+1e-6;
  const spanT=maxT-minT;

  const X=t=> padL + ((W-padL-padR)*(t-minT)/spanT);
  const Y=v=> H - padB - ((H-padT-padB)*(v-FIX_Y[0])/(FIX_Y[1]-FIX_Y[0]));

  // grid + y labels
  ctx.strokeStyle="#eee"; ctx.beginPath();
  for (let fy=FIX_Y[0]; fy<=FIX_Y[1]; fy++) { const yy=Y(fy); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); }
  ctx.stroke();
  ctx.fillStyle="#666"; ctx.font="12px sans-serif";
  for (let fy=FIX_Y[0]; fy<=FIX_Y[1]; fy++) { const yy=Y(fy); ctx.fillText(String(fy), 8, yy+4); }

  // x ticks
  ctx.fillStyle="#666"; ctx.font="12px sans-serif"; ctx.strokeStyle="#eee";
  const tickN=6;
  for (let k=0;k<=tickN;k++) {
    const tt=minT + spanT*k/tickN; const xx=X(tt);
    ctx.beginPath(); ctx.moveTo(xx,H-padB); ctx.lineTo(xx,padT); ctx.stroke();
    const label= spanT<10?tt.toFixed(2):(spanT<100?tt.toFixed(1):Math.round(tt).toString());
    ctx.fillText(label, xx-12, H-padB+16);
  }

  // axes
  ctx.strokeStyle="#ccc"; ctx.beginPath();
  ctx.moveTo(padL,H-padB); ctx.lineTo(W-padR,H-padB);
  ctx.moveTo(padL,padT);   ctx.lineTo(padL,H-padB); ctx.stroke();

  // visible + 다운샘플
  const pts = SERIES.filter(p=>p.t>=minT && p.t<=maxT);
  if (!pts.length) return;
  const pxW=Math.max(1,(W-padL-padR));
  const step=Math.max(1, Math.floor(pts.length/pxW));

  let prev=null;
  for (let i=0;i<pts.length;i+=step) {
    const cur=pts[i];
    const c = fixColor(cur.fixType);
    if (prev) {
      ctx.strokeStyle=c;
      ctx.beginPath();
      ctx.moveTo(X(prev.t), Y(prev.fixType));
      ctx.lineTo(X(cur.t),  Y(cur.fixType));
      ctx.stroke();
    }
    ctx.fillStyle=c;
    ctx.beginPath(); ctx.arc(X(cur.t), Y(cur.fixType), 2, 0, Math.PI*2); ctx.fill();
    prev=cur;
  }
}

// ===== numSV Timeline =====
function drawSVTimeline() {
  const cvs = document.getElementById('svTimeline');
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const padL=50, padR=10, padT=10, padB=40;
  const W=cvs.width, H=cvs.height;
  const N=SV_SER.length; if (!N) return;

  const dataMinT=SV_SER[0].t, dataMaxT=SV_SER[N-1].t;
  let minT = viewT ? viewT[0] : dataMinT;
  let maxT = viewT ? viewT[1] : dataMaxT;
  if (maxT<=minT) maxT=minT+1e-6;
  const spanT=maxT-minT;
  const X=t=> padL + ((W-padL-padR)*(t-minT)/spanT);

  // y-range
  const vpts = SV_SER.filter(p=>p.t>=minT && p.t<=maxT);
  let minV=Infinity, maxV=-Infinity;
  for (const p of (vpts.length? vpts : SV_SER)) { if (p.v<minV) minV=p.v; if (p.v>maxV) maxV=p.v; }
  if (!Number.isFinite(minV)) { minV=0; maxV=1; }
  if (minV==maxV) { minV-=1; maxV+=1; }
  const padV=Math.max(1,(maxV-minV)*0.1);
  minV=Math.floor(minV-padV); maxV=Math.ceil(maxV+padV);
  const Y=v=> H - padB - ((H-padT-padB)*(v-minV)/(maxV-minV));

  // grid/labels
  ctx.strokeStyle="#eee"; ctx.beginPath();
  const yTicks=6;
  for (let k=0;k<=yTicks;k++) { const vy=minV+(maxV-minV)*k/yTicks; const yy=Y(vy); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); }
  ctx.stroke();
  ctx.fillStyle="#666"; ctx.font="12px sans-serif";
  for (let k=0;k<=yTicks;k++) { const vy=minV+(maxV-minV)*k/yTicks; const yy=Y(vy); ctx.fillText(String(Math.round(vy)), 8, yy+4); }

  // x ticks
  ctx.fillStyle="#666"; ctx.font="12px sans-serif"; ctx.textBaseline="top"; ctx.strokeStyle="#eee";
  const tickN=6;
  for (let k=0;k<=tickN;k++) { const tt=minT + spanT*k/tickN; const xx=X(tt);
    ctx.beginPath(); ctx.moveTo(xx,H-padB); ctx.lineTo(xx,padT); ctx.stroke();
    const label= spanT<10?tt.toFixed(2):(spanT<100?tt.toFixed(1):Math.round(tt).toString());
    ctx.fillText(label, xx-12, H-padB+4);
  }

  // line+markers
  const pts = SV_SER.filter(p=>p.t>=minT && p.t<=maxT);
  if (!pts.length) return;
  const pxW=Math.max(1,(W-padL-padR));
  const step=Math.max(1, Math.floor(pts.length/pxW));
  const xy=[];
  for (let i=0;i<pts.length;i+=step) xy.push([X(pts[i].t), Y(pts[i].v)]);
  drawPolylineWithMarkers(ctx, xy, "#333", 2);
}

// ===== gSpeed Timeline =====
function drawGSTimeline() {
  const cvs = document.getElementById('gsTimeline');
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const padL=50, padR=10, padT=10, padB=40;
  const W=cvs.width, H=cvs.height;
  const N=GS_SER.length; if (!N) return;

  const dataMinT=GS_SER[0].t, dataMaxT=GS_SER[N-1].t;
  let minT = viewT ? viewT[0] : dataMinT;
  let maxT = viewT ? viewT[1] : dataMaxT;
  if (maxT<=minT) maxT=minT+1e-6;
  const spanT=maxT-minT;
  const X=t=> padL + ((W-padL-padR)*(t-minT)/spanT);

  // y-range
  const vpts = GS_SER.filter(p=>p.t>=minT && p.t<=maxT);
  let minV=Infinity, maxV=-Infinity;
  for (const p of (vpts.length? vpts : GS_SER)) { if (p.v<minV) minV=p.v; if (p.v>maxV) maxV=p.v; }
  if (!Number.isFinite(minV)) { minV=0; maxV=1; }
  if (minV==maxV) { minV-=1; maxV+=1; }
  const padV=Math.max(0.5,(maxV-minV)*0.1);
  minV=Math.floor(minV-padV); maxV=Math.ceil(maxV+padV);
  const Y=v=> H - padB - ((H-padT-padB)*(v-minV)/(maxV-minV));

  // grid/labels (Y)
  ctx.strokeStyle="#eee"; ctx.beginPath();
  const yTicks=6;
  for (let k=0;k<=yTicks;k++) { const vy=minV+(maxV-minV)*k/yTicks; const yy=Y(vy); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); }
  ctx.stroke();
  ctx.fillStyle="#666"; ctx.font="12px sans-serif";
  for (let k=0;k<=yTicks;k++) { const vy=minV+(maxV-minV)*k/yTicks; const yy=Y(vy); ctx.fillText(String(vy.toFixed(1)), 8, yy+4); }

  // x ticks + labels
  ctx.fillStyle="#666"; ctx.font="12px sans-serif"; ctx.textBaseline="top"; ctx.strokeStyle="#eee";
  const tickN=6;
  for (let k=0;k<=tickN;k++) {
    const tt=minT + spanT*k/tickN; const xx=X(tt);
    ctx.beginPath(); ctx.moveTo(xx,H-padB); ctx.lineTo(xx,padT); ctx.stroke();
    const label= spanT<10?tt.toFixed(2):(spanT<100?tt.toFixed(1):Math.round(tt).toString());
    ctx.fillText(label, xx-12, H-padB+4);
  }

  // axes
  ctx.strokeStyle="#ccc"; ctx.beginPath();
  ctx.moveTo(padL,H-padB); ctx.lineTo(W-padR,H-padB);
  ctx.moveTo(padL,padT);   ctx.lineTo(padL,H-padB); ctx.stroke();

  // line+markers
  const pts = GS_SER.filter(p=>p.t>=minT && p.t<=maxT);
  if (!pts.length) return;
  const pxW=Math.max(1,(W-padL-padR));
  const step=Math.max(1, Math.floor(pts.length/pxW));
  const xy=[];
  for (let i=0;i<pts.length;i+=step) xy.push([X(pts[i].t), Y(pts[i].v)]);
  drawPolylineWithMarkers(ctx, xy, "#1f6feb", 2);
}

// ===== OK Timeline =====
function drawOKTimeline() {
  const cvs = document.getElementById('okTimeline');
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const padL=50, padR=10, padT=10, padB=40;
  const W=cvs.width, H=cvs.height;
  const N=OK_SER.length; if (!N) return;

  const dataMinT=OK_SER[0].t, dataMaxT=OK_SER[N-1].t;
  let minT = viewT ? viewT[0] : dataMinT;
  let maxT = viewT ? viewT[1] : dataMaxT;
  if (maxT<=minT) maxT=minT+1e-6;
  const spanT=maxT-minT;
  const X=t=> padL + ((W-padL-padR)*(t-minT)/spanT);

  const minV=-0.1, maxV=1.1;
  const Y=v=> H - padB - ((H-padT-padB)*(v-minV)/(maxV-minV));

  // grid
  ctx.strokeStyle="#eee"; ctx.beginPath();
  for (let v=0; v<=1; v++) { const yy=Y(v); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); }
  ctx.stroke();
  ctx.fillStyle="#666"; ctx.font="12px sans-serif";
  ctx.fillText("0", 8, Y(0)+4); ctx.fillText("1", 8, Y(1)+4);

  // x ticks
  ctx.fillStyle="#666"; ctx.font="12px sans-serif"; ctx.strokeStyle="#eee";
  const tickN=6;
  for (let k=0;k<=tickN;k++) {
    const tt=minT + spanT*k/tickN; const xx=X(tt);
    ctx.beginPath(); ctx.moveTo(xx,H-padB); ctx.lineTo(xx,padT); ctx.stroke();
    const label= spanT<10?tt.toFixed(2):(spanT<100?tt.toFixed(1):Math.round(tt).toString());
    ctx.fillText(label, xx-12, H-padB+16);
  }

  // Draw steps / markers
  const pts = OK_SER.filter(p=>p.t>=minT && p.t<=maxT);
  if (!pts.length) return;
  const pxW=Math.max(1,(W-padL-padR));
  const step=Math.max(1, Math.floor(pts.length/pxW));
  ctx.strokeStyle="#444"; ctx.beginPath();
  ctx.moveTo(X(pts[0].t), Y(pts[0].ok));
  for (let i=1;i<pts.length;i+=step) {
    const prev = pts[i-1], cur = pts[i];
    ctx.lineTo(X(cur.t), Y(prev.ok)); // horizontal
    ctx.lineTo(X(cur.t), Y(cur.ok));  // vertical step
  }
  ctx.stroke();
  ctx.fillStyle="#111";
  for (let i=0;i<pts.length;i+=step) { ctx.beginPath(); ctx.arc(X(pts[i].t), Y(pts[i].ok), 2, 0, Math.PI*2); ctx.fill(); }
}

// ===== 상호작용 (공유) =====
function attachInteractions(id){
  const cvs = document.getElementById(id);
  cvs.addEventListener('mousedown', e => {
    isDragging=true; dragStartX=e.offsetX;
    const dm = SERIES.length ? SERIES[0].t : 0;
    const dM = SERIES.length ? SERIES[SERIES.length-1].t : 1;
    const curMin = viewT ? viewT[0] : dm;
    dragStartT0 = curMin; cvs.style.cursor='grabbing';
  });
  cvs.addEventListener('mouseup',   () => { isDragging=false; cvs.style.cursor='crosshair'; });
  cvs.addEventListener('mouseleave',() => { isDragging=false; cvs.style.cursor='crosshair'; });
  cvs.addEventListener('mousemove', e => {
    if (!isDragging || SERIES.length<2) return;
    const W=cvs.width, padL=50, padR=10, innerW=Math.max(1, W-padL-padR);
    const dm=SERIES[0].t, dM=SERIES[SERIES.length-1].t;
    let minT=viewT?viewT[0]:dm; let maxT=viewT?viewT[1]:dM;
    const spanT=Math.max(1e-6, maxT-minT);
    const dx=e.offsetX - dragStartX;
    const dT = -dx * (spanT/innerW);
    let start = dragStartT0 + dT;
    if (start<dm) start=dm;
    if (start+spanT>dM) start=dM-spanT;
    viewT=[start, start+spanT];
    drawFixTimeline(); drawSVTimeline(); drawGSTimeline(); drawOKTimeline();
  });
  cvs.addEventListener('wheel', e => {
    if (SERIES.length<2) return;
    e.preventDefault();
    const W=cvs.width, padL=50, padR=10, innerW=Math.max(1, W-padL-padR);
    const dm=SERIES[0].t, dM=SERIES[SERIES.length-1].t;
    let minT=viewT?viewT[0]:dm; let maxT=viewT?viewT[1]:dM;
    let spanT=Math.max(1e-6, maxT-minT);
    const mx=Math.min(Math.max(e.offsetX-padL,0), innerW);
    const focusT=minT + (mx/innerW)*spanT;
    const factor = (e.deltaY<0)?0.9:1.1;
    let newSpan=spanT*factor;
    const minSpan=(dM-dm)/1e6 + 1e-4;
    if (newSpan<minSpan) newSpan=minSpan;
    if (newSpan>(dM-dm)) newSpan=(dM-dm);
    let newMin=focusT - (mx/innerW)*newSpan;
    let newMax=newMin + newSpan;
    if (newMin<dm) { newMin=dm; newMax=newMin+newSpan; }
    if (newMax>dM) { newMax=dM; newMin=newMax-newSpan; }
    viewT=[newMin,newMax];
    drawFixTimeline(); drawSVTimeline(); drawGSTimeline(); drawOKTimeline();
  }, {passive:false});
  cvs.addEventListener('dblclick', e => {
    if (SERIES.length<2) return;
    const W=cvs.width, padL=50, padR=10, innerW=Math.max(1, W-padL-padR);
    const dm=SERIES[0].t, dM=SERIES[SERIES.length-1].t;
    let minT=viewT?viewT[0]:dm; let maxT=viewT?viewT[1]:dM;
    let spanT=Math.max(1e-6, maxT-minT);
    const mx=Math.min(Math.max(e.offsetX-padL,0), innerW);
    const focusT=minT + (mx/innerW)*spanT;
    const factor=e.shiftKey?1.25:0.8;
    let newSpan=spanT*factor;
    const minSpan=(dM-dm)/1e6 + 1e-4;
    if (newSpan<minSpan) newSpan=minSpan;
    if (newSpan>(dM-dm)) newSpan=(dM-dm);
    let newMin=focusT - (mx/innerW)*newSpan;
    let newMax=newMin + newSpan;
    if (newMin<dm) { newMin=dm; newMax=newMin+newSpan; }
    if (newMax>dM) { newMax=dM; newMin=newMax-newSpan; }
    viewT=[newMin,newMax];
    drawFixTimeline(); drawSVTimeline(); drawGSTimeline(); drawOKTimeline();
  });
}

function attachAllInteractions(){
  attachInteractions('svTimeline');
  attachInteractions('timeline');
  attachInteractions('gsTimeline');
  attachInteractions('okTimeline');
}

function renderAll(){
  drawSVTimeline();
  drawFixTimeline();
  drawGSTimeline();
  drawOKTimeline();
}

attachAllInteractions();
renderAll();
window.addEventListener('resize', renderAll);
document.getElementById('resetBtn').addEventListener('click', () => { viewT=null; renderAll(); });

// ===== Resize helpers =====
function setCanvasHeight(id, px) {
  const cvs = document.getElementById(id);
  if (!cvs) return;
  const clamped = Math.max(140, Math.min(700, Math.round(px)));
  // update both attribute and CSS to ensure canvas bitmap resizes
  cvs.height = clamped;
  cvs.style.height = clamped + "px";
  renderAll();
}

(function attachVerticalResizers(){
  let active = null; // {id, startY, startH}
  document.querySelectorAll('.v-resizer').forEach(bar => {
    bar.addEventListener('mousedown', (e) => {
      const id = bar.getAttribute('data-target');
      const cvs = document.getElementById(id);
      if (!cvs) return;
      active = { id, startY: e.clientY, startH: cvs.height };
      document.body.style.cursor = 'row-resize';
      e.preventDefault();
    });
  });
  window.addEventListener('mousemove', (e) => {
    if (!active) return;
    const dy = e.clientY - active.startY;
    setCanvasHeight(active.id, active.startH + dy);
  });
  window.addEventListener('mouseup', () => {
    if (active) {
      active = null;
      document.body.style.cursor = '';
    }
  });
})();


// ===== 카드 표시 토글 =====
function updateVisibility() {
  const m = [
    {chk: 'chkFix', card: 'card-fix'},
    {chk: 'chkSV',  card: 'card-sv' },
    {chk: 'chkGS',  card: 'card-gs' },
    {chk: 'chkOK',  card: 'card-ok' }
  ];
  let changed = false;
  m.forEach(({chk, card}) => {
    const c = document.getElementById(chk);
    const el = document.getElementById(card);
    if (!c || !el) return;
    const want = c.checked;
    const isHidden = el.classList.contains('hidden');
    if (want && isHidden) { el.classList.remove('hidden'); changed = true; }
    if (!want && !isHidden) { el.classList.add('hidden'); changed = true; }
  });
  if (changed) { renderAll(); }
}
['chkFix','chkSV','chkGS','chkOK'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', updateVisibility);
});
updateVisibility();
</script>
</body>
</html>
"""
    return head + data_block + js_rest

def run(ubx_path: str, hz: int = None, use_nav2: bool = False,
        alt_abs: bool = False, verify_ck: bool = False, html: bool = False, mapm: bool = False, progress_cb=None):
    base, _ = os.path.splitext(ubx_path)
    if mapm:
        kmz_path = base + "_mapm.kmz"
        print(f"{now_str()} | MAPM-only mode: writing {kmz_path}")
        kml_text = build_kml_mapm_only(ubx_path, alt_abs=alt_abs, verify_ck=verify_ck, progress_cb=progress_cb)
        write_kmz(kml_text, kmz_path)
        return

    suffix = "_nav2" if use_nav2 else "_nav"
    if hz:
        suffix = f"_{hz}hz" + suffix
    if alt_abs:
        suffix = suffix + "_abs"
    suffix = suffix + ("_ck" if verify_ck else "_nock")
    kmz_path = base + suffix + ".kmz"
    html_path = base + suffix + ".html" if html else None

    print(f"{now_str()} | KMZ mode: {kmz_path}{' + HTML' if html else ''}")
    kml_text, html_text = build_kml(
        ubx_path, hz=hz, use_nav2=use_nav2,
        alt_abs=alt_abs, verify_ck=verify_ck,
        want_html=html, progress_cb=progress_cb
    )
    write_kmz(kml_text, kmz_path)
    if html and html_text is not None:
        write_html(html_text, html_path)

def main():
    ap = argparse.ArgumentParser(description="UBX NAV/NAV2-PVT -> KMZ (+ optional HTML report incl. gnssFixOK) + AID-MAPM (--mapm)")
    ap.add_argument("ubx", help="Input UBX file path")
    ap.add_argument("--hz", type=int, choices=[1, 2, 5, 10],
                    help="Downsample to given rate (Hz). If omitted, keep all frames.")
    ap.add_argument("--nav2", action="store_true",
                    help="Use NAV2-PVT instead of NAV-PVT")
    ap.add_argument("--alt-abs", action="store_true",
                    help="Use altitudeMode=absolute with extrude=1 in <Point>")
    ap.add_argument("--ck", action="store_true",
                    help="Enable UBX checksum verification (default: off)")
    ap.add_argument("--html", action="store_true",
                    help="Also write an offline HTML report with fixType/numSV/gSpeed/gnssFixOK")
    ap.add_argument("--mapm", action="store_true",
                    help="Parse only UBX-AID-MAPM and overlay white arrows in KMZ; ignores NAV/NAV2")
    args = ap.parse_args()
    run(args.ubx, hz=args.hz, use_nav2=args.nav2,
        alt_abs=args.alt_abs, verify_ck=args.ck, html=args.html, mapm=args.mapm)

if __name__ == "__main__":
    main()
