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
SAT_ID     = 0x35

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
    """Parse UBX-AID-MAPM (length typically 28 bytes)."""
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

def build_kml_mapm_only(ubx_path: str, alt_abs: bool = False, verify_ck: bool = False):
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
        while i + 8 <= n:
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
    
    # Check length validity
    if len(payload) < 8 + 12 * numSvs:
        return None

    cnos = []
    offset = 8
    for _ in range(numSvs):
        # Block structure: gnssId(1), svId(1), cno(1), elev(1), azim(2), prRes(2), flags(4)
        cno = payload[offset + 2] # dBHz
        
        if cno > 0: # 유효한 신호만 수집
            cnos.append(cno)
        offset += 12
        
    return {"iTOW": iTOW, "cnos": cnos}

def normalize_heading(deg: float) -> float:
    h = math.fmod(deg if deg is not None else 0.0, 360.0)
    if h < 0:
        h += 360.0
    return h

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
        "cno_labels": [],      
        "cno_top_avg": [],     
        "cno_scatter": [],
        # [추가] 통계 정보 저장용
        "stats": {
            "epoch_total": 0,
            "epoch_missing": 0
        }
    }
    
    kept_itows = set() 
    itow_to_cno = {}

    target_class = NAV2_CLASS if use_nav2 else NAV_CLASS
    period = 1000 // hz if hz else None

    buf.append(HEADER)
    with open(ubx_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        n = len(mv)
        i = 0
        while i + 8 < n:
            if mv[i] != UBX_SYNC1 or mv[i+1] != UBX_SYNC2:
                i += 1
                continue
            if i + 6 >= n: break
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
                if rec and rec["validDate"] and rec["validTime"] and rec["fixType"] in (1, 3, 4):
                    
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

                    # KML Placemark 생성 (기존 유지)
                    gnssFixOK = (rec.get("flags", 0) & 0x01) != 0
                    if not gnssFixOK:
                        if rec["headVeh"] is not None:
                            heading_raw = rec["headVeh"]; heading_src = "headVeh"
                        else:
                            heading_raw = rec["headMot"]; heading_src = "headMot"
                        color = "FFFF0000"
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

                    ts = f"{rec['year']:04d}-{rec['month']:02d}-{rec['day']:02d}T{rec['hour']:02d}:{rec['min']:02d}:{rec['sec']:02d}Z"
                    href = "https://maps.google.com/mapfiles/kml/shapes/arrow.png"
                    
                    if alt_abs:
                        point_block = f"      <Point>\n        <extrude>1</extrude>\n        <altitudeMode>absolute</altitudeMode>\n        <coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['hMSL']:.3f}</coordinates>\n      </Point>\n"
                    else:
                        point_block = f"      <Point><coordinates>{rec['lon']:.7f},{rec['lat']:.7f},{rec['hMSL']:.3f}</coordinates></Point>\n"

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
                    if hz: kept += 1

            # 2. NAV-SAT 처리
            elif cls_ == NAV_CLASS and id_ == SAT_ID:
                sat_rec = parse_nav_sat(payload)
                if sat_rec:
                    itow_to_cno[sat_rec["iTOW"]] = sat_rec["cnos"]

            i = frame_end

    # [최적화된 로직] Merge PVT Time and SAT CN0 (Scatter Downsampling 포함)
    sorted_itows = sorted(list(kept_itows))
    
    estimated_points = len(sorted_itows) * 25
    MAX_POINTS = 300000 
    
    scatter_stride = 1
    if estimated_points > MAX_POINTS:
        scatter_stride = math.ceil(estimated_points / MAX_POINTS)
        print(f"{now_str()} | High density data detected. Downsampling CN0 scatter by factor of {scatter_stride}...")

    all_scatter_points = []
    
    for idx, itow in enumerate(sorted_itows):
        if itow in itow_to_cno:
            cnos = itow_to_cno[itow]
            
            # [수정] 신호가 없으면(empty list) 0으로 처리하여 그래프 끊김 방지
            avg_cno = 0.0
            if cnos:
                sorted_cnos = sorted(cnos, reverse=True)
                top_k = sorted_cnos[:5]
                avg_cno = sum(top_k) / 5

            # 데이터가 0이어도 항상 추가
            graph_data["cno_labels"].append(itow)
            graph_data["cno_top_avg"].append(round(avg_cno, 1))
            
            # Scatter는 데이터가 있을 때만 추가
            if cnos and (idx % scatter_stride == 0):
                for c in cnos:
                    all_scatter_points.append({"x": itow, "y": c})

    graph_data["cno_scatter"] = all_scatter_points

    # [수정] 통계 정보 최종 저장
    graph_data["stats"]["epoch_total"] = valid_msgs
    graph_data["stats"]["epoch_missing"] = missing_epochs

    buf.append(FOOTER)
    print(f"{now_str()} | Finished doc.kml (Total: {valid_msgs}, Missing: {missing_epochs}, kept: {kept if hz else 'All'})")

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
    
    # JSON 파일 경로
    json_path = base + "_graph.json"

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