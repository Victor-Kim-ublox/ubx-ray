import math
import re
from datetime import datetime

def dm_to_dd(nmea_coord):
    """
    NMEA 포맷(DDMM.MMMM)을 Decimal Degrees(DD.DDDD)로 변환합니다.
    """
    if not nmea_coord:
        return 0.0
    
    if '.' in nmea_coord:
        split_idx = nmea_coord.index('.') - 2
    else:
        split_idx = len(nmea_coord) - 2
        
    degrees = float(nmea_coord[:split_idx])
    minutes = float(nmea_coord[split_idx:])
    
    return degrees + (minutes / 60.0)

def parse_nmea_line(line):
    """
    NMEA 라인 파싱. $GNGGA, $GPGGA 처리.
    """
    try:
        if not line.startswith('$'):
            return None
            
        parts = line.split(',')
        
        if 'GGA' not in parts[0]:
            return None
            
        if not parts[2] or not parts[4]:
            return None

        raw_time = parts[1] # HHMMSS.SS
        
        lat = dm_to_dd(parts[2])
        if parts[3] == 'S': lat = -lat
            
        lon = dm_to_dd(parts[4])
        if parts[5] == 'W': lon = -lon
            
        return {
            'time': raw_time,
            'lat': lat,
            'lon': lon
        }
    except Exception:
        return None

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    두 지점 사이의 거리(m) 계산
    """
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def format_time_for_label(raw_time):
    """
    HHMMSS.SS -> HH:MM:SS 변환 (그래프 X축 라벨용)
    """
    try:
        # 길이가 충분한지 확인
        if len(raw_time) >= 6:
            return f"{raw_time[0:2]}:{raw_time[2:4]}:{raw_time[4:6]}"
        return raw_time
    except:
        return raw_time

def analyze_nmea_files(ref_file_path, test_file_path):
    """
    두 NMEA 파일을 비교하여 통계 및 그래프 데이터를 생성합니다.
    """
    ref_data = {}
    test_data = {}

    # 1. 파일 읽기
    with open(ref_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_nmea_line(line.strip())
            if parsed: ref_data[parsed['time']] = parsed

    with open(test_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_nmea_line(line.strip())
            if parsed: test_data[parsed['time']] = parsed

    # 2. 비교 및 데이터 생성
    results = []
    errors = []
    
    # 그래프용 데이터 배열 (X축: 시간, Y축: 오차)
    graph_labels = [] 
    graph_values = []

    common_times = sorted(list(set(ref_data.keys()) & set(test_data.keys())))

    for t in common_times:
        ref_pt = ref_data[t]
        test_pt = test_data[t]
        
        distance = calculate_haversine_distance(
            ref_pt['lat'], ref_pt['lon'],
            test_pt['lat'], test_pt['lon']
        )
        
        errors.append(distance)
        
        # 상세 데이터 저장
        results.append({
            'time': t,
            'ref_lat': ref_pt['lat'],
            'ref_lon': ref_pt['lon'],
            'test_lat': test_pt['lat'],
            'test_lon': test_pt['lon'],
            'error_meters': distance
        })

        # 그래프용 데이터 저장
        graph_labels.append(format_time_for_label(t))
        graph_values.append(round(distance, 3)) # 소수점 3자리까지 반올림

    # 3. 통계 계산
    if not errors:
        return {"status": "error", "message": "No matching timestamps found."}

    sorted_errors = sorted(errors)
    stats = {
        "count": len(errors),
        "max_error": round(max(errors), 3),
        "min_error": round(min(errors), 3),
        "avg_error": round(sum(errors) / len(errors), 3),
        "cep_50": round(sorted_errors[int(len(errors) * 0.5)], 3),
        "cep_95": round(sorted_errors[int(len(errors) * 0.95)], 3)
    }

    return {
        "status": "success",
        "statistics": stats,
        "graph_data": {
            "labels": graph_labels, # X축 (시간)
            "values": graph_values  # Y축 (오차 m)
        },
        # "raw_data": results # 필요하다면 주석 해제하여 원본 데이터도 전송
    }