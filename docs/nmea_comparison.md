# nmea_comparison.py

## Overview
An analysis module that matches two NMEA files (Reference vs. Test Device) by timestamp and calculates **position error (in meters)**, returning statistics and chart data.

Imported in `app.py` as `from nmea_comparison import analyze_nmea_files`, and executed without blocking the async event loop via `run_in_threadpool`.

---

## Supported NMEA Sentences

`$GNGGA` and `$GPGGA` (any talker ID containing GGA is accepted)

GGA fields used:
- `parts[1]`: UTC time `HHMMSS.SS` (used as matching key)
- `parts[2]`, `parts[3]`: latitude + N/S
- `parts[4]`, `parts[5]`: longitude + E/W

---

## Key Functions

### `dm_to_dd(nmea_coord)`
Converts NMEA DDMM.MMMM format to Decimal Degrees.

```
degrees = leading digits (total digits minus last 2 before decimal)
minutes = remaining portion
result  = degrees + minutes / 60.0
```

### `parse_nmea_line(line)`
Parses a single GGA sentence and returns `{'time': str, 'lat': float, 'lon': float}`.
- Returns `None` if the line does not start with `$` or does not contain GGA
- Returns `None` if latitude or longitude fields are empty

### `calculate_haversine_distance(lat1, lon1, lat2, lon2)`
Calculates the **Haversine distance (meters)** between two GPS coordinates.
- Earth radius R = 6,371,000 m
- Appropriate for small-scale error analysis

### `format_time_for_label(raw_time)`
Converts `HHMMSS.SS` → `HH:MM:SS`. Used for chart X-axis labels.

### `analyze_nmea_files(ref_file_path, test_file_path)`
Main analysis function.

---

## Analysis Processing Flow

```
1. Parse ref file  → ref_data  = { time_str: {time, lat, lon} }
2. Parse test file → test_data = { time_str: {time, lat, lon} }
3. Find intersection of common timestamps (sorted)
4. Calculate haversine distance at each timestamp
5. Compute statistics
6. Return results
```

---

## Return Format

### Success
```json
{
  "status": "success",
  "statistics": {
    "count": 1500,
    "max_error": 12.345,
    "min_error": 0.123,
    "avg_error": 2.456,
    "cep_50": 1.987,
    "cep_95": 8.123
  },
  "graph_data": {
    "labels": ["12:00:01", "12:00:02", ...],
    "values": [1.234, 0.987, ...]
  }
}
```

### Failure
```json
{
  "status": "error",
  "message": "No matching timestamps found."
}
```

---

## Statistics Explained

| Metric | Description |
|---|---|
| `count` | Number of matched common timestamps |
| `max_error` | Maximum position error (m) |
| `min_error` | Minimum position error (m) |
| `avg_error` | Mean position error (m) |
| `cep_50` | CEP50 — 50th percentile error (m) |
| `cep_95` | CEP95 — 95th percentile error (m) |

CEP (Circular Error Probable): a standard metric for position accuracy. CEP50 means 50% of measurements fall within this radius.

---

## Limitations / Notes

- **Timestamp matching requires exact string equality.** If files use different output rates or have time offsets, the number of matched points will be reduced.
- Midnight boundary (day rollover) is not handled — matching is based on `HHMMSS` alone, so data is assumed to be within the same calendar day.
- Only GGA sentences are parsed; RMC, VTG, and others are ignored.
- File encoding: `utf-8`, with `errors='ignore'` (non-standard bytes are silently skipped).
