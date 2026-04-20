# ubx2kmz.py

## Overview
The core conversion script that transforms u-blox UBX binary log files into **KMZ (compressed KML)** format. Called by `app.py` via subprocess. Parses NAV-PVT or NAV2-PVT messages and converts each epoch into a Google Earth / OpenLayers compatible Placemark.

---

## Usage
```bash
python ubx2kmz.py <ubx_file> [options]
```

### CLI Options

| Option | Description |
|---|---|
| `--nav2` | Use NAV2-PVT (class=0x29, id=0x07). Default is NAV-PVT (class=0x01, id=0x07) |
| `--hz {1,2,5,10}` | Keep only epochs aligned to the given Hz (based on iTOW ms) |
| `--alt-abs` | Apply KML `altitudeMode=absolute` + `extrude=1` |
| `--ck` | Enable Fletcher checksum verification (default: off, speed priority) |
| `--mapm` | Extract AID-MAPM points only (white arrows) |

---

## Supported UBX Messages

| Message | Class | ID | Description |
|---|---|---|---|
| NAV-PVT | 0x01 | 0x07 | Position, velocity, and time (default) |
| NAV2-PVT | 0x29 | 0x07 | NAV2 protocol, same payload format |
| NAV-SAT | 0x01 | 0x35 | Per-SV CN0 used for the CN0 chart |
| AID-MAPM | 0x0B | 0x05 | Map-matching points (white arrows, `--mapm`) |
| SEC-SIG | 0x27 | 0x09 | Jamming / spoofing status (version 0x02) |

NAV-PVT payload valid lengths: 92 or 96 bytes (`VALID_LEN_SET`)

### NAV-SAT sanity filters

With checksum verification disabled (the default), a false sync-byte match can occasionally be interpreted as a NAV-SAT frame. `parse_nav_sat` applies two guards to drop these:

- `numSvs > 128` → frame rejected. Real receivers track ≤ ~60 SVs; higher counts indicate a garbage header.
- per-SV `cno` outside `(0, 63]` dBHz → that block's CN0 is discarded. Valid u-blox CN0 values are at most 63 dBHz; larger bytes are from mis-aligned payloads.
- per-SV `qualityInd < 4` (from flags bits 2..0) → that block's CN0 is discarded. Per u-blox spec: 0=no signal, 1=searching, 2=acquired, 3=detected but unusable, 4+=code/carrier locked. NAV-SAT frames contain stale / placeholder rows (e.g. duplicated `gnssId=0, svId=3, cno=60, qualityInd=0`) whose `cno` field is not from a real tracked signal; counting them inflates the Top-5 average with spurious spikes up to ~52 dBHz.

The "Top 5 Avg" is now divided by `len(top_k)` instead of a fixed 5, so epochs with fewer than 5 tracked signals are not artificially pulled down.

---

## Key Parsed Fields (NAV-PVT payload)

| Field | Offset | Conversion |
|---|---|---|
| iTOW | 0–3 | ms (32-bit unsigned) |
| year/month/day/hour/min/sec | 4–9 | Constructs UTC timestamp |
| fixType | 20 | 0=NoFix, 2=2D, 3=3D, 4=GNSS+DR, 5=Time-only |
| flags | 21 | gnssFixOK bit (bit 0) |
| lon | 24–27 | ×1e-7 → degrees |
| lat | 28–31 | ×1e-7 → degrees |
| height | 32–35 | mm → m |
| hAcc | 40–43 | mm → m (horizontal accuracy) |
| vAcc | 44–47 | mm → m (vertical accuracy) |
| gSpeed | 60–63 | mm/s → m/s (ground speed) |
| sAcc | 68–71 | mm/s → m/s (speed accuracy) |
| headMot | 64–67 | ×1e-5 → degrees (heading of motion) |
| headVeh | 84–87 | ×1e-5 → degrees (heading of vehicle) |
| headAcc | 72–75 | ×1e-5 → degrees (heading accuracy) |

---

## KML Output Structure

Each NAV-PVT epoch produces one `<Placemark>`:

```xml
<Placemark>
  <TimeStamp><when>2024-01-01T12:00:00Z</when></TimeStamp>
  <Style>
    <IconStyle>
      <color>FF00FF00</color>   <!-- color based on fixType -->
      <scale>0.5</scale>
      <heading>180.0</heading>  <!-- icon rotation direction -->
      <Icon><href>...arrow.png</href></Icon>
    </IconStyle>
  </Style>
  <description><![CDATA[
    UTC, iTOW, FixType, Flags, Heading, Speed, Lat, Lon, Acc...
  ]]></description>
  <Point><coordinates>lon,lat,alt</coordinates></Point>
</Placemark>
```

### Icon Color by fixType (KML AABBGGRR format)

| fixType | Color | Meaning |
|---|---|---|
| 3 or 4 | `FF00FF00` green | Normal 3D / DR Fix |
| 2 | `FF00FFFF` yellow | 2D Fix |
| 0 or 1 | `FF0000FF` red | No Fix / Dead Reckoning only |

If `gnssFixOK=0`, the icon is always red regardless of fix type.

---

## Heading Selection Logic (`pick_heading`)

1. If `headVeh` is valid (flags2 bit5=1), use vehicle heading
2. Otherwise, use `headMot`
3. Icon is rotated opposite to direction of travel (`heading + 180°`) — arrow tail points in the direction of motion

---

## Hz Filtering Logic

When `--hz N` is specified:
- Only epochs where `iTOW % (1000 / N) == 0` are included
- Example: `--hz 1` → 1000 ms interval, `--hz 5` → 200 ms interval

---

## AID-MAPM Parsing (`parse_aid_mapm`)

Payload is 28 bytes:
- iTOW (0–3), headMM (6–7, ×1e-2 = degrees), lat (8–11, ×1e-7), lon (12–15, ×1e-7), alt (16–19, ×1e-3)
- pos_acc (20–21, ×0.1 m), alt_acc (22–23, ×0.1 m), head_acc (24–25, ×0.01°)

In `--mapm` mode, only AID-MAPM entries are output as white arrow Placemarks.

---

## Graph JSON Output

After conversion, a `_graph.json` file is saved alongside the KMZ. Used for chart rendering in `app.py`'s report view.

```json
{
  "stats": { "epoch_total": 1234, "epoch_missing": 5 },
  "labels":    [iTOW, ...],
  "acc2d":     [float, ...],
  "acc3d":     [float, ...],
  "fix_type":  [int, ...],
  "speed":     [float, ...],
  "altitude":  [float, ...],
  "num_sv":    [int, ...],
  "lat":       [float, ...],
  "lon":       [float, ...],
  "cno_labels":  [iTOW, ...],
  "cno_top_avg": [float, ...],
  // cno_scatter has been removed — only the Top-5 line is rendered now, and
  // emitting every tracked satellite per epoch ballooned the graph JSON.

  "sec_labels":       [iTOW, ...],
  "jam_state":        [int, ...],
  "spf_state":        [int, ...],
  "jam_det_enabled":  [0|1, ...],
  "spf_det_enabled":  [0|1, ...],
  "sec_freqs":        [[{"freq_mhz": float, "jammed": bool}, ...], ...]
}
```

### UBX-SEC-SIG (jamming / spoofing)

Parsed by `parse_sec_sig()` from class `0x27` id `0x09`, message version `0x02`.

Payload layout:

| Offset | Field | Type | Notes |
|---|---|---|---|
| 0 | `version` | U1 | Must be `0x02` |
| 1 | `sigSecFlags` | X1 | bit 0 `jamDetEnabled`, bits 2..1 `jamState`, bit 3 `spfDetEnabled`, bits 5..4 `spfState` |
| 2 | reserved | U1 | |
| 3 | `jamNumCentFreqs` | U1 | `N` |
| 4 + 4·n | `jamStateCentFreq` | X4 | bits 23..0 `centFreq` (kHz), bit 24 `jammed` |

State enumerations:

- `jamState`: `0`=Unknown, `1`=OK (no jamming), `2`=Warning (jamming indicated)
- `spfState`: `0`=Unknown, `1`=OK, `2`=Indicated, `3`=Affirmed

SEC-SIG frames carry no iTOW. Each frame is associated with the most recent valid NAV-PVT iTOW, and only samples whose iTOW is retained in the graph (after Hz filtering) appear in the output arrays.

---

## Output File Naming Convention

Auto-generated based on the input filename stem:

| Option Combination | Output Filename Pattern |
|---|---|
| Default | `{stem}_nav_nock.kmz` |
| `--nav2` | `{stem}_nav2_nock.kmz` |
| `--ck` | `{stem}_nav_ck.kmz` |
| `--alt-abs` | `{stem}_nav_abs_nock.kmz` |
| `--mapm` | `{stem}_mapm.kmz` |

`app.py` uses glob to find the most recently modified `.kmz` and copies it to `outputs/{rid}/result.kmz`.

---

## Performance Considerations

- Uses `mmap` + `memoryview` for memory-efficient processing of large files
- With `--ck` disabled, frame boundaries are located using only sync bytes + class/id + length — no CRC computation, very fast
- Progress log output every `PROGRESS_EVERY=1000` frames
