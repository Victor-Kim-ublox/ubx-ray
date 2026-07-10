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
| `--mapm` | Extract AID-MAPM points only (sky-blue arrows, absolute-position only) |
| `--progress-file PATH` | Write scan progress (percent of bytes scanned, one decimal) to `PATH`, throttled to one write per 0.5 s (`PROGRESS_INTERVAL_SEC`); ends with `100.0`. Best-effort — write failures never abort the conversion. `app.py` points this at `{upload}.progress` and `/api/status` serves it to the web progress bar |

---

## Supported UBX Messages

| Message | Class | ID | Description |
|---|---|---|---|
| NAV-PVT | 0x01 | 0x07 | Position, velocity, and time (default) |
| NAV2-PVT | 0x29 | 0x07 | NAV2 protocol, same payload format |
| NAV-SAT | 0x01 | 0x35 | Per-SV CN0 used for the CN0 chart |
| AID-MAPM | 0x0B | 0x05 | Map-matching points (sky-blue arrows) — parsed alongside NAV-PVT in the default path, or standalone with `--mapm` |
| SEC-SIG | 0x27 | 0x09 | Jamming / spoofing status (version 0x02) |
| INF-* | 0x04 | 0x00–0x04 | Receiver text messages (ERROR/WARNING/NOTICE/TEST/DEBUG) |

### UBX-INF messages

Class `0x04` carries ASCII text emitted by the receiver. All five levels are
collected into `graph_data["inf_messages"]` as `{itow, utc, level, text}` in
file order (capped at `INF_MAX = 2000`). INF frames have no time of their own,
so each is tagged with the **nearest preceding PVT** time reference
(`inf_ref_itow` / `inf_ref_utc`, updated for any valid-time epoch incl. no-fix);
messages before the first fix have `itow`/`utc` = `null`. A printable-ratio
guard (≥ 80% printable ASCII) drops false sync matches when checksum
verification is off. Rendered in the report as the "Receiver Messages" table.

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
| nano | 16–19 | signed ns offset of the second; gives the timestamp two decimal places (`SS.cc`, may roll the second backwards when negative) |
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
  <TimeStamp><when>2024-01-01T12:00:00.00Z</when></TimeStamp>
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

Epochs with `fixType` in `(1, 2, 3, 4)` are kept; `0` (no fix) and `5`
(time only) carry no usable position and are skipped entirely.

| fixType | Color | Meaning |
|---|---|---|
| 3 | `FF00C800` green | 3D Fix |
| 4 | `FF00A5FF` orange | GNSS + Dead Reckoning |
| 2 | `FF00FFFF` yellow | 2D Fix |
| 1 | `FF800080` purple | Dead Reckoning only |

If `gnssFixOK=0`, the icon is always red (`FFFF0000`) regardless of fix type.

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
- iTOW (0–3), flags (4–5, X2), headMM (6–7, ×1e-2 = degrees), lat (8–11, ×1e-7),
  lon (12–15, ×1e-7), alt (16–19, ×1e-3)
- pos_acc (20–21, ×0.1 m), alt_acc (22–23, ×0.1 m), head_acc (24–25, ×0.01°)
- reserved0 (26–27)

`flags` bits decoded to booleans: `0` **latLonValid**, `1` **altValid**,
`2` **headValid**, `3` **hmsl**, `6` **relativePos**.

### Rendering (sky-blue arrows, `MAPM_COLOR = FFEBCE87`)

AID-MAPM points are rendered as **sky-blue** arrow Placemarks
(`mapm_placemark()` → `MAPM_PLACEMARK_TEMPLATE`, `<name>AID-MAPM</name>`,
scale 0.5, icon rotated `heading + 180°` like the NAV-PVT arrows) so they are
visually distinct from the fixType-coloured vehicle track.

- **Default path (`build_kml`)** — AID-MAPM frames with `latLonValid = 1` are
  buffered during the scan and **resolved after** it, once every NAV-PVT fix is
  known. Two independent time references are kept per point:
  - **Delta anchor** — when `relativePos = 1`, `lat`/`lon` are **deltas** (same
    1e-7 deg units) added to the NAV-PVT fix at the **same iTOW** as the MAPM
    message (`itow_to_fix` map; nearest iTOW as a fallback, skipped only if no
    fix exists at all). Deferred resolution means the matching fix is found even
    when it appears *later* in the byte stream.
  - **`arrived iTOW`** — the iTOW of the primary NAV-PVT that *preceded* the
    MAPM frame in **stream order**, captured at scan time. The message's own
    `itowMM` is the map-matching *solution* time, which lags the arrival point,
    so this shows after which NAV-PVT the message actually arrived. Shown on
    **every** point (relative and absolute) directly under the message's own
    `iTOW`.

  For `relativePos` points the position block shows **both** the raw delta
  (`ΔLat`/`ΔLon`) and the resulting `computed` absolute Lat/Lon; absolute points
  show a plain Lat/Lon.
- **`--mapm` mode (`build_kml_mapm_only`)** — emits *only* AID-MAPM Placemarks
  and has no NAV reference, so `relativePos` points are skipped there; only
  absolute (`latLonValid`, non-`relativePos`) points are output.

`map.html` playback excludes `<name>AID-MAPM</name>` Placemarks (they are
map-matching markers, not vehicle-track epochs).

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

  "primary_pvt": "NAV-PVT" | "NAV2-PVT",   // class used for the KML track
  "alt_pvt":     "NAV2-PVT" | "NAV-PVT",   // the *other* PVT class
  "alt_labels":  [iTOW, ...],   // accuracy series of the other PVT class;
  "alt_acc2d":   [float, ...],  // empty when the log carries no frames of
  "alt_acc3d":   [float, ...],  // that class. Same validity + Hz filtering.
  // cno_scatter has been removed — only the Top-5 line is rendered now, and
  // emitting every tracked satellite per epoch ballooned the graph JSON.

  "sec_labels":       [iTOW, ...],
  "jam_state":        [int, ...],
  "spf_state":        [int, ...],
  "jam_det_enabled":  [0|1, ...],
  "spf_det_enabled":  [0|1, ...],
  "sec_freqs":        [[{"freq_mhz": float, "jammed": bool}, ...], ...],

  "inf_messages":     [{"itow": int|null, "utc": "…Z"|null,
                        "level": "ERROR|WARNING|NOTICE|TEST|DEBUG",
                        "text": str}, ...]
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
- **Sync scanning via `mmap.find()`** — the next `B5 62` pattern is located by
  the C-implemented `find`, not a per-byte Python loop. Non-UBX content between
  frames (interleaved NMEA sentences, text headers, corrupt runs) is skipped at
  native speed, and resync after a checksum failure is equally cheap.
- **Precompiled `struct.Struct` unpackers** (`_PVT_HEAD`, `_PVT_DYN`,
  `_PVT_HEADVEH`) — NAV-PVT fields are read with two `unpack_from` calls
  instead of re-parsing format strings / repeated `int.from_bytes` per frame.
- **f-string Placemark rendering** (`pvt_placemark`) — replaces a
  `str.format` template that re-parsed ~1 KB of template text per epoch.
- **Lazy INF time reference** — the UTC string for `inf_ref_*` is rendered only
  when an INF frame actually arrives, not for every PVT epoch; `pvt_utc_str`
  itself uses a pure-integer fast path (no `datetime`/`strftime`) except for the
  rare negative-`nano` / roll-over cases.
- Measured on a 1 h synthetic log (10 Hz PVT + 1 Hz NAV-SAT, 5 MB): clean
  5.2 → 7.6 MB/s, NMEA-interleaved 6.5 → 12.0 MB/s, `--ck` 4.1 → 5.6 MB/s,
  with byte-identical KML/graph output.
- With `--ck` disabled, frame boundaries are located using only sync bytes + class/id + length — no CRC computation, very fast
- Progress log output every `PROGRESS_EVERY=1000` frames
