# templates/compare4_report.html

## Overview
The **comparison analysis report screen** for Multi .ubx Analysis. Polls the conversion status of up to 4 UBX files in real time, then displays statistics (epochs, accuracy, speed, etc.) side by side in charts and tables once conversion completes.

---

## URL
`GET /compare4/report/{rid1}/{rid2}/{rid3}/{rid4}`

---

## Jinja2 Template Variables

| Variable | Type | Description |
|---|---|---|
| `rids` | list[str] | 4 result IDs (`"_"` for empty slots) |
| `filenames` | list[str] | Filename corresponding to each rid |

---

## Screen Layout

### Header (sticky)
- ubX-ray brand logo
- Per-file color badges (File 1–4: blue / red / green / purple)
- Buttons: `⊞ Split Map View` → `/compare4/view/{rids}` (opens in a new tab via `target="_blank"`), `⊙ Overlay Map View` → `/compare4/overlay/{rids}` (also new tab), `🕘 Recent`. The `← New` button was removed; to start a fresh comparison use the brand link or navigate to `/compare4` directly.

### Status Section
Conversion status card for each file:
- Spinning loader + `"Converting…"` while polling
- Green checkmark + filename once `done`

### Summary Stats Cards
Core statistics displayed side by side for all converted files:
- Epoch Total / Missing (epoch count, missing rate %)
- Average hAcc (mean horizontal accuracy, m)
- Average Speed (km/h)
- Max Speed (km/h)

### Chart Section
Interactive charts powered by Chart.js (zoom/pan supported):

| Chart | X-axis | Y-axis | Description |
|---|---|---|---|
| Position Accuracy (hAcc) | Time | m | Horizontal accuracy time series |
| Speed | Time | km/h | Speed time series |
| Fix Type | Time | 0–5 | Fix type time series |

Each file is distinguished by its unique color (c1–c4).

---

## Real-time Polling Logic (JavaScript)

```javascript
const POLL_INTERVAL = 2000; // poll every 2 seconds

async function pollAll() {
  for (rid of activeRids) {
    const res = await fetch(`/api/status/${rid}`);
    const data = await res.json();

    if (data.status === 'done') {
      // fetch graph data
      const graph = await fetch(`/api/graph/${rid}`);
      renderChart(graph);
      markDone(rid);
    } else if (data.status === 'error') {
      showError(rid);
    }
  }

  if (pendingRids.length > 0) {
    setTimeout(pollAll, POLL_INTERVAL);
  }
}
```

Polling stops automatically once all active rids are complete.

---

## Chart Interaction
- **Chart.js** `@4.4.0`
- **chartjs-plugin-zoom** `@2.0.1` + **hammerjs** `@2.0.8` (pinch-zoom / drag-pan)
- Each section has a single `🔍 Reset Zoom` `.ctl-btn`
- Charts are grouped by key in `chartRegistry`; zoom/pan on any chart
  propagates to its group via `syncGroup`:
  - `acc` — Accuracy Overlay
  - `spdaltsv` — Speed, Altitude, and Satellite Count charts share one group
    so panning/zooming any of the three moves the other two in lockstep
  - `cno` — CNO Top-5 Avg
- **Tooltip alignment** — all time-series charts use a custom interaction
  mode `interaction: { mode:'xOnePerDataset', intersect:false }`, **not**
  `mode:'index'` or plain `mode:'x'`.
  - Each track is plotted as an independent `{x, y}` array (different start
    iTOW, length, and `downsampleXY` stride). `mode:'index'` matched points
    by array position and showed values from mismatched timestamps (hovered
    track read 405 m while another read 0.7 m from a different time).
  - Plain `mode:'x'` fixed the time alignment but could list a single track
    multiple times (several points fall in the nearest x-bucket after
    downsampling).
  - `xOnePerDataset` (registered once via
    `Chart.Interaction.modes.xOnePerDataset`) walks the `'x'` candidates and
    keeps only the single closest element per `datasetIndex`, so the tooltip
    shows exactly one time-aligned row per track.

---

## Dependencies (CDN)
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js">
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js">
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/...">
```
Google Fonts Inter.

---

## Color Coding

| File | CSS Variable | Hex |
|---|---|---|
| File 1 | `--c1` | `#0078ff` blue |
| File 2 | `--c2` | `#e03e3e` red |
| File 3 | `--c3` | `#16a34a` green |
| File 4 | `--c4` | `#9333ea` purple |

---

## API Usage
- `GET /api/status/{rid}`: polls conversion status
- `GET /api/graph/{rid}`: loads graph JSON data for chart rendering
