# templates/report_nmea.html

## Overview
The **result report page** for NMEA comparison analysis. Displays position error statistics between a Reference and Test Device NMEA file as stat cards, and visualizes error over time as a Chart.js line chart.

---

## URL
Rendered directly after `POST /analyze_nmea` (server returns HTML; no redirect)

---

## Jinja2 Template Variables

| Variable | Type | Description |
|---|---|---|
| `stats` | dict | Statistics dictionary returned by `nmea_comparison.py` |
| `graph_data` | str | JSON string (`{"labels": [...], "values": [...]}`) |

### `stats` Dictionary Structure

| Key | Description |
|---|---|
| `count` | Number of matched timestamps |
| `max_error` | Maximum error (m) |
| `min_error` | Minimum error (m) |
| `avg_error` | Mean error (m) |
| `cep_50` | CEP50 — 50th percentile circular error (m) |
| `cep_95` | CEP95 — 95th percentile circular error (m) |

---

## Screen Layout

### Header
- UBX-ray brand (gradient text)
- `← Home` button

### Summary Statistics Card
6-item stat-box grid:
```
Matched Points | Max Error | Min Error
Avg Error      | CEP50     | CEP95
```
Each box: metric label (muted) + value (white, large font) + unit

### Visualization Card (Position Error Over Time)
Chart.js line chart:
- **X-axis**: UTC time (HH:MM:SS format)
- **Y-axis**: Error (m)
- Line color: `#6ea8ff` (accent)
- Area fill: gradient (semi-transparent)
- Point radius: 0 (line only, for performance)
- Tooltip: time + error value

```javascript
const graphData = JSON.parse('{{ graph_data | safe }}');
new Chart(ctx, {
  type: 'line',
  data: {
    labels: graphData.labels,
    datasets: [{
      data: graphData.values,
      borderColor: '#6ea8ff',
      fill: true,
      backgroundColor: gradient,
      pointRadius: 0,
      tension: 0.3
    }]
  }
});
```

---

## Design Theme
Shared dark-blue theme with `home.html` and `report_nmea.html`:
- `--bg1: #0f1226`, `--bg2: #11163a`
- `--card: #151a44`
- `--accent: #6ea8ff`, `--accent2: #9d7bff`
- `--shadow: 0 10px 40px rgba(0,0,0,.35)...`

### stat-box Style
```css
.stat-box {
  background: #0d1540;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
}
.stat-label { font-size: 12px; color: var(--muted); }
.stat-value { font-size: 28px; font-weight: 800; color: #fff; }
.stat-unit  { font-size: 13px; color: var(--muted); }
```

---

## Dependencies (CDN)
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js">
```

---

## Notes
- Has no dedicated URL — rendered directly as the response body of POST `/analyze_nmea`
- In `home.html`, `startComparison()` replaces the page using `document.write(html)`
- Analysis results are not persisted to the database (session-only, one-time result)
