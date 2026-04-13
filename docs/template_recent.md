# templates/recent.html

## Overview
A **recent analysis results list** page scoped to the current user (identified by cookie). When accessed with an admin token, displays results from all users.

---

## URL
`GET /recent`

Admin access: include `Authorization: Bearer {ADMIN_TOKEN}` header to retrieve up to 50 results across all users.

---

## Jinja2 Template Variables

| Variable | Type | Description |
|---|---|---|
| `rows` | list[tuple] | DB results rows (up to 50, most recent first) |

Row field order:
`(id, filename, uploaded_at, epoch_total, epoch_missing, crc_errors, kmz_path, opts_json, status, error)`

---

## Screen Layout

### Header
- "Recent Results" title
- `← Home` link

### Results Table

| Column | Description |
|---|---|
| # | Row number |
| Filename | Original filename |
| Uploaded | Upload timestamp (UTC) |
| Epochs | epoch_total (NAV-PVT frame count) |
| Missing | epoch_missing count + percentage |
| Status | Status badge |
| Actions | Report / Map / Download links |

### Status Badges

| Status | Color | Meaning |
|---|---|---|
| `done` | green | Conversion complete |
| `queued` | yellow | Waiting in queue |
| `running` | blue | Conversion in progress |
| `error` | red | Conversion failed |

### Action Links (active only when `done`)

| Button | Link | Description |
|---|---|---|
| Report | `/report/{rid}` | Analysis report page |
| Map | `/map/{rid}` | Map viewer |
| KMZ | `/download?path={kmz_path}` | Download KMZ file |

---

## Epoch Missing Display

```
epoch_missing > 0  →  "123 (5.2%)" - warning color (yellow)
epoch_missing == 0 →  "0" - normal color (green)
epoch_missing NULL →  "-" (before conversion)
```

---

## Table Style
- `table-layout: auto` + `white-space: nowrap` → automatic column widths, no line wrapping
- `max-width: 1200px`, centered layout
- Row background brightens on hover

---

## Design
Light theme:
- `--bg: #f9fafc`, `--card-bg: #ffffff`, `--border: #e5e7eb`
- `--primary: #0078ff`
- Status badges use `--success-*`, `--error-*`, `--warn-*` CSS variables

---

## Security
- Results are scoped to **the current user's cookie** only (`user_id` filter applied)
- Admin (Bearer token) can view up to 50 results across all users
- KMZ downloads go through `/download` which re-validates ownership server-side
