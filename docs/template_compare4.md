# templates/compare4.html

## Overview
The **standalone upload page** for Multi .ubx Analysis. Rendered when accessing `/compare4` directly. Provides the same functionality as the Multi tab in `home.html`, but as a separate standalone page.

> **Note**: When submitting the Multi tab form in `home.html`, the same `/compare4/upload` endpoint is used — not this page. `compare4.html` is rendered only on `GET /compare4`.

---

## URL
`GET /compare4`

---

## Tab Navigation

Uses the same tab style as the home screen, but navigation is implemented as **links between pages** rather than in-page tab switching:
- `Single .ubx Analysis` → `href="/"`
- `Multi .ubx Analysis` → current page (active span, not clickable)
- `NMEA Comparison` → `href="/?tab=nmea"`

---

## Form Structure

**Action**: `POST /compare4/upload`
**Encoding**: `multipart/form-data`

### File Slots (2×2 Grid)

| Slot | Input Name | Required |
|---|---|---|
| File 1 | `file1` | required |
| File 2 | `file2` | optional |
| File 3 | `file3` | optional |
| File 4 | `file4` | optional |

- Each slot supports drag-and-drop + click to browse
- On file selection: dropzone border changes to `#6ea8ff` and filename is displayed
- Optional slots start at opacity 0.75 and become 1.0 when a file is selected

### Conversion Options

| Option | Description |
|---|---|
| `hz` | Downsampling (None / 1 / 2 / 5 / 10 Hz) |
| `nav2` | Use NAV2-PVT messages |
| `alt_abs` | Set altitudeMode=absolute |
| `mapm` | Extract AID-MAPM points only |

All options are applied equally to all 4 files.

---

## Submit Behavior (JavaScript)

```javascript
// Validate File 1 is selected
if (!file1Input.files.length) {
  alert('Please select at least File 1.');
  return;
}
// Count how many files are selected
// Disable button + show spinner + loading text
// "Uploading N file(s)... please wait"
```

Standard HTML form submit (not fetch) → server issues 303 redirect to `/compare4/report/{r1}/{r2}/{r3}/{r4}`.

Empty slots are handled server-side as sentinel `"_"` and appear as empty panels in the comparison view.

---

## Design

Shares the same dark-blue CSS variables and components as `home.html`. Tab styles, dropzone styles, buttons, and fieldsets all use shared styling.

Unique components:
- `.grid4`: `grid-template-columns: 1fr 1fr` two-column grid
- `.slot`: flex column wrapping dropzone + slot-label
- `.slot-label`: file slot title (uppercase, accent color)
- `.card-title`: "Multi .ubx Analysis" heading (18px, bold)
- `.card-sub`: subtitle (13px, muted color)
- `.notice`: instructions info box

---

## Related Endpoints
- Form submits to → `POST /compare4/upload`
- After completion, redirects to → `GET /compare4/report/{rid1}/{rid2}/{rid3}/{rid4}`
