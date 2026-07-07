# Design Document
## SentinelAI — Driver Drowsiness Detection System

---

## Visual Design System

### Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#080c18` | Page background |
| `--bg2` | `#0d1424` | Secondary background |
| `--card` | `#111827` | Card background |
| `--card-border` | `rgba(0,212,255,0.08)` | Default card border |
| `--cyan` | `#00d4ff` | Primary accent, EAR bar, controls |
| `--purple` | `#7c3aed` | Logo gradient end |
| `--green` | `#22c55e` | AWAKE state, safe indicators |
| `--yellow` | `#f59e0b` | YAWNING state, medium risk |
| `--red` | `#ef4444` | DROWSY state, high risk, alert |
| `--text-1` | `#f0f6ff` | Primary text |
| `--text-2` | `#94a3b8` | Secondary text |
| `--text-3` | `#475569` | Muted labels |

### Typography

| Role | Font | Weight |
|------|------|--------|
| Brand, headings | Space Grotesk | 700 |
| Body, labels | Inter | 400–600 |
| Metric values | JetBrains Mono | 400–500 |

Monospace for numbers ensures values don't cause layout shifts as digits change.

### Spacing

- Base unit: `0.75rem` (12px)
- Card padding: `1.25rem` (20px)
- Grid gap: `1.25rem`
- Border radius: `12px` (cards), `8px` (small elements)

---

## Layout

### Dashboard Tab (two-column grid)

```
┌─────────────────┬───────────────────────────────────┐
│  LIVE FEED      │  LIVE METRICS                     │
│  [camera]       │  [EAR bar]  [MAR bar]             │
│                 │  [Eye State] [Face Detection]      │
│  [Start][Stop]  │                                   │
│  [Reset]        │  [Alert Banner]                   │
│                 │                                   │
│  DRIVER STATUS  │  DETECTION SUMMARY                │
│  [icon][label]  │  [Drowsy] [Yawn] [Time] [Risk]   │
└─────────────────┴───────────────────────────────────┘
```

### Session Report Tab

```
┌──────────┬──────────┬──────────┬──────────┐
│  Time    │  Drowsy  │  Yawns   │  Risk    │
├──────────┴──────────┴──────────┴──────────┤
│  Safety Recommendation (full width)       │
└───────────────────────────────────────────┘
```

---

## Status States

| Status | Ring | Icon bg | Label color | Hint |
|--------|------|---------|-------------|------|
| AWAKE | Green glow | Green tint | `--green` | Driver is alert |
| DROWSY | Red pulse animation | Red tint | `--red` | ⚠ Pull over |
| YAWNING | Yellow glow | Yellow tint | `--yellow` | Stay alert |
| NO FACE | Grey | Grey tint | `--text-3` | No face in frame |
| OFFLINE | None | None | `--text-3` | Start detection |

---

## Responsive Breakpoints

| Width | Layout |
|-------|--------|
| > 900px | Two-column dashboard, four-column report |
| 560–900px | Single-column dashboard, two-column report |
| < 560px | Single-column everything, hidden brand subtitle |
