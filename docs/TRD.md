# Technical Reference Document (TRD)
## SentinelAI — Driver Drowsiness Detection System

---

## Architecture

```
Browser (index.html)
    │
    ├── GET  /video_feed  ──────────► MJPEG stream (multipart/x-mixed-replace)
    │
    └── GET/POST /api/*  ──────────► JSON REST API
                                          │
                                     Flask (app.py)
                                          │
                              ┌───────────┴───────────┐
                         detector.py              OpenCV
                              │
                         MediaPipe FaceMesh
```

---

## Backend — Flask (`app.py`)

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/start` | POST | Init camera + detector, set `is_running=True` |
| `/api/stop` | POST | Release camera, destroy detector |
| `/api/status` | GET | Return current `detector.state` as JSON |
| `/api/reset` | POST | Reset session counters in detector |
| `/api/health` | GET | Liveness check |
| `/video_feed` | GET | MJPEG stream via `generate_frames()` generator |

Camera is protected by a `threading.Lock` to prevent concurrent read/write.

---

## Detector (`detector.py`)

### MediaPipe Landmark Indices

| Feature | Indices |
|---------|---------|
| Left Eye | 33, 160, 158, 133, 153, 144 |
| Right Eye | 362, 385, 387, 263, 380, 373 |
| Mouth Top | 13 |
| Mouth Bottom | 14 |
| Mouth Left | 78 |
| Mouth Right | 308 |

### Eye Aspect Ratio (EAR)

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

- p1, p4 = horizontal corners
- p2, p3, p5, p6 = vertical pairs

### Mouth Aspect Ratio (MAR)

```
MAR = ||top - bottom|| / ||left - right||
```

### Drowsiness Logic

```
eyes_closed  = EAR < 0.25
closed_frames += 1 (if closed), else reset to 0
DROWSY = closed_frames >= 20

yawning      = MAR > 0.55
yawn_frames += 1 (if yawning), else reset to 0
YAWNING = yawn_frames >= 15
```

Status priority: DROWSY > YAWNING > AWAKE

---

## Frontend (`script.js`)

- Polls `/api/status` every 300ms when running
- Health-checks `/api/health` every 3s when stopped
- If backend already running on page reload, auto-reconnects
- Calculates risk: `score = drowsy_events × 3 + yawn_events`
  - score 0 → LOW, ≤3 → MEDIUM, >3 → HIGH

---

## Dependencies

```
flask==3.0.3
flask-cors==4.0.1
opencv-python-headless==4.10.0.84
mediapipe==0.10.14
numpy==1.26.4
scipy==1.13.1
```
