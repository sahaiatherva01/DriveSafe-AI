# Application Flow
## SentinelAI — Driver Drowsiness Detection System

---

## Startup Flow

```
User opens frontend/index.html
        │
        ▼
script.js loads → startHealthCheck() polls /api/health every 3s
        │
        ├─ Backend UP   → connDot turns green, "Connected"
        └─ Backend DOWN → connDot stays grey, "Offline"
```

---

## Detection Start Flow

```
User clicks "Start Detection"
        │
        ▼
POST /api/start
        │
        ├─ Camera opens (cv2.VideoCapture(0))
        │         │
        │         ├─ FAIL → 500 response → alert shown to user
        │         └─ OK   → DrowsinessDetector() created
        │                        │
        │                   is_running = True
        │
        ▼
Frontend receives { success: true }
        │
        ├─ Sets img src → http://localhost:5000/video_feed
        ├─ Starts polling /api/status every 300ms
        └─ Stops health check polling
```

---

## Per-Frame Processing Loop

```
generate_frames() [runs in Flask thread]
        │
        ├─ camera.read() → BGR frame
        ├─ cv2.flip(frame, 1) → mirror
        ├─ detector.process_frame(frame)
        │         │
        │         ├─ MediaPipe FaceMesh.process(rgb)
        │         ├─ Extract 478 face landmarks
        │         ├─ Calculate EAR (left + right avg)
        │         ├─ Calculate MAR
        │         ├─ Smooth via rolling average (window=5)
        │         ├─ Update closed_frames / yawn_frames counters
        │         ├─ Determine status: AWAKE / DROWSY / YAWNING
        │         ├─ Count new events (edge detection on state change)
        │         └─ Draw overlays on frame
        │
        ├─ cv2.imencode('.jpg', annotated)
        └─ yield MJPEG chunk → browser displays in <img>
```

---

## Status Polling Loop

```
Every 300ms: GET /api/status
        │
        ▼
applyStatus(data)
        │
        ├─ Update status ring color (green/red/yellow/grey)
        ├─ Update status icon + label (AWAKE/DROWSY/YAWNING/NO FACE)
        ├─ Update EAR value + progress bar + color coding
        ├─ Update MAR value + progress bar + color coding
        ├─ Update eye state badge (Open/Closed)
        ├─ Update face detection badge (Detected/None)
        ├─ Show/hide alert banner with message
        ├─ Update session counters (drowsy events, yawns, time)
        ├─ Calculate risk score → LOW / MEDIUM / HIGH
        └─ Sync Session Report tab values
```

---

## Stop Flow

```
User clicks "Stop"
        │
        ▼
POST /api/stop
        │
        ├─ is_running = False  → generate_frames() exits loop
        ├─ camera.release()
        └─ detector.release() → face_mesh.close()

Frontend:
        ├─ Clears img src → camera overlay shown
        ├─ Status ring cleared
        ├─ Restores Start button
        └─ Resumes health check polling
```

---

## Risk Scoring

```
score = (drowsy_events × 3) + (yawn_events × 1)

score == 0   → LOW    (green)
score <= 3   → MEDIUM (yellow)
score  > 3   → HIGH   (red)
```
