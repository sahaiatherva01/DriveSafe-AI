# Implementation Notes
## SentinelAI — Driver Drowsiness Detection System

---

## Key Implementation Decisions

### 1. No flask-cors dependency
CORS is handled manually via Flask's `@app.after_request` hook. This removes an external dependency and makes the project easier to set up on any machine.

### 2. Camera opened only on `/api/start`
The webcam is never touched until the user explicitly clicks Start. This avoids permission errors and frees the camera for other apps when not in use.

### 3. `cam_lock` scope
The threading lock wraps only the `camera.read()` call — not the entire frame processing pipeline. This ensures the MJPEG stream and stop/start routes can't race on the camera object.

### 4. Rolling average smoother
EAR and MAR values are smoothed over a window of 5 frames to prevent single-blink false positives. Window size is tunable in `utils.py`.

### 5. Edge-detected event counting
Drowsy and yawn events are counted only on the **rising edge** (transition from not-drowsy to drowsy), not on every frame while the state persists. This gives accurate event totals.

### 6. MJPEG stream via `<img>` tag
The browser `<img>` element natively handles `multipart/x-mixed-replace` MJPEG streams. No WebSocket or custom decoding needed. The `onload` event is used to fade in the feed once the first frame arrives.

### 7. Auto-reconnect on page reload
If the backend is already running when the page loads, `healthCheck()` detects `running: true` from `/api/health` and automatically restores the video feed and starts polling — no need to click Start again.

---

## File Roles

| File | Role |
|------|------|
| `app.py` | Flask server: routes, camera lifecycle, MJPEG stream |
| `detector.py` | All CV logic: MediaPipe, EAR, MAR, overlays, state machine |
| `utils.py` | Pure math utilities: distance, EAR, MAR, smoother |
| `index.html` | Dashboard structure: two tabs, all DOM elements |
| `style.css` | HUD dark theme: CSS variables, all component styles |
| `script.js` | All frontend logic: API calls, DOM updates, tab switching |

---

## Known Limitations

- Single face only (MediaPipe configured with `max_num_faces=1`)
- No audio alert
- No screenshot on drowsy event
- Session data lost on backend restart (no persistence)
- MJPEG stream may lag slightly on slow machines

---

## Tested On

- macOS (primary target)
- Python 3.10+, 3.11, 3.12
- Chrome, Firefox, Safari
