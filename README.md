# SentinelAI — AI Driver Monitoring System

A real-time driver monitoring dashboard built with Python, Flask, MediaPipe, OpenCV, optional PyTorch/TensorFlow, and a single-page Vanilla JS frontend.

**Python 3.10.19+ compatible.** Cross-platform: Windows, macOS, Linux.

---

## Features

- Single-page glassmorphism dashboard (Live Feed → Metrics → Charts → Analytics → Performance → Timeline → AI Agent → Summary)
- Real-time EAR / MAR / blink / yawn / microsleep detection
- Head pose estimation (pitch, yaw, roll) via `solvePnP`
- Attention & fatigue indices, temporal smoothing
- Hybrid AI decision engine: rule-based CV + PyTorch MLP + TensorFlow temporal smoother (graceful fallback if either is absent)
- Local offline AI Safety Agent: observations, recommendations, A–F driver grade
- Live Chart.js charts: EAR, MAR, attention/fatigue, risk timeline, blink rate, head pose
- System performance panel (CPU, RAM, FPS, latency) via `psutil`
- One-click Start / Stop / Reset, auto-reconnect on page reload

---

## Project Structure

```
SentinelAI/
├── backend/
│   ├── app.py            # Flask server, routes, camera streaming, system metrics
│   ├── detector.py        # MediaPipe pipeline, overlays, state assembly
│   ├── features.py        # CV feature engineering (blink/pose/attention/fatigue)
│   ├── ai_model.py        # PyTorch MLP, TF temporal smoother, hybrid engine, agent
│   ├── utils.py            # EAR/MAR geometry, rolling smoother
│   ├── run.py              # Launcher — downloads MediaPipe model, starts Flask
│   └── requirements.txt
├── frontend/
│   ├── index.html          # Single-page dashboard (all sections, no tabs)
│   ├── style.css           # Glassmorphism dark theme
│   └── script.js           # Polling, charts, controls, agent, timeline
└── docs/
```

---

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Optional — enables PyTorch MLP and TensorFlow temporal smoother
pip install torch tensorflow-cpu

python run.py
```

Open `frontend/index.html` in your browser, click **Start**.

If PyTorch / TensorFlow are not installed, the system automatically falls back to rule-based risk classification and exponential moving-average smoothing — the dashboard works identically either way.

---

## API Reference

| Method | Endpoint        | Description                          |
|--------|-----------------|---------------------------------------|
| POST   | /api/start      | Initialize camera and detector        |
| POST   | /api/stop       | Stop detection, release camera        |
| POST   | /api/reset      | Reset session counters                |
| GET    | /api/status     | Current detection + AI state (JSON)   |
| GET    | /api/health     | Backend health check                  |
| GET    | /api/system     | CPU, RAM, camera resolution           |
| GET    | /api/agent      | Safety agent analysis + driver grade  |
| GET    | /api/analytics  | Time-series data for charts           |
| GET    | /video_feed     | MJPEG annotated video stream          |

---

## Tech Stack

- **Backend**: Python 3.10+, Flask, OpenCV, MediaPipe Tasks API, NumPy, SciPy, psutil
- **AI** (optional): PyTorch (risk MLP), TensorFlow (temporal smoother) — both with rule-based/EMA fallback
- **Frontend**: HTML5, CSS3 (glassmorphism), Vanilla JavaScript, Chart.js
- **Fonts**: Space Grotesk, Inter, JetBrains Mono

---

## Requirements

- Python 3.10.19 or newer
- Webcam connected and accessible
- Modern browser (Chrome, Firefox, Edge, Safari)
