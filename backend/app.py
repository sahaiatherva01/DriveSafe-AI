import cv2
import threading
import time
from flask import Flask, Response, jsonify, request

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

from detector import DrowsinessDetector

app = Flask(__name__)

# ── CORS (manual — no flask-cors dependency needed) ───────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>',             methods=['OPTIONS'])
def handle_options(path):
    return '', 204

# ── Global camera state ───────────────────────────────────────────────────────
camera     = None
detector   = None
cam_lock   = threading.Lock()
is_running = False

# Snapshot of the most recently completed session, captured at stop-time
# (the live detector/camera are released right after stopping, so the
# /api/agent and /api/analytics routes need this cache to serve the
# post-session report).
last_session_report    = None
last_session_analytics = None
last_session_meta      = None   # {'start': ts, 'end': ts, 'frames': int}


def open_camera():
    """Open and configure the webcam. Returns the capture object."""
    global camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
    camera = cap
    return cap


def release_camera():
    """Release the webcam and destroy the detector."""
    global camera, detector, is_running
    is_running = False
    if camera is not None:
        if camera.isOpened():
            camera.release()
        camera = None
    if detector is not None:
        try:
            detector.release()
        except Exception:
            pass
        detector = None


# ── Stream generator ──────────────────────────────────────────────────────────
def generate_frames():
    """MJPEG generator — reads frames, runs detector, yields JPEG chunks."""
    global is_running, detector
    while is_running:
        with cam_lock:
            if camera is None or not camera.isOpened():
                time.sleep(0.05)
                continue
            success, frame = camera.read()

        if not success or frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)              # mirror so it feels natural
        annotated, _ = detector.process_frame(frame)

        ret, buffer = cv2.imencode(
            '.jpg', annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )

        time.sleep(1 / 30)      # ~30 fps cap


# ── API routes ────────────────────────────────────────────────────────────────
@app.route('/api/start', methods=['POST'])
def start_detection():
    global detector, is_running
    if is_running:
        return jsonify({'success': True, 'message': 'Already running'})

    with cam_lock:
        cap = open_camera()
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Cannot access webcam. '
                           'Make sure no other application is using it.'
            }), 500

    detector   = DrowsinessDetector()
    is_running = True
    print('[SentinelAI] Detection started')
    return jsonify({'success': True, 'message': 'Detection started'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    global last_session_report, last_session_analytics, last_session_meta

    with cam_lock:
        if detector is not None:
            # Capture the final analysis snapshot BEFORE the detector/camera
            # are released — the report page needs this after the session ends.
            try:
                last_session_report = detector.get_agent_report()
            except Exception:
                last_session_report = None

            try:
                timeline = detector._safety_agent._timeline[-200:]
                session_start = detector.session_start
                points = []
                for entry in timeline:
                    t_sec = round(entry['ts'] - session_start, 1)
                    f, ai = entry['features'], entry['ai']
                    points.append({
                        't':           t_sec,
                        'ear':         f.get('ear',            0.0),
                        'mar':         f.get('mar',             0.0),
                        'attention':   f.get('attention_index', 0.0),
                        'fatigue':     f.get('fatigue_index',   0.0),
                        'blink_rate':  f.get('blink_rate',      0.0),
                        'risk_index':  ai.get('risk_index',     0),
                        'pitch':       f.get('pitch',           0.0),
                        'yaw':         f.get('yaw',             0.0),
                    })
                last_session_analytics = points
            except Exception:
                last_session_analytics = None

            last_session_meta = {
                'start':  detector.session_start,
                'end':    time.time(),
                'frames': detector.frame_count,
            }

        release_camera()
    print('[SentinelAI] Detection stopped')
    return jsonify({'success': True, 'message': 'Detection stopped'})


@app.route('/api/status')
def get_status():
    if not is_running or detector is None:
        return jsonify({
            'running':         False,
            'face_detected':   False,
            'ear':             0.0,
            'mar':             0.0,
            'eye_state':       'N/A',
            'status':          'OFFLINE',
            'drowsy':          False,
            'yawning':         False,
            'closed_frames':   0,
            'session_seconds': 0,
            'drowsy_events':   0,
            'yawn_events':     0,
        })
    state = dict(detector.state)
    state['running'] = True
    return jsonify(state)


@app.route('/api/reset', methods=['POST'])
def reset_session():
    if detector:
        detector.reset_session()
    return jsonify({'success': True, 'message': 'Session reset'})


@app.route('/video_feed')
def video_feed():
    if not is_running:
        return jsonify({'error': 'Detection not started'}), 400
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'running': is_running})


@app.route('/api/agent')
def agent_report():
    """Safety Agent analysis — observations, recommendations, driver grade."""
    if is_running and detector is not None:
        return jsonify(detector.get_agent_report())

    if last_session_report is not None:
        return jsonify(last_session_report)

    return jsonify({
        'observations':    ['Start a session to receive safety analysis.'],
        'recommendations': ['Start detection first.'],
        'driver_grade':    '—',
        'grade_reason':    '',
        'grade_score':     0,
        'summary':         'No session active.',
        'generated_at':    0,
    })


@app.route('/api/analytics')
def analytics():
    """
    Return time-series data for the analytics charts.
    Samples from the agent's internal timeline (recorded every 3s).
    Falls back to the cached final-session snapshot once stopped.
    """
    if is_running and detector is not None:
        timeline = detector._safety_agent._timeline[-120:]   # last 6 minutes max
        session_start = detector.session_start

        points = []
        for entry in timeline:
            t_sec = round(entry['ts'] - session_start, 1)
            f     = entry['features']
            ai    = entry['ai']
            points.append({
                't':           t_sec,
                'ear':         f.get('ear',            0.0),
                'mar':         f.get('mar',            0.0),
                'attention':   f.get('attention_index', 0.0),
                'fatigue':     f.get('fatigue_index',   0.0),
                'blink_rate':  f.get('blink_rate',      0.0),
                'risk_index':  ai.get('risk_index',     0),
                'pitch':       f.get('pitch',           0.0),
                'yaw':         f.get('yaw',             0.0),
            })
        return jsonify({'timeline': points, 'running': True})

    if last_session_analytics is not None:
        return jsonify({'timeline': last_session_analytics, 'running': False})

    return jsonify({'timeline': [], 'running': False})


@app.route('/api/session_meta')
def session_meta():
    """Start/end timestamps and frame count for the most recent session."""
    if is_running and detector is not None:
        return jsonify({
            'start':  detector.session_start,
            'end':    None,
            'frames': detector.frame_count,
        })
    if last_session_meta is not None:
        return jsonify(last_session_meta)
    return jsonify({'start': None, 'end': None, 'frames': 0})


@app.route('/api/system')
def system_metrics():
    """Return CPU, RAM, and camera resolution for the dashboard."""
    cpu  = round(psutil.cpu_percent(interval=None), 1) if PSUTIL_OK else 0.0
    ram  = round(psutil.virtual_memory().percent, 1)   if PSUTIL_OK else 0.0
    cam_w, cam_h = 0, 0
    if camera and camera.isOpened():
        cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return jsonify({
        'cpu_percent':  cpu,
        'ram_percent':  ram,
        'cam_width':    cam_w,
        'cam_height':   cam_h,
        'psutil_available': PSUTIL_OK,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('🚗  SentinelAI — http://localhost:5000')
    print('    Open frontend/index.html in your browser to begin.')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
