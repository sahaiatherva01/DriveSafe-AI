"""
detector.py — Core Detection Module (Extended)
===============================================
Wraps MediaPipe FaceLandmarker + FeatureExtractor + HybridEngine.

Responsibilities
  • Run MediaPipe on each frame → face landmarks
  • Call FeatureExtractor.update() → CV features
  • Call HybridEngine.decide()    → AI risk decision
  • Draw rich HUD overlays        → annotated frame
  • Maintain session stats and export full state dict

State dict fields (returned by process_frame and /api/status)
  Core CV (from detector):
    face_detected, ear, mar, eye_state, status, drowsy, yawning,
    closed_frames, session_seconds, drowsy_events, yawn_events

  Extended CV features (from features.py):
    pitch, yaw, roll, blink_count, blink_rate, long_blink_count,
    microsleep_count, last_closure_dur, yawn_count, last_yawn_dur,
    face_visibility, face_stability, detection_confidence,
    attention_index, fatigue_index, fps

  AI outputs (from ai_model.py):
    rule_risk, rule_confidence, model_risk, model_confidence,
    model_probs, model_available, temporal_smoothed, tf_available,
    final_risk, final_confidence, risk_color, risk_index

  Agent (from ai_model.SafetyAgent, updated every 5s):
    agent_observations, agent_recommendations, driver_grade,
    grade_reason, grade_score, agent_summary
"""

from __future__ import annotations

import cv2
import math
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from utils    import eye_aspect_ratio, mouth_aspect_ratio, smooth_value
from features import FeatureExtractor
from ai_model import HybridEngine, SafetyAgent

# ── Model path ────────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_BACKEND_DIR, 'face_landmarker.task')
MODEL_URL    = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
)


def _ensure_model():
    """Download the face landmarker model if not already present."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print('[SentinelAI] Downloading face_landmarker model (~6 MB)...')
    try:
        req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp, \
             open(MODEL_PATH, 'wb') as f:
            f.write(resp.read())
        print(f'[SentinelAI] Model saved → {MODEL_PATH}')
    except Exception as e:
        raise RuntimeError(
            f'Failed to download face landmarker model: {e}\n'
            f'Manual download URL: {MODEL_URL}\n'
            f'Save to: {MODEL_PATH}'
        )


# ── MediaPipe landmark indices (478-point mesh) ───────────────────────────────
# Left eye:  outer(33), top-inner(160), top-outer(158),
#            inner(133), bot-inner(153), bot-outer(144)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]

# Right eye: outer(362), top-inner(385), top-outer(387),
#            inner(263), bot-inner(380), bot-outer(373)
RIGHT_EYE = [362, 385, 387, 263, 380, 373]

# Mouth: top(13), bottom(14), left-corner(78), right-corner(308)
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308

# Additional mouth points for fuller contour
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
               375, 321, 405, 314, 17, 84, 181, 91, 146]

# Iris centres (with refine_landmarks=True)
LEFT_IRIS  = 468
RIGHT_IRIS = 473

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.55
DROWSY_FRAMES = 20
YAWN_FRAMES   = 15

# Agent recording interval (seconds)
AGENT_RECORD_INTERVAL = 3.0


class DrowsinessDetector:
    def __init__(self):
        _ensure_model()

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)

        # Core CV smoothing (keep existing — do not remove)
        self._ear_history: list[float] = []
        self._mar_history: list[float] = []
        self._closed_frames = 0
        self._yawn_frames   = 0

        # Session stats (keep existing)
        self.session_start = time.time()
        self.drowsy_events = 0
        self.yawn_events   = 0
        self._was_drowsy   = False
        self._was_yawning  = False
        self.frame_count   = 0

        # ── NEW: extended AI components ───────────────────────────────────────
        self._features_extractor = FeatureExtractor()
        self._hybrid_engine      = HybridEngine()
        self._safety_agent       = SafetyAgent()
        self._last_agent_record  = 0.0

        # Processing time tracking
        self._proc_times: list[float] = []

        # Current exported state
        self.state = self._default_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """
        Process one BGR frame.
        Returns (annotated_frame, state_dict).
        """
        t0 = time.time()
        h, w = frame.shape[:2]
        self.frame_count += 1

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        state = self._default_state()

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            def px(idx):
                return (int(lms[idx].x * w), int(lms[idx].y * h))

            left_pts  = [px(i) for i in LEFT_EYE]
            right_pts = [px(i) for i in RIGHT_EYE]
            mouth_pts = [px(MOUTH_TOP), px(MOUTH_BOTTOM),
                         px(MOUTH_LEFT), px(MOUTH_RIGHT)]

            # ── EAR (keep original logic) ──────────────────────────────────
            left_ear  = eye_aspect_ratio(left_pts)
            right_ear = eye_aspect_ratio(right_pts)
            raw_ear   = (left_ear + right_ear) / 2.0
            ear, self._ear_history = smooth_value(self._ear_history, raw_ear)

            # ── MAR (keep original logic) ──────────────────────────────────
            raw_mar = mouth_aspect_ratio(mouth_pts)
            mar, self._mar_history = smooth_value(self._mar_history, raw_mar)

            # ── Frame counters (keep original logic) ───────────────────────
            eyes_closed = ear < EAR_THRESHOLD
            self._closed_frames = (self._closed_frames + 1) if eyes_closed else 0

            is_yawning  = mar > MAR_THRESHOLD
            self._yawn_frames = (self._yawn_frames + 1) if is_yawning else 0

            drowsy  = self._closed_frames >= DROWSY_FRAMES
            yawning = self._yawn_frames   >= YAWN_FRAMES

            if drowsy  and not self._was_drowsy:  self.drowsy_events += 1
            if yawning and not self._was_yawning: self.yawn_events   += 1
            self._was_drowsy  = drowsy
            self._was_yawning = yawning

            cv_status = "DROWSY" if drowsy else ("YAWNING" if yawning else "AWAKE")

            # ── Feature engineering (NEW) ──────────────────────────────────
            feats = self._features_extractor.update(
                lms, h, w, ear, mar, eyes_closed, is_yawning
            )

            # ── AI decision (NEW) ──────────────────────────────────────────
            ai = self._hybrid_engine.decide(feats, cv_status)

            # ── Agent recording (NEW, every AGENT_RECORD_INTERVAL seconds) ─
            now = time.time()
            if now - self._last_agent_record >= AGENT_RECORD_INTERVAL:
                self._safety_agent.record(now, feats, ai)
                self._last_agent_record = now

            # ── Build state dict ───────────────────────────────────────────
            state.update({
                # ---- original fields (unchanged) ----
                'face_detected': True,
                'ear':           ear,
                'mar':           mar,
                'eye_state':     'Closed' if eyes_closed else 'Open',
                'status':        cv_status,
                'drowsy':        drowsy,
                'yawning':       yawning,
                'closed_frames': self._closed_frames,
                # ---- extended CV features ----
                'pitch':               feats['pitch'],
                'yaw':                 feats['yaw'],
                'roll':                feats['roll'],
                'blink_count':         feats['blink_count'],
                'blink_rate':          feats['blink_rate'],
                'long_blink_count':    feats['long_blink_count'],
                'microsleep_count':    feats['microsleep_count'],
                'last_closure_dur':    feats['last_closure_dur'],
                'yawn_count':          feats['yawn_count'],
                'last_yawn_dur':       feats['last_yawn_dur'],
                'face_visibility':     feats['face_visibility'],
                'face_stability':      feats['face_stability'],
                'detection_confidence':feats['detection_confidence'],
                'attention_index':     feats['attention_index'],
                'fatigue_index':       feats['fatigue_index'],
                'fps':                 feats['fps'],
                # ---- AI outputs ----
                'rule_risk':           ai['rule_risk'],
                'rule_confidence':     ai['rule_confidence'],
                'model_risk':          ai['model_risk'],
                'model_confidence':    ai['model_confidence'],
                'model_probs':         ai['model_probs'],
                'model_available':     ai['model_available'],
                'temporal_smoothed':   ai['temporal_smoothed'],
                'tf_available':        ai['tf_available'],
                'final_risk':          ai['final_risk'],
                'final_confidence':    ai['final_confidence'],
                'risk_color':          ai['risk_color'],
                'risk_index':          ai['risk_index'],
            })

            # Draw HUD overlays
            frame = self._draw_overlays(
                frame, lms, h, w,
                left_pts, right_pts, mouth_pts,
                ear, mar, feats, ai, cv_status
            )

        else:
            state['status'] = 'NO FACE'
            self._closed_frames = 0
            self._yawn_frames   = 0

        # ── Session stats (keep original) ──────────────────────────────────
        state['session_seconds'] = int(time.time() - self.session_start)
        state['drowsy_events']   = self.drowsy_events
        state['yawn_events']     = self.yawn_events

        # Processing time (NEW)
        proc_ms = (time.time() - t0) * 1000
        self._proc_times.append(proc_ms)
        if len(self._proc_times) > 30:
            self._proc_times.pop(0)
        state['proc_ms'] = round(float(sum(self._proc_times) / len(self._proc_times)), 1)

        self.state = state
        return frame, state

    def get_agent_report(self) -> dict:
        """Generate safety agent analysis report."""
        return self._safety_agent.analyse(
            session_seconds=self.state.get('session_seconds', 0),
            drowsy_events=self.drowsy_events,
            yawn_events=self.yawn_events,
            features=self._features_extractor.features,
            ai_result=self.state,
        )

    def reset_session(self):
        self.session_start = time.time()
        self.drowsy_events = 0
        self.yawn_events   = 0
        self.frame_count   = 0
        self._closed_frames = 0
        self._yawn_frames   = 0
        self._ear_history   = []
        self._mar_history   = []
        self._was_drowsy    = False
        self._was_yawning   = False
        self._features_extractor.reset()
        self.state = self._default_state()

    def release(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_overlays(self, frame, lms, h, w,
                       left_pts, right_pts, mouth_pts,
                       ear, mar, feats, ai, cv_status):
        """Rich HUD overlay: eyes, mouth contour, head pose axis, metrics."""

        # ── Color scheme by risk ──────────────────────────────────────────────
        COLORS = {
            'NORMAL': (80,  220, 100),   # green
            'LOW':    (80,  220, 100),
            'MEDIUM': (30,  190, 255),   # amber
            'HIGH':   (60,  60,  240),   # red
        }
        final_risk = ai.get('final_risk', 'NORMAL')
        color = COLORS.get(final_risk, COLORS['NORMAL'])

        # Override to red if CV says DROWSY
        if cv_status == 'DROWSY':
            color = COLORS['HIGH']
        elif cv_status == 'YAWNING':
            color = COLORS['MEDIUM']

        # ── Eye contours ──────────────────────────────────────────────────────
        for pts in (left_pts, right_pts):
            poly = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)
            # Fill with translucent tint
            cv2.fillPoly(
                frame,
                [poly],
                tuple(max(0, int(c * 0.15)) for c in color)
            )

        # ── Iris dots (if landmarks available) ────────────────────────────────
        try:
            for iris_idx in [LEFT_IRIS, RIGHT_IRIS]:
                ix = int(lms[iris_idx].x * w)
                iy = int(lms[iris_idx].y * h)
                cv2.circle(frame, (ix, iy), 3, color, -1, cv2.LINE_AA)
        except (IndexError, AttributeError):
            pass

        # ── Mouth contour ─────────────────────────────────────────────────────
        top, bot, mleft, mright = mouth_pts
        cv2.line(frame, mleft, mright, color, 1, cv2.LINE_AA)
        cv2.line(frame, top,   bot,    color, 1, cv2.LINE_AA)

        # Draw fuller mouth outline
        mouth_outer_pts = np.array(
            [(int(lms[i].x * w), int(lms[i].y * h)) for i in MOUTH_OUTER],
            dtype=np.int32
        )
        cv2.polylines(frame, [mouth_outer_pts], True, color, 1, cv2.LINE_AA)

        # ── Head pose axis ────────────────────────────────────────────────────
        self._draw_head_pose_axis(frame, lms, h, w, feats)

        # ── Face bounding box ─────────────────────────────────────────────────
        xs = [int(lms[i].x * w) for i in [234, 454, 10, 152]]
        ys = [int(lms[i].y * h) for i in [234, 454, 10, 152]]
        x1, y1 = max(0, min(xs) - 10), max(0, min(ys) - 10)
        x2, y2 = min(w, max(xs) + 10), min(h, max(ys) + 10)
        # Draw corner brackets instead of full rectangle for cleaner HUD look
        blen = 14
        bc   = color
        for (px, py, dx, dy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(frame, (px, py), (px + dx*blen, py),           bc, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px,           py + dy*blen), bc, 2, cv2.LINE_AA)

        # ── Status badge (top-left) ───────────────────────────────────────────
        risk_label = f' {final_risk} RISK '
        if cv_status == 'DROWSY':   risk_label = ' DROWSY '
        elif cv_status == 'YAWNING': risk_label = ' YAWNING '
        elif cv_status == 'AWAKE' and final_risk == 'NORMAL': risk_label = ' AWAKE '

        (tw, th), _ = cv2.getTextSize(risk_label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.rectangle(frame, (8, 8), (14 + tw, 16 + th), color, -1)
        cv2.putText(frame, risk_label, (10, 11 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (10, 14, 26), 2, cv2.LINE_AA)

        # ── Bottom HUD strip ──────────────────────────────────────────────────
        fps     = feats.get('fps', 0)
        conf    = ai.get('final_confidence', 0)
        att     = feats.get('attention_index', 0)
        fatigue = feats.get('fatigue_index', 0)

        hud_items = [
            f'EAR {ear:.3f}',
            f'MAR {mar:.3f}',
            f'FPS {fps:.0f}',
            f'ATT {att:.0%}',
            f'FAT {fatigue:.0%}',
            f'CONF {conf:.0%}',
        ]
        strip_y = h - 14
        x_cursor = 8
        for item in hud_items:
            (iw, _), _ = cv2.getTextSize(item, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(frame, item, (x_cursor, strip_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
            x_cursor += iw + 14

        # ── Head pose angles (top-right) ──────────────────────────────────────
        pitch = feats.get('pitch', 0)
        yaw_v = feats.get('yaw', 0)
        roll  = feats.get('roll', 0)
        pose_lines = [
            f'P {pitch:+.0f}',
            f'Y {yaw_v:+.0f}',
            f'R {roll:+.0f}',
        ]
        for i, line in enumerate(pose_lines):
            cv2.putText(frame, line, (w - 70, 20 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        # ── Blink / microsleep indicators ─────────────────────────────────────
        blinks = feats.get('blink_count', 0)
        micro  = feats.get('microsleep_count', 0)
        cv2.putText(frame, f'BLK {blinks}  MIC {micro}',
                    (w - 100, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

        return frame

    def _draw_head_pose_axis(self, frame, lms, h, w, feats):
        """
        Draw 3-axis head orientation arrows from nose tip.
        Red=X(roll), Green=Y(pitch), Blue=Z(yaw/depth).
        """
        pitch = feats.get('pitch', 0)
        yaw   = feats.get('yaw',   0)
        roll  = feats.get('roll',  0)

        nose_x = int(lms[1].x * w)
        nose_y = int(lms[1].y * h)
        L = 40  # arrow length

        # Convert degrees to radians
        p = math.radians(pitch)
        y = math.radians(yaw)
        r = math.radians(roll)

        # Project simplified axis vectors
        # X axis (roll) → red
        end_x = (int(nose_x + L * math.cos(r)), int(nose_y + L * math.sin(r)))
        # Y axis (pitch) → green
        end_y = (int(nose_x - L * math.sin(p)), int(nose_y - L * math.cos(p)))
        # Z axis (yaw/depth) → blue (project onto image plane)
        end_z = (int(nose_x + L * math.sin(y)), int(nose_y - L * 0.3))

        cv2.arrowedLine(frame, (nose_x, nose_y), end_x,
                        (60,  60, 220), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, (nose_x, nose_y), end_y,
                        (60, 200,  60), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, (nose_x, nose_y), end_z,
                        (220, 100,  60), 2, cv2.LINE_AA, tipLength=0.25)

    # ── Default state ─────────────────────────────────────────────────────────

    def _default_state(self) -> dict:
        return {
            # ---- original ----
            'face_detected':    False,
            'ear':              0.0,
            'mar':              0.0,
            'eye_state':        'N/A',
            'status':           'NO FACE',
            'drowsy':           False,
            'yawning':          False,
            'closed_frames':    0,
            'session_seconds':  0,
            'drowsy_events':    0,
            'yawn_events':      0,
            # ---- extended CV ----
            'pitch':              0.0,
            'yaw':                0.0,
            'roll':               0.0,
            'blink_count':        0,
            'blink_rate':         0.0,
            'long_blink_count':   0,
            'microsleep_count':   0,
            'last_closure_dur':   0.0,
            'yawn_count':         0,
            'last_yawn_dur':      0.0,
            'face_visibility':    0.0,
            'face_stability':     0.0,
            'detection_confidence': 0.0,
            'attention_index':    0.0,
            'fatigue_index':      0.0,
            'fps':                0.0,
            'proc_ms':            0.0,
            # ---- AI ----
            'rule_risk':          'NORMAL',
            'rule_confidence':    0.0,
            'model_risk':         'NORMAL',
            'model_confidence':   0.0,
            'model_probs':        [0.25, 0.25, 0.25, 0.25],
            'model_available':    False,
            'temporal_smoothed':  0.0,
            'tf_available':       False,
            'final_risk':         'NORMAL',
            'final_confidence':   0.0,
            'risk_color':         '#22c55e',
            'risk_index':         0,
        }
