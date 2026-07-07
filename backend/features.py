"""
features.py — Computer Vision Feature Engineering Module
=========================================================
Extracts and tracks all CV features from MediaPipe face landmarks.

Features computed:
  • Blink detection, duration, rate, long-blink, microsleep
  • Yawn duration tracking
  • Head pose estimation (pitch, yaw, roll) via solvePnP
  • Face visibility and landmark stability
  • Attention index (head direction + eye openness)
  • Fatigue index (composite score)
  • Temporal rolling averages (window N frames)
  • FPS and processing time tracking

Mathematical notes
------------------
EAR (Eye Aspect Ratio) — Soukupová & Čech, 2016:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Open eye ≈ 0.30–0.40, Closed eye < 0.25

Head Pose via solvePnP:
    Maps 6 canonical 3D face points → 2D image landmarks.
    cv2.solvePnP returns rvec (rotation), tvec (translation).
    cv2.Rodrigues converts rvec → 3×3 rotation matrix R.
    Euler angles extracted from R:
        pitch = atan2(-R[2,0], sqrt(R[2,1]²+R[2,2]²))
        yaw   = atan2(R[1,0], R[0,0])
        roll  = atan2(R[2,1], R[2,2])

Attention Index:
    Combines head yaw/pitch (looking forward) + EAR (eyes open).
    0.0 = completely inattentive, 1.0 = fully attentive.

Fatigue Index:
    Weighted combination of:
      – Low EAR (35%)
      – Blink rate deviation from baseline (20%)
      – Long blink frequency (25%)
      – Head pose drooping (10%)
      – Yawn frequency (10%)
    0.0 = fully rested, 1.0 = severely fatigued.
"""

from __future__ import annotations

import time
import math
import numpy as np
import cv2
from collections import deque


# ── 3D reference face model (generic) ─────────────────────────────────────────
# These are approximate 3D coordinates (mm) of key facial landmarks in a
# canonical frontal face. Used with solvePnP to estimate head pose.
# Indices correspond to MediaPipe landmark numbers.
FACE_3D_MODEL = {
    # (landmark_index): (X, Y, Z) in mm
    1:   (0.0,    0.0,    0.0),     # nose tip (origin)
    152: (0.0,  -63.6,  -12.5),     # chin
    226: (-43.3,  32.7,  -26.0),    # left eye left corner
    446: ( 43.3,  32.7,  -26.0),    # right eye right corner
    57:  (-28.9, -28.9,  -24.1),    # left mouth corner
    287: ( 28.9, -28.9,  -24.1),    # right mouth corner
}
_3D_PTS = np.array(list(FACE_3D_MODEL.values()), dtype=np.float64)
_3D_IDX = list(FACE_3D_MODEL.keys())

# ── Blink detection thresholds ────────────────────────────────────────────────
BLINK_EAR_THRESH  = 0.22   # EAR below this = eye event
BLINK_MIN_FRAMES  = 2      # minimum consecutive frames to count as blink
BLINK_MAX_FRAMES  = 12     # above this = long blink (300ms @ 40fps)
MICROSLEEP_FRAMES = 50     # ~1.5s of closure = microsleep

# ── Temporal window sizes ──────────────────────────────────────────────────────
BLINK_RATE_WINDOW = 60     # seconds to measure blink rate over
TEMPORAL_N        = 10     # frames for temporal smoothing of AI outputs
STABILITY_N       = 8      # frames for landmark stability


class FeatureExtractor:
    """
    Stateful feature extractor. Call update(lms, h, w, dt) each frame.
    All features are stored in self.features dict.
    """

    def __init__(self):
        # ── Blink state machine ───────────────────────────────────────────────
        self._eye_closed      = False       # currently below threshold?
        self._close_frame_cnt = 0           # consecutive frames eye closed
        self._blink_timestamps: list[float] = []   # rolling list of blink times

        # Session blink counters
        self.blink_count    = 0
        self.long_blink_cnt = 0
        self.microsleep_cnt = 0

        # Yawn timing
        self._yawn_active    = False
        self._yawn_start     = 0.0
        self.yawn_durations: list[float] = []   # seconds each yawn lasted

        # Eye closure timing (for current closure)
        self._closure_start  = 0.0
        self.last_closure_dur = 0.0       # duration of most recent closure

        # Head pose history (for stability / temporal)
        self._pose_hist = deque(maxlen=STABILITY_N)

        # Landmark positions for stability tracking (nose tip)
        self._nose_hist = deque(maxlen=STABILITY_N)

        # Temporal confidence (deque of per-frame attention values)
        self._attention_hist = deque(maxlen=TEMPORAL_N)
        self._fatigue_hist   = deque(maxlen=TEMPORAL_N)

        # FPS tracking
        self._fps_times: deque = deque(maxlen=30)

        # Head pose (Euler angles, degrees)
        self.pitch = 0.0    # up/down tilt  (+ = looking down)
        self.yaw   = 0.0    # left/right    (+ = looking right)
        self.roll  = 0.0    # head tilt     (+ = tilted right)

        # Camera intrinsics (updated each frame from image size)
        self._camera_matrix = None
        self._dist_coeffs   = np.zeros((4, 1))

        # Exported feature dict (updated each frame)
        self.features: dict = self._blank_features()

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, lms, h: int, w: int, ear: float, mar: float,
               eye_closed: bool, is_yawning: bool) -> dict:
        """
        Main update called once per frame.

        Args:
            lms       : list of MediaPipe NormalizedLandmark objects
            h, w      : frame height, width
            ear       : smoothed Eye Aspect Ratio (from detector.py)
            mar       : smoothed Mouth Aspect Ratio
            eye_closed: bool flag from detector
            is_yawning: bool flag from detector

        Returns:
            dict of all features (also stored as self.features)
        """
        now = time.time()

        # ── FPS ───────────────────────────────────────────────────────────────
        self._fps_times.append(now)
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0

        # ── Head pose ─────────────────────────────────────────────────────────
        pitch, yaw, roll = self._estimate_head_pose(lms, h, w)
        self._pose_hist.append((pitch, yaw, roll))

        # ── Landmark stability (nose tip pixel movement) ───────────────────────
        nose_px = (lms[1].x * w, lms[1].y * h)
        self._nose_hist.append(nose_px)
        face_stability = self._compute_stability()

        # ── Blink analysis ────────────────────────────────────────────────────
        blink_just_completed, closure_dur = self._update_blink(
            ear, eye_closed, now
        )

        # Blink rate: blinks per minute in last BLINK_RATE_WINDOW seconds
        cutoff = now - BLINK_RATE_WINDOW
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]
        blink_rate = (len(self._blink_timestamps) / BLINK_RATE_WINDOW) * 60.0

        # ── Yawn timing ───────────────────────────────────────────────────────
        yawn_duration = self._update_yawn(is_yawning, now)

        # ── Face visibility (how centred & complete the face is) ──────────────
        face_visibility = self._compute_face_visibility(lms)

        # ── Attention index ───────────────────────────────────────────────────
        # Driver attention = head aligned forward + eyes open
        # Yaw/pitch penalty: ±30° = half attention, ±60° = zero
        yaw_factor   = max(0.0, 1.0 - abs(yaw)   / 60.0)
        pitch_factor = max(0.0, 1.0 - abs(pitch)  / 50.0)
        ear_factor   = min(1.0, max(0.0, (ear - 0.15) / 0.20))  # 0 at EAR=0.15, 1 at EAR=0.35
        attention_raw = 0.45 * yaw_factor + 0.35 * pitch_factor + 0.20 * ear_factor

        # ── Fatigue index ─────────────────────────────────────────────────────
        # Composite weighted score (0=rested, 1=fatigued)
        # 1. EAR fatigue: lower EAR → more fatigue
        ear_fatigue   = max(0.0, min(1.0, (0.32 - ear) / 0.18))

        # 2. Blink rate deviation: baseline ~15/min; very low or very high = fatigue
        baseline_rate = 15.0
        rate_dev = abs(blink_rate - baseline_rate) / baseline_rate
        blink_fatigue = min(1.0, rate_dev * 0.5)

        # 3. Long blink frequency in session
        long_blink_fatigue = min(1.0, self.long_blink_cnt / 5.0)

        # 4. Head drooping (pitch > 10° looking down)
        head_fatigue = min(1.0, max(0.0, (pitch - 10.0) / 30.0))

        # 5. Yawn frequency
        yawn_fatigue = min(1.0, len(self.yawn_durations) / 5.0)

        fatigue_raw = (
            0.35 * ear_fatigue
            + 0.20 * blink_fatigue
            + 0.25 * long_blink_fatigue
            + 0.10 * head_fatigue
            + 0.10 * yawn_fatigue
        )

        # ── Temporal smoothing ────────────────────────────────────────────────
        # Keep rolling history and report smoothed value to reduce noise
        self._attention_hist.append(attention_raw)
        self._fatigue_hist.append(fatigue_raw)

        attention = float(np.mean(self._attention_hist))
        fatigue   = float(np.mean(self._fatigue_hist))

        # ── Detection confidence ──────────────────────────────────────────────
        # How reliable are our estimates this frame?
        # Based on face visibility + landmark stability
        detection_confidence = (face_visibility * 0.6 + face_stability * 0.4)

        # ── Build & store feature dict ────────────────────────────────────────
        f = {
            # Core CV
            'ear':                  round(ear, 4),
            'mar':                  round(mar, 4),
            # Head pose (degrees)
            'pitch':                round(pitch, 1),
            'yaw':                  round(yaw, 1),
            'roll':                 round(roll, 1),
            # Blink features
            'blink_count':          self.blink_count,
            'blink_rate':           round(blink_rate, 1),        # blinks/min
            'long_blink_count':     self.long_blink_cnt,
            'microsleep_count':     self.microsleep_cnt,
            'last_closure_dur':     round(self.last_closure_dur, 3),  # seconds
            # Yawn features
            'yawn_count':           len(self.yawn_durations),
            'last_yawn_dur':        round(yawn_duration, 2),
            # Face quality
            'face_visibility':      round(face_visibility, 3),
            'face_stability':       round(face_stability, 3),
            'detection_confidence': round(detection_confidence, 3),
            # AI indices (0–1)
            'attention_index':      round(attention, 3),
            'fatigue_index':        round(fatigue, 3),
            # Performance
            'fps':                  round(fps, 1),
        }
        self.features = f
        return f

    def blank_frame_features(self) -> dict:
        """Return zero-filled features when no face is detected."""
        return self._blank_features()

    def reset(self):
        """Reset all session accumulators (called on session reset)."""
        self.__init__()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_head_pose(self, lms, h, w) -> tuple[float, float, float]:
        """
        Estimate head orientation using PnP solver.

        PnP (Perspective-n-Point) solves for the rotation and translation
        that best maps known 3D points to their 2D image projections.

        Steps:
          1. Extract 2D pixel coords of 6 known face landmarks.
          2. Match them to canonical 3D positions.
          3. cv2.solvePnP gives rotation vector rvec.
          4. cv2.Rodrigues converts rvec → 3×3 rotation matrix R.
          5. Extract Euler angles (pitch, yaw, roll) from R.

        Returns: (pitch, yaw, roll) in degrees
        """
        # Build camera intrinsic matrix (simple approximation)
        focal = w
        cx, cy = w / 2.0, h / 2.0
        cam = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1.0]
        ], dtype=np.float64)

        # 2D image points for the 6 reference landmarks
        pts_2d = np.array(
            [(lms[i].x * w, lms[i].y * h) for i in _3D_IDX],
            dtype=np.float64
        )

        try:
            ok, rvec, tvec = cv2.solvePnP(
                _3D_PTS, pts_2d, cam, self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return self.pitch, self.yaw, self.roll

            # Convert rotation vector → matrix
            R, _ = cv2.Rodrigues(rvec)

            # Decompose to Euler angles (radians → degrees)
            # Using ZYX convention (yaw-pitch-roll)
            pitch_rad = math.atan2(-R[2, 0],
                                   math.sqrt(R[2, 1]**2 + R[2, 2]**2))
            yaw_rad   = math.atan2(R[1, 0], R[0, 0])
            roll_rad  = math.atan2(R[2, 1], R[2, 2])

            self.pitch = math.degrees(pitch_rad)
            self.yaw   = math.degrees(yaw_rad)
            self.roll  = math.degrees(roll_rad)

        except cv2.error:
            pass   # keep previous values on solver failure

        return self.pitch, self.yaw, self.roll

    def _update_blink(self, ear: float, eye_closed: bool,
                      now: float) -> tuple[bool, float]:
        """
        Blink state machine.

        A blink is:  eye closes (EAR < thresh) for ≥ BLINK_MIN_FRAMES
                     then reopens.

        A long blink: closure duration > BLINK_MAX_FRAMES
        A microsleep: closure duration > MICROSLEEP_FRAMES

        Returns: (blink_completed_this_frame, closure_duration_seconds)
        """
        just_completed = False
        closure_dur    = 0.0

        if eye_closed:
            if not self._eye_closed:
                # Rising edge — start timing
                self._closure_start = now
            self._eye_closed      = True
            self._close_frame_cnt += 1
        else:
            if self._eye_closed:
                # Falling edge — eye reopened
                self._eye_closed = False
                frames = self._close_frame_cnt

                if frames >= BLINK_MIN_FRAMES:
                    just_completed = True
                    closure_dur    = now - self._closure_start
                    self.last_closure_dur = closure_dur
                    self.blink_count     += 1
                    self._blink_timestamps.append(now)

                    if frames >= MICROSLEEP_FRAMES:
                        self.microsleep_cnt += 1
                    elif frames >= BLINK_MAX_FRAMES:
                        self.long_blink_cnt += 1

                self._close_frame_cnt = 0

        return just_completed, closure_dur

    def _update_yawn(self, is_yawning: bool, now: float) -> float:
        """Track yawn duration. Returns duration of last completed yawn."""
        if is_yawning:
            if not self._yawn_active:
                self._yawn_active = True
                self._yawn_start  = now
        else:
            if self._yawn_active:
                self._yawn_active = False
                dur = now - self._yawn_start
                if dur > 0.3:   # filter sub-0.3s false positives
                    self.yawn_durations.append(dur)
                    return dur

        if self._yawn_active:
            return now - self._yawn_start
        return self.yawn_durations[-1] if self.yawn_durations else 0.0

    def _compute_stability(self) -> float:
        """
        Face stability: how much the nose landmark is moving.
        High variance = unstable/occluded face → low confidence.
        Returns 0.0 (unstable) – 1.0 (perfectly stable).
        """
        if len(self._nose_hist) < 2:
            return 1.0
        pts = np.array(self._nose_hist)
        # Standard deviation of (x, y) nose position over last N frames
        std = float(np.mean(np.std(pts, axis=0)))
        # Map: 0 px std → 1.0, 20 px std → 0.0
        return max(0.0, 1.0 - std / 20.0)

    def _compute_face_visibility(self, lms) -> float:
        """
        How fully is the face visible and centred?
        Check that key landmarks are not at the image edges (0 or 1).
        Returns 0.0 – 1.0.
        """
        key_indices = [1, 33, 263, 61, 291, 152, 10]  # nose, eyes, mouth, chin, head
        scores = []
        for i in key_indices:
            x, y = lms[i].x, lms[i].y
            # penalty for being near the edge (within 5% of border)
            margin = 0.05
            edge_score = min(x - margin, 1 - x - margin, y - margin, 1 - y - margin)
            scores.append(min(1.0, max(0.0, edge_score / 0.3)))
        return float(np.mean(scores))

    @staticmethod
    def _blank_features() -> dict:
        return {
            'ear': 0.0, 'mar': 0.0,
            'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
            'blink_count': 0, 'blink_rate': 0.0,
            'long_blink_count': 0, 'microsleep_count': 0,
            'last_closure_dur': 0.0,
            'yawn_count': 0, 'last_yawn_dur': 0.0,
            'face_visibility': 0.0, 'face_stability': 0.0,
            'detection_confidence': 0.0,
            'attention_index': 0.0, 'fatigue_index': 0.0,
            'fps': 0.0,
        }
