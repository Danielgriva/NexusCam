"""
NEXUS FRAME — Holographic Hand Tracker v3.0
============================================
Native desktop app using OpenCV + MediaPipe for real-time
hand tracking with GPU-accelerated rendering.

Features:
  - 4 visual effect modes: Hologram, Neon, Matrix, Plasma
  - Proper quadrilateral frame from fingertip positions
  - Kalman-filtered landmark smoothing for precision
  - Depth-aware effects (palm size → distance estimation)
  - Timer capture (OFF / 3s / 5s / 10s)
  - Video recording with effects baked in
  - Screenshot capture
  - No hand skeleton drawn on screen

Controls:
  1-4  : Switch effect mode (Hologram / Neon / Matrix / Plasma)
  T    : Cycle timer (OFF → 3s → 5s → 10s → OFF)
  SPACE: Take photo (with timer if set)
  R    : Start / Stop video recording
  Q/ESC: Quit
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# KALMAN FILTER for smooth landmark tracking
# ─────────────────────────────────────────────
class KalmanPoint:
    """2D Kalman filter for a single landmark point."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # state=[x,y,vx,vy], measure=[x,y]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.initialized = False

    def update(self, x: float, y: float):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return float(prediction[0][0]), float(prediction[1][0])


class HandSmoother:
    """Smooths all 21 landmarks of one hand using Kalman filters."""
    def __init__(self):
        self.filters = [KalmanPoint() for _ in range(21)]

    def smooth(self, landmarks, w, h):
        """Returns smoothed pixel coordinates for all 21 landmarks."""
        result = []
        for i, lm in enumerate(landmarks):
            sx, sy = self.filters[i].update(lm.x * w, lm.y * h)
            result.append((sx, sy))
        return result


# ─────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────
@dataclass
class AppState:
    mode: str = "hologram"           # hologram, neon, matrix, plasma
    mode_names: list = field(default_factory=lambda: ["hologram", "neon", "matrix", "plasma"])
    timer_options: list = field(default_factory=lambda: [0, 3, 5, 10])
    timer_index: int = 0
    countdown_active: bool = False
    countdown_remaining: int = 0
    countdown_start: float = 0.0
    recording: bool = False
    rec_start_time: float = 0.0
    video_writer: Optional[cv2.VideoWriter] = None
    frame_count: int = 0
    fps: float = 0.0
    fps_timer: float = 0.0
    fps_count: int = 0
    start_time: float = field(default_factory=time.time)
    flash_until: float = 0.0
    # Hand smoothers
    smoothers: dict = field(default_factory=dict)
    # Matrix column state
    matrix_cols: list = field(default_factory=list)

    @property
    def timer_delay(self):
        return self.timer_options[self.timer_index]

    @property
    def timer_label(self):
        v = self.timer_delay
        return "OFF" if v == 0 else f"{v}s"


# ─────────────────────────────────────────────
# COLOR PALETTES
# ─────────────────────────────────────────────
# BGR format for OpenCV
COLORS = {
    "hologram": {
        "primary": (200, 255, 0),     # cyan in BGR
        "secondary": (255, 97, 123),  # purple
        "fill": (200, 255, 0),
        "text": (200, 255, 0),
    },
    "neon": {
        "primary": (216, 97, 255),    # pink
        "secondary": (255, 97, 123),  # purple
        "fill": (216, 97, 255),
        "text": (216, 97, 255),
    },
    "matrix": {
        "primary": (100, 255, 0),     # green
        "secondary": (80, 200, 0),
        "fill": (100, 255, 0),
        "text": (100, 255, 0),
    },
    "plasma": {
        "primary": (66, 140, 255),    # orange
        "secondary": (216, 97, 255),
        "fill": (255, 97, 123),
        "text": (66, 140, 255),
    },
}


# ─────────────────────────────────────────────
# UTILITY DRAWING FUNCTIONS
# ─────────────────────────────────────────────
def overlay_transparent(frame, overlay, alpha):
    """Blend overlay onto frame with given alpha."""
    cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)


def draw_quad_filled(frame, pts, color_bgr, alpha=0.15):
    """Draw a filled quadrilateral with transparency."""
    overlay = frame.copy()
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], color_bgr)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_quad_border(frame, pts, color_bgr, thickness=2, alpha=1.0):
    """Draw quad border lines."""
    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    if alpha < 1.0:
        overlay = frame.copy()
        cv2.polylines(overlay, [pts_arr], True, color_bgr, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        cv2.polylines(frame, [pts_arr], True, color_bgr, thickness, cv2.LINE_AA)


def draw_corner_brackets(frame, pts, color_bgr, alpha=0.8):
    """Draw L-shaped brackets at each corner of the quad."""
    pts_list = list(pts)
    for i in range(4):
        curr = np.array(pts_list[i], dtype=np.float64)
        prev = np.array(pts_list[(i + 3) % 4], dtype=np.float64)
        nxt = np.array(pts_list[(i + 1) % 4], dtype=np.float64)

        # Direction vectors
        to_prev = prev - curr
        to_next = nxt - curr
        len_prev = np.linalg.norm(to_prev)
        len_next = np.linalg.norm(to_next)
        if len_prev < 1 or len_next < 1:
            continue
        to_prev = to_prev / len_prev
        to_next = to_next / len_next

        bracket_len = min(25, len_prev * 0.15, len_next * 0.15)

        p1 = (int(curr[0] + to_prev[0] * bracket_len),
              int(curr[1] + to_prev[1] * bracket_len))
        p2 = (int(curr[0]), int(curr[1]))
        p3 = (int(curr[0] + to_next[0] * bracket_len),
              int(curr[1] + to_next[1] * bracket_len))

        cv2.line(frame, p1, p2, color_bgr, 2, cv2.LINE_AA)
        cv2.line(frame, p2, p3, color_bgr, 2, cv2.LINE_AA)

        # Corner dot
        cv2.circle(frame, p2, 4, color_bgr, -1, cv2.LINE_AA)
        # Outer glow ring
        cv2.circle(frame, p2, 10, color_bgr, 1, cv2.LINE_AA)


def draw_energy_dots(frame, pts, color_bgr, anim_frame):
    """Animated dots moving along the quad edges."""
    pts_list = list(pts)
    num_dots = 8
    t = anim_frame * 0.05

    for i in range(num_dots):
        progress = ((t + i / num_dots) % 1.0)
        total = progress * 4
        edge = int(total) % 4
        frac = total - int(total)
        p1 = np.array(pts_list[edge], dtype=np.float64)
        p2 = np.array(pts_list[(edge + 1) % 4], dtype=np.float64)
        pos = p1 + (p2 - p1) * frac
        px, py = int(pos[0]), int(pos[1])

        alpha_mod = 0.5 + 0.4 * math.sin(t * 5 + i)
        c = tuple(int(v * alpha_mod) for v in color_bgr)
        cv2.circle(frame, (px, py), 2, c, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# EFFECT RENDERERS
# ─────────────────────────────────────────────
def render_hologram(frame, pts, state):
    """Holographic scan-line effect with HUD text."""
    t = state.frame_count * 0.03
    colors = COLORS["hologram"]

    # 1. Filled quad
    alpha = 0.12 + 0.04 * math.sin(t)
    draw_quad_filled(frame, pts, colors["primary"], alpha)

    # 2. Clip region for internal effects
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

    # Bounding rect for efficiency
    x, y, bw, bh = cv2.boundingRect(np.array(pts, np.int32))

    # 3. Scan lines
    overlay = np.zeros_like(frame)
    for ly in range(y, y + bh, 4):
        cv2.line(overlay, (x, ly), (x + bw, ly), colors["primary"], 1)
    overlay_masked = cv2.bitwise_and(overlay, overlay, mask=mask)
    cv2.addWeighted(overlay_masked, 0.06, frame, 1.0, 0, frame)

    # 4. Moving sweep line
    sweep_y = y + int((state.frame_count * 2) % max(1, bh))
    if y <= sweep_y <= y + bh:
        overlay2 = np.zeros_like(frame)
        cv2.line(overlay2, (x, sweep_y), (x + bw, sweep_y), colors["primary"], 2)
        overlay2_m = cv2.bitwise_and(overlay2, overlay2, mask=mask)
        cv2.addWeighted(overlay2_m, 0.25, frame, 1.0, 0, frame)

    # 5. HUD text
    info_lines = [
        "SYS://NEXUS_FRAME",
        f"DIM: {bw}x{bh}",
        "MODE: HOLOGRAPHIC",
        "STATUS: TRACKING",
    ]
    for i, line in enumerate(info_lines):
        # Check if text position is inside the quad
        tx, ty = x + 10, y + 18 + i * 16
        if mask[min(ty, frame.shape[0]-1), min(tx, frame.shape[1]-1)] > 0:
            cv2.putText(frame, line, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (*colors["text"],), 1, cv2.LINE_AA)

    # 6. Border
    border_alpha = 0.4 + 0.15 * math.sin(t)
    draw_quad_border(frame, pts, colors["primary"], 2, border_alpha)


def render_neon(frame, pts, state):
    """Multi-layer neon glow border with crosshair."""
    t = state.frame_count * 0.04
    colors = COLORS["neon"]

    # Fill
    alpha = 0.08 + 0.04 * math.sin(t)
    draw_quad_filled(frame, pts, colors["secondary"], alpha)

    # Multi-layer borders (thick → thin for fake glow)
    for thickness, a in [(8, 0.15), (4, 0.35), (2, 0.7), (1, 0.9)]:
        c = tuple(int(v * a) for v in colors["primary"])
        draw_quad_border(frame, pts, c, thickness)

    # Crosshair at center
    cx = int(sum(p[0] for p in pts) / 4)
    cy = int(sum(p[1] for p in pts) / 4)
    cs = int(18 + 6 * math.sin(t * 2))

    cross_c = tuple(int(v * 0.4) for v in colors["primary"])
    cv2.line(frame, (cx - cs, cy), (cx + cs, cy), cross_c, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - cs), (cx, cy + cs), cross_c, 1, cv2.LINE_AA)

    # Rotating targeting arcs
    angle_deg = int((t * 80) % 360)
    cv2.ellipse(frame, (cx, cy), (cs * 2, cs * 2), 0,
                angle_deg, angle_deg + 70, colors["primary"], 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (cs * 2, cs * 2), 0,
                angle_deg + 180, angle_deg + 250, colors["primary"], 2, cv2.LINE_AA)


def render_matrix(frame, pts, state):
    """Matrix rain inside the quad."""
    colors = COLORS["matrix"]

    # Dark fill
    draw_quad_filled(frame, pts, (0, 15, 0), 0.2)

    # Mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    x, y, bw, bh = cv2.boundingRect(np.array(pts, np.int32))

    # Initialize matrix columns if needed
    col_width = 16
    num_cols = max(1, bw // col_width)
    if len(state.matrix_cols) != num_cols:
        state.matrix_cols = [
            {"y": np.random.randint(0, max(1, bh)), "speed": 2 + np.random.randint(0, 6)}
            for _ in range(num_cols)
        ]

    chars = list("アイウエオカキクケコ0123456789ABCDEF")
    overlay = np.zeros_like(frame)

    for i, col in enumerate(state.matrix_cols):
        cx_pos = x + i * col_width + col_width // 2
        col["y"] = (col["y"] + col["speed"]) % max(1, bh + 100)
        head_y = y + int(col["y"])

        if y <= head_y <= y + bh:
            # Head character (bright)
            ch = chars[np.random.randint(0, len(chars))]
            cv2.putText(overlay, ch, (cx_pos, head_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 255, 120), 1, cv2.LINE_AA)

        # Trail
        for t_idx in range(1, 8):
            trail_y = head_y - t_idx * 14
            if trail_y < y or trail_y > y + bh:
                continue
            ch = chars[np.random.randint(0, len(chars))]
            a = max(0, 0.5 - t_idx * 0.06)
            c = tuple(int(v * a) for v in colors["primary"])
            cv2.putText(overlay, ch, (cx_pos, trail_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)

    overlay_m = cv2.bitwise_and(overlay, overlay, mask=mask)
    cv2.addWeighted(overlay_m, 0.9, frame, 1.0, 0, frame)

    # Border
    draw_quad_border(frame, pts, colors["primary"], 2)
    # Thick glow border
    draw_quad_border(frame, pts, tuple(int(v * 0.3) for v in colors["primary"]), 6)


def render_plasma(frame, pts, state):
    """Plasma blobs with rainbow cycling border."""
    t = state.frame_count * 0.025
    cx = int(sum(p[0] for p in pts) / 4)
    cy = int(sum(p[1] for p in pts) / 4)
    x, y, bw, bh = cv2.boundingRect(np.array(pts, np.int32))

    # Mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

    # Plasma blobs
    overlay = np.zeros_like(frame)
    blob_colors = [
        (216, 97, 255),
        (200, 255, 0),
        (66, 140, 255),
    ]
    for i, bc in enumerate(blob_colors):
        angle = t * (0.8 + i * 0.3) + i * 2.09
        bx = int(cx + math.cos(angle) * bw * 0.25)
        by = int(cy + math.sin(angle) * bh * 0.25)
        r = int(max(bw, bh) * 0.4)

        # Draw a soft blob using a gaussian-like circle
        for ring in range(r, 0, -4):
            a = max(0, 0.08 * (1 - ring / r))
            c = tuple(int(v * a) for v in bc)
            cv2.circle(overlay, (bx, by), ring, c, 3, cv2.LINE_AA)

    overlay_m = cv2.bitwise_and(overlay, overlay, mask=mask)
    cv2.addWeighted(overlay_m, 0.6, frame, 1.0, 0, frame)

    # Rainbow border
    hue = int((state.frame_count * 2.5) % 180)
    hsv_color = np.uint8([[[hue, 255, 230]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    border_c = tuple(int(v) for v in bgr_color)

    draw_quad_border(frame, pts, border_c, 2)
    # Thick glow
    draw_quad_border(frame, pts, tuple(int(v * 0.3) for v in border_c), 6)


# Map mode names to render functions
RENDER_FNS = {
    "hologram": render_hologram,
    "neon": render_neon,
    "matrix": render_matrix,
    "plasma": render_plasma,
}


# ─────────────────────────────────────────────
# DEPTH ESTIMATION
# ─────────────────────────────────────────────
def estimate_depth(landmarks):
    """Estimate distance from camera using palm size.
    Returns scale 0.4..2.5 (bigger = closer)."""
    wrist = landmarks[0]
    mcp = landmarks[9]
    dx = wrist.x - mcp.x
    dy = wrist.y - mcp.y
    palm = math.sqrt(dx * dx + dy * dy)
    return max(0.4, min(2.5, palm / 0.2))


# ─────────────────────────────────────────────
# BUILD QUAD FROM TWO HANDS
# ─────────────────────────────────────────────
def build_quad(smooth1, smooth2, w, h):
    """Build quad from smoothed landmark pixel coords.
    Each hand: index tip (8) = top, thumb tip (4) = bottom.
    Returns [TL, TR, BR, BL] or None if too small."""
    h1_top = smooth1[8]   # index tip
    h1_bot = smooth1[4]   # thumb tip
    h2_top = smooth2[8]
    h2_bot = smooth2[4]

    # Figure left/right by x position
    if h1_top[0] < h2_top[0]:
        left_top, left_bot = h1_top, h1_bot
        right_top, right_bot = h2_top, h2_bot
    else:
        left_top, left_bot = h2_top, h2_bot
        right_top, right_bot = h1_top, h1_bot

    quad = [
        (int(left_top[0]), int(left_top[1])),
        (int(right_top[0]), int(right_top[1])),
        (int(right_bot[0]), int(right_bot[1])),
        (int(left_bot[0]), int(left_bot[1])),
    ]

    # Size check
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if max(xs) - min(xs) < 30 or max(ys) - min(ys) < 30:
        return None

    return quad


# ─────────────────────────────────────────────
# HUD DRAWING
# ─────────────────────────────────────────────
def draw_hud(frame, state):
    """Draw header bar, status indicators, and controls info."""
    h, w = frame.shape[:2]

    # Top bar background
    bar = np.zeros((44, w, 3), dtype=np.uint8)
    bar[:] = (18, 15, 10)
    cv2.addWeighted(bar, 0.7, frame[:44], 0.3, 0, frame[:44])

    # Title
    cv2.putText(frame, "NEXUS FRAME", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "v3.0", (170, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 60), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"{state.fps:.0f} FPS"
    cv2.putText(frame, fps_text, (w - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 0), 1, cv2.LINE_AA)

    # Elapsed time
    elapsed = int(time.time() - state.start_time)
    time_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
    cv2.putText(frame, time_str, (w - 200, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 100), 1, cv2.LINE_AA)

    # Bottom bar
    bot_y = h - 50
    bar_bot = np.zeros((50, w, 3), dtype=np.uint8)
    bar_bot[:] = (18, 15, 10)
    cv2.addWeighted(bar_bot, 0.7, frame[bot_y:], 0.3, 0, frame[bot_y:])

    # Mode indicator
    mode_text = f"MODE: {state.mode.upper()}"
    cv2.putText(frame, mode_text, (14, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[state.mode]["primary"], 1, cv2.LINE_AA)

    # Timer
    timer_text = f"TIMER: {state.timer_label}"
    cv2.putText(frame, timer_text, (220, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Controls
    controls = "[1-4] Mode  [T] Timer  [SPACE] Photo  [R] Record  [Q] Quit"
    cv2.putText(frame, controls, (380, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 100), 1, cv2.LINE_AA)

    # Recording indicator
    if state.recording:
        rec_elapsed = int(time.time() - state.rec_start_time)
        rec_str = f"REC {rec_elapsed // 60:02d}:{rec_elapsed % 60:02d}"
        # Blinking dot
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w // 2 - 50, 28), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, rec_str, (w // 2 - 38, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)

    # Corner decorations
    deco_c = (200, 255, 0)
    deco_len = 20
    # TL
    cv2.line(frame, (4, 48), (4 + deco_len, 48), deco_c, 1, cv2.LINE_AA)
    cv2.line(frame, (4, 48), (4, 48 + deco_len), deco_c, 1, cv2.LINE_AA)
    # TR
    cv2.line(frame, (w - 4, 48), (w - 4 - deco_len, 48), deco_c, 1, cv2.LINE_AA)
    cv2.line(frame, (w - 4, 48), (w - 4, 48 + deco_len), deco_c, 1, cv2.LINE_AA)
    # BL
    cv2.line(frame, (4, bot_y), (4 + deco_len, bot_y), deco_c, 1, cv2.LINE_AA)
    cv2.line(frame, (4, bot_y), (4, bot_y - deco_len), deco_c, 1, cv2.LINE_AA)
    # BR
    cv2.line(frame, (w - 4, bot_y), (w - 4 - deco_len, bot_y), deco_c, 1, cv2.LINE_AA)
    cv2.line(frame, (w - 4, bot_y), (w - 4, bot_y - deco_len), deco_c, 1, cv2.LINE_AA)


def draw_countdown(frame, remaining):
    """Draw big countdown number in center."""
    h, w = frame.shape[:2]
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    text = str(remaining)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5.0
    thickness = 8

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = (w - tw) // 2
    ty = (h + th) // 2

    # Glow layers
    cv2.putText(frame, text, (tx, ty), font, scale, (100, 200, 0), thickness + 8, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (200, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness - 4, cv2.LINE_AA)


def draw_flash(frame):
    """White flash overlay for screenshot."""
    cv2.addWeighted(np.ones_like(frame) * 255, 0.6, frame, 0.4, 0, frame)


# ─────────────────────────────────────────────
# CAPTURE FUNCTIONS
# ─────────────────────────────────────────────
def save_screenshot(frame):
    """Save current frame as PNG."""
    ts = int(time.time() * 1000)
    filename = f"nexus_frame_{ts}.png"
    cv2.imwrite(filename, frame)
    print(f"[PHOTO] Saved: {filename}")


def start_recording(state, frame):
    """Start video recording."""
    h, w = frame.shape[:2]
    ts = int(time.time())
    filename = f"nexus_frame_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    state.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
    state.recording = True
    state.rec_start_time = time.time()
    print(f"[REC] Recording started: {filename}")


def stop_recording(state):
    """Stop video recording."""
    if state.video_writer:
        state.video_writer.release()
        state.video_writer = None
    state.recording = False
    print("[REC] Recording stopped & saved.")


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  NEXUS FRAME — Holographic Hand Tracker v3.0")
    print("=" * 50)
    print()
    print("Controls:")
    print("  1-4   : Switch effect mode")
    print("  T     : Cycle timer (OFF/3s/5s/10s)")
    print("  SPACE : Capture photo")
    print("  R     : Start/Stop recording")
    print("  Q/ESC : Quit")
    print()

    # Initialize MediaPipe Hands via Tasks API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = os.path.abspath('hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"ERROR: Model '{model_path}' not found.")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {actual_w}x{actual_h}")

    state = AppState()

    # Window
    window_name = "NEXUS FRAME // Holographic Hand Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    start_ms = int(time.time() * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # FPS calculation
        state.fps_count += 1
        now = time.time()
        if now - state.fps_timer >= 1.0:
            state.fps = state.fps_count / (now - state.fps_timer)
            state.fps_count = 0
            state.fps_timer = now

        state.frame_count += 1

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Calculate timestamp for video mode
        frame_timestamp_ms = int(time.time() * 1000) - start_ms
        if frame_timestamp_ms < 0:
            frame_timestamp_ms = 0
            
        # Ensure monotonically increasing timestamps
        if hasattr(state, 'last_ts'):
            if frame_timestamp_ms <= state.last_ts:
                frame_timestamp_ms = state.last_ts + 1
        state.last_ts = frame_timestamp_ms

        results = detector.detect_for_video(mp_image, frame_timestamp_ms)

        frame_active = False

        if results.hand_landmarks and len(results.hand_landmarks) >= 2:
            # Get or create smoothers for each hand
            smoothed = []
            for idx, hand_lms in enumerate(results.hand_landmarks[:2]):
                key = f"hand_{idx}"
                if key not in state.smoothers:
                    state.smoothers[key] = HandSmoother()
                smooth = state.smoothers[key].smooth(hand_lms, w, h)
                smoothed.append(smooth)

            # Build quad
            quad = build_quad_from_smooth(smoothed[0], smoothed[1], w, h)
            if quad is not None:
                frame_active = True

                # Render the active effect
                render_fn = RENDER_FNS.get(state.mode, render_hologram)
                render_fn(frame, quad, state)

                # Corner brackets
                draw_corner_brackets(frame, quad, COLORS[state.mode]["primary"])

                # Energy dots
                draw_energy_dots(frame, quad, COLORS[state.mode]["primary"], state.frame_count)

        # HUD
        draw_hud(frame, state)

        # Hand count status
        n_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        status_color = (0, 255, 100) if frame_active else (100, 100, 150)
        status_text = f"{n_hands} HAND{'S' if n_hands != 1 else ''}" if n_hands > 0 else "NO HANDS"
        cv2.putText(frame, status_text, (14, h - 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1, cv2.LINE_AA)

        if frame_active:
            cv2.putText(frame, "FRAME: ACTIVE", (w - 160, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "FRAME: STANDBY", (w - 160, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 150), 1, cv2.LINE_AA)

        # Countdown handling
        if state.countdown_active:
            elapsed_cd = time.time() - state.countdown_start
            remaining = state.countdown_remaining - int(elapsed_cd)
            if remaining <= 0:
                state.countdown_active = False
                save_screenshot(frame)
                state.flash_until = time.time() + 0.2
            else:
                draw_countdown(frame, remaining)

        # Flash effect
        if time.time() < state.flash_until:
            draw_flash(frame)

        # Write to video if recording
        if state.recording and state.video_writer:
            state.video_writer.write(frame)

        # Display
        cv2.imshow(window_name, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('1'):
            state.mode = "hologram"
            state.matrix_cols = []
        elif key == ord('2'):
            state.mode = "neon"
            state.matrix_cols = []
        elif key == ord('3'):
            state.mode = "matrix"
        elif key == ord('4'):
            state.mode = "plasma"
            state.matrix_cols = []
        elif key == ord('t') or key == ord('T'):
            state.timer_index = (state.timer_index + 1) % len(state.timer_options)
            print(f"[TIMER] Set to: {state.timer_label}")
        elif key == ord(' '):  # SPACE
            if not state.countdown_active:
                if state.timer_delay > 0:
                    state.countdown_active = True
                    state.countdown_remaining = state.timer_delay
                    state.countdown_start = time.time()
                    print(f"[TIMER] Countdown: {state.timer_delay}s")
                else:
                    save_screenshot(frame)
                    state.flash_until = time.time() + 0.2
        elif key == ord('r') or key == ord('R'):
            if state.recording:
                stop_recording(state)
            else:
                start_recording(state, frame)

    # Cleanup
    if state.recording:
        stop_recording(state)
    cap.release()
    cv2.destroyAllWindows()
    
    # In Tasks API there's no hands.close(), we can close the detector
    detector.close()
    
    print("Goodbye!")


def build_quad_from_smooth(smooth1, smooth2, w, h):
    """Build quad from smoothed landmark pixel coords."""
    h1_top = smooth1[8]
    h1_bot = smooth1[4]
    h2_top = smooth2[8]
    h2_bot = smooth2[4]

    if h1_top[0] < h2_top[0]:
        left_top, left_bot = h1_top, h1_bot
        right_top, right_bot = h2_top, h2_bot
    else:
        left_top, left_bot = h2_top, h2_bot
        right_top, right_bot = h1_top, h1_bot

    quad = [
        (int(left_top[0]), int(left_top[1])),
        (int(right_top[0]), int(right_top[1])),
        (int(right_bot[0]), int(right_bot[1])),
        (int(left_bot[0]), int(left_bot[1])),
    ]

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if max(xs) - min(xs) < 30 or max(ys) - min(ys) < 30:
        return None
    return quad


if __name__ == "__main__":
    main()
