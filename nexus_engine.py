import cv2
import numpy as np
import time
import os
import sys
import json
import math
import threading
from collections import deque
import tkinter as tk
from tkinter import simpledialog
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import qrcode
from PIL import Image
from fx_pipeline import ParticleSystem, render_hologram, render_neon, render_matrix, render_plasma

RENDER_FNS = {
    "hologram": render_hologram,
    "neon": render_neon,
    "matrix": render_matrix,
    "plasma": render_plasma
}

# ==========================================
# CAMERA THREAD (Zero Lag I/O)
# ==========================================
class CameraThread:
    def __init__(self, src=0):
        if isinstance(src, str) and src.startswith("http"):
            self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        if not self.cap.isOpened():
            print("ERROR: Camera not found")
            self.running = False
            return
            
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame
                
    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
            
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()

# ==========================================
# KALMAN FILTER
# ==========================================
class KalmanPoint:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        self.kf.correct(measurement)
        pred = self.kf.predict()
        return float(pred[0][0]), float(pred[1][0])

class HandSmoother:
    def __init__(self):
        self.filters = [KalmanPoint() for _ in range(21)]

    def smooth(self, landmarks, w, h):
        return [(self.filters[i].update(lm.x * w, lm.y * h)) for i, lm in enumerate(landmarks)]

# ==========================================
# GESTURE ENGINE
# ==========================================
class GestureEngine:
    """Detects pinch, swipe, and peace sign gestures."""
    def __init__(self):
        self.enabled = False
        self.peace_start = 0
        self.peace_triggered = False
        self.last_hand_x = {}
        self.swipe_cooldown = 0
        
    def detect_pinch_distance(self, smooth1, smooth2):
        """Returns distance between the two index fingertips."""
        p1 = np.array(smooth1[8])
        p2 = np.array(smooth2[8])
        return np.linalg.norm(p1 - p2)
    
    def detect_peace_sign(self, landmarks, w, h):
        """Detects if hand is making a peace/V sign (index + middle extended, others curled)."""
        if not landmarks:
            return False
        # Index finger extended: tip(8) above pip(6)
        index_ext = landmarks[8].y < landmarks[6].y
        # Middle finger extended: tip(12) above pip(10) 
        middle_ext = landmarks[12].y < landmarks[10].y
        # Ring finger curled: tip(16) below pip(14)
        ring_curled = landmarks[16].y > landmarks[14].y
        # Pinky curled: tip(20) below pip(18)
        pinky_curled = landmarks[20].y > landmarks[18].y
        return index_ext and middle_ext and ring_curled and pinky_curled
    
    def detect_swipe(self, hand_idx, smooth_landmarks):
        """Detects horizontal swipe. Returns 'left', 'right', or None."""
        if time.time() < self.swipe_cooldown:
            return None
        wrist_x = smooth_landmarks[0][0]
        key = f"swipe_{hand_idx}"
        if key not in self.last_hand_x:
            self.last_hand_x[key] = wrist_x
            return None
        dx = wrist_x - self.last_hand_x[key]
        self.last_hand_x[key] = wrist_x
        if abs(dx) > 40:
            self.swipe_cooldown = time.time() + 1.0
            return "right" if dx > 0 else "left"
        return None

# ==========================================
# PRESET SYSTEM
# ==========================================
class PresetManager:
    """Manages 9 effect preset slots saved as JSON."""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.slots = {}
        self._load_all()
        
    def _path(self, slot):
        return os.path.join(self.save_dir, f"preset_{slot}.json")
        
    def _load_all(self):
        for i in range(1, 10):
            p = self._path(i)
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        self.slots[i] = json.load(f)
                except:
                    self.slots[i] = None
            else:
                self.slots[i] = None
                
    def save(self, slot, settings):
        self.slots[slot] = settings.copy()
        with open(self._path(slot), 'w') as f:
            json.dump(settings, f, indent=2)
            
    def load(self, slot):
        return self.slots.get(slot)
    
    def get_label(self, slot):
        if self.slots.get(slot):
            return self.slots[slot].get("name", f"Slot {slot}")
        return f"Empty {slot}"

# ==========================================
# COLOR PALETTE (Minimalist Professional)
# ==========================================
C_BG = (10, 10, 10)           # Deep black background
C_PANEL = (25, 25, 25)        # Clean dark gray panel
C_PANEL_HOVER = (40, 40, 40)  # Lighter gray on hover
C_ACCENT = (255, 255, 255)    # Pure white for outlines and accents
C_ACCENT2 = (200, 200, 200)   # Soft white
C_TEXT = (240, 240, 240)      # Crisp white text
C_TEXT_DIM = (120, 120, 120)  # Professional muted gray
C_DANGER = (50, 50, 220)      # Standard red for recording
C_SUCCESS = (100, 200, 100)   # Subtle green
C_SLIDER_TRACK = (40, 40, 40) # Subtle track
C_SLIDER_FILL = (255, 255, 255) # Clean white slider fill

# Minimalist color palette for user selection
NEON_PALETTE = [
    (255, 255, 255),  # Pure White
    (220, 220, 220),  # Soft White
    (200, 200, 200),  # Light Gray
    (150, 150, 150),  # Medium Gray
    (255, 200, 200),  # Subtle Warm
    (200, 200, 255),  # Subtle Cool
    (100, 255, 100),  # Clean Green
    (100, 200, 255),  # Clean Blue
    (255, 100, 100),  # Clean Red
]

# ==========================================
# UI HELPERS
# ==========================================
def draw_rounded_rect(frame, x, y, w, h, color, radius=6, fill=True, thickness=1):
    """Draw a rectangle with rounded corners."""
    if fill:
        # Fill center
        cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, -1)
        # Fill corners
        cv2.circle(frame, (x + radius, y + radius), radius, color, -1)
        cv2.circle(frame, (x + w - radius, y + radius), radius, color, -1)
        cv2.circle(frame, (x + radius, y + h - radius), radius, color, -1)
        cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, -1)
    else:
        # Draw outline only
        cv2.line(frame, (x + radius, y), (x + w - radius, y), color, thickness)
        cv2.line(frame, (x + radius, y + h), (x + w - radius, y + h), color, thickness)
        cv2.line(frame, (x, y + radius), (x, y + h - radius), color, thickness)
        cv2.line(frame, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
        cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_text_centered(frame, text, x, y, w, h, color, scale=0.4, thickness=1):
    """Draw text centered within a bounding box."""
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    tx = x + (w - ts[0]) // 2
    ty = y + (h + ts[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def generate_qr_opencv(url):
    """Generates a QR code and returns it as an OpenCV BGR image."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_slider(frame, x, y, w, value, min_val, max_val, label, color=C_SLIDER_FILL):
    """Draw a horizontal slider. Returns the slider hitbox."""
    h = 8
    label_h = 14
    # Label
    cv2.putText(frame, f"{label}: {value:.1f}" if isinstance(value, float) else f"{label}: {value}", 
                (x, y + label_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1, cv2.LINE_AA)
    # Track
    track_y = y + label_h + 4
    cv2.rectangle(frame, (x, track_y), (x + w, track_y + h), C_SLIDER_TRACK, -1)
    # Fill
    ratio = (value - min_val) / max(0.001, max_val - min_val)
    fill_w = int(w * ratio)
    cv2.rectangle(frame, (x, track_y), (x + fill_w, track_y + h), color, -1)
    # Knob
    knob_x = x + fill_w
    cv2.circle(frame, (knob_x, track_y + h // 2), 6, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (knob_x, track_y + h // 2), 6, (255, 255, 255), 1, cv2.LINE_AA)
    return (x, track_y - 4, w, h + 8)  # hitbox

# ==========================================
# APP CORE
# ==========================================
class NexusApp:
    def __init__(self):
        self.cam_index = 0
        self.cam = CameraThread(src=self.cam_index)
        if not self.cam.running:
            pass  # Will show NO CAMERA UI
            
        # Model path
        if hasattr(sys, '_MEIPASS'):
            model_path = os.path.join(sys._MEIPASS, 'hand_landmarker.task')
        else:
            model_path = os.path.abspath('hand_landmarker.task')
            
        if not os.path.exists(model_path):
            print(f"ERROR: Model '{model_path}' not found. Exiting.")
            if self.cam.running:
                self.cam.stop()
            return
            
        self.latest_result = None
        self.result_lock = threading.Lock()
        
        def result_callback(result, output_image, timestamp_ms):
            with self.result_lock:
                self.latest_result = result
                
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            result_callback=result_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # State
        self.smoothers = {}
        self.p_sys = ParticleSystem(max_particles=2000)
        self.anim_frame = 0
        self.last_quad = None
        
        # Performance
        self.fps = 0
        self.fps_timer = time.time()
        self.fps_count = 0
        self.start_ms = int(time.time() * 1000)
        
        # ── Customization Settings ──
        self.mode = "hologram"
        self.line_thickness = 2
        self.transparency = 0.15
        self.glow_intensity = 1.0
        self.animation_speed = 1.0
        self.particles_enabled = True
        self.particle_color = NEON_PALETTE[0]
        self.particle_color_idx = 0
        self.neon_color_idx = 0
        self.neon_colors = NEON_PALETTE
        self.holo_text = ""
        self.bg_mode = "none"  # none, solid, gradient
        self.bg_color = (0, 0, 0)
        
        # ── UI State ──
        self.mouse_x = 0
        self.mouse_y = 0
        self.show_settings = False
        self.settings_tab = 0  # 0=Effects, 1=Particles, 2=Capture, 3=Camera
        self.tab_names = ["EFFECTS", "PARTICLES", "CAPTURE", "CAMERA"]
        self.dragging_slider = None
        
        # ── Capture State ──
        self.timer_mode = 0
        self.action_start_time = 0
        self.pending_action = None
        self.is_recording = False
        self.video_writer = None
        self.media_dir = os.path.join(os.path.expanduser("~"), "Desktop", "NexusMedia")
        os.makedirs(self.media_dir, exist_ok=True)
        self.capture_image = False
        self.holo_only_capture = False
        self.flash_until = 0
        
        # ── X-Ray & Masking ──
        self.xray_enabled = False
        self.holo_masking_enabled = False
        
        # ── Gesture Engine ──
        self.gesture = GestureEngine()
        
        # ── Preset Manager ──
        self.presets = PresetManager(os.path.join(self.media_dir, "presets"))
        
        # ── Fullscreen ──
        self.is_fullscreen = False
        
        # ── UI Hitboxes (rebuilt every frame) ──
        self.ui_boxes = {}
        self.slider_boxes = {}
        
        self.run()
        
    def get_settings_dict(self):
        return {
            "mode": self.mode,
            "line_thickness": self.line_thickness,
            "transparency": self.transparency,
            "glow_intensity": self.glow_intensity,
            "animation_speed": self.animation_speed,
            "particles_enabled": self.particles_enabled,
            "particle_color_idx": self.particle_color_idx,
            "neon_color_idx": self.neon_color_idx,
            "holo_text": self.holo_text,
            "xray_enabled": self.xray_enabled,
            "holo_masking_enabled": self.holo_masking_enabled,
            "gesture_enabled": self.gesture.enabled,
        }
        
    def apply_settings_dict(self, d):
        if not d:
            return
        self.mode = d.get("mode", self.mode)
        self.line_thickness = d.get("line_thickness", self.line_thickness)
        self.transparency = d.get("transparency", self.transparency)
        self.glow_intensity = d.get("glow_intensity", self.glow_intensity)
        self.animation_speed = d.get("animation_speed", self.animation_speed)
        self.particles_enabled = d.get("particles_enabled", self.particles_enabled)
        self.particle_color_idx = d.get("particle_color_idx", self.particle_color_idx)
        self.particle_color = NEON_PALETTE[self.particle_color_idx % len(NEON_PALETTE)]
        self.neon_color_idx = d.get("neon_color_idx", self.neon_color_idx)
        self.holo_text = d.get("holo_text", self.holo_text)
        self.xray_enabled = d.get("xray_enabled", self.xray_enabled)
        self.holo_masking_enabled = d.get("holo_masking_enabled", self.holo_masking_enabled)
        self.gesture.enabled = d.get("gesture_enabled", self.gesture.enabled)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
            # Handle slider dragging
            if self.dragging_slider and (flags & cv2.EVENT_FLAG_LBUTTON):
                self._update_slider_drag(x)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check slider hitboxes first
            for name, (sx, sy, sw, sh) in self.slider_boxes.items():
                if sx <= x <= sx + sw and sy <= y <= sy + sh:
                    self.dragging_slider = name
                    self._update_slider_drag(x)
                    return
            # Check button hitboxes
            for name, (bx, by, bw, bh) in self.ui_boxes.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.handle_click(name)
                    return
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_slider = None
            
    def _update_slider_drag(self, mouse_x):
        """Update a slider value based on mouse X position."""
        if not self.dragging_slider:
            return
        info = self.slider_boxes.get(self.dragging_slider)
        if not info:
            return
        sx, sy, sw, sh = info
        ratio = max(0.0, min(1.0, (mouse_x - sx) / max(1, sw)))
        
        if self.dragging_slider == "line_thickness":
            self.line_thickness = int(1 + ratio * 9)
        elif self.dragging_slider == "transparency":
            self.transparency = round(ratio * 0.5, 2)
        elif self.dragging_slider == "glow_intensity":
            self.glow_intensity = round(ratio * 3.0, 1)
        elif self.dragging_slider == "animation_speed":
            self.animation_speed = round(0.1 + ratio * 2.9, 1)
                    
    def handle_click(self, name):
        if name == "SETTINGS":
            self.show_settings = not self.show_settings
        elif name == "CLOSE_SETTINGS":
            self.show_settings = False
        elif name.startswith("TAB_"):
            self.settings_tab = int(name.split("_")[1])
        elif name.startswith("MODE_"):
            self.mode = name.split("_", 1)[1].lower()
        elif name == "SNAPSHOT":
            if self.timer_mode > 0:
                self.action_start_time = time.time()
                self.pending_action = "CAPTURE"
            else:
                self.capture_image = True
        elif name == "RECORD":
            if not self.is_recording and self.timer_mode > 0:
                self.action_start_time = time.time()
                self.pending_action = "RECORD"
            else:
                self.toggle_record()
        elif name == "HOLO_CAPTURE":
            self.holo_only_capture = True
            self.capture_image = True
        elif name.startswith("TIMER_"):
            self.timer_mode = int(name.split("_")[1])
        elif name == "SWITCH_CAM":
            self.cam.stop()
            self.cam_index = (self.cam_index + 1) % 10
            self.cam = CameraThread(src=self.cam_index)
        elif name == "CONNECT_PHONE":
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            ip = simpledialog.askstring("NexusCam", "Enter IP address (e.g. 192.168.1.5):")
            root.destroy()
            if ip:
                ip = ip.strip()
                if not ip.startswith("http"):
                    ip = f"http://{ip}:8080/video"
                
                # Generate and show QR Code
                qr_img = generate_qr_opencv(ip)
                cv2.imshow("NexusCam - Scan QR to Connect", qr_img)
                cv2.waitKey(2000) # Show for 2 seconds
                cv2.destroyWindow("NexusCam - Scan QR to Connect")

                self.cam.stop()
                self.cam = CameraThread(src=ip)
        elif name.startswith("NEON_COLOR_"):
            self.neon_color_idx = int(name.split("_")[-1])
        elif name.startswith("PARTICLE_COLOR_"):
            self.particle_color_idx = int(name.split("_")[-1])
            self.particle_color = NEON_PALETTE[self.particle_color_idx % len(NEON_PALETTE)]
        elif name == "PARTICLES_TOGGLE":
            self.particles_enabled = not self.particles_enabled
        elif name == "XRAY_TOGGLE":
            self.xray_enabled = not self.xray_enabled
        elif name == "MASKING_TOGGLE":
            self.holo_masking_enabled = not self.holo_masking_enabled
        elif name == "GESTURE_TOGGLE":
            self.gesture.enabled = not self.gesture.enabled
        elif name.startswith("PRESET_SAVE_"):
            slot = int(name.split("_")[-1])
            d = self.get_settings_dict()
            d["name"] = f"{self.mode.title()} {slot}"
            self.presets.save(slot, d)
        elif name.startswith("PRESET_LOAD_"):
            slot = int(name.split("_")[-1])
            d = self.presets.load(slot)
            if d:
                self.apply_settings_dict(d)
        elif name == "HOLO_TEXT_EDIT":
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            txt = simpledialog.askstring("Hologram Text", "Enter text to display in hologram:", initialvalue=self.holo_text)
            root.destroy()
            if txt is not None:
                self.holo_text = txt
                    
    def toggle_record(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        else:
            self.is_recording = True
        
    def get_quad(self, smooth1, smooth2):
        h1_top, h1_bot = smooth1[8], smooth1[4]
        h2_top, h2_bot = smooth2[8], smooth2[4]
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
            (int(left_bot[0]), int(left_bot[1]))
        ]
        
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        if max(xs) - min(xs) < 20 or max(ys) - min(ys) < 20:
            return None
        return quad

    # ==========================================
    # HUD / UI DRAWING
    # ==========================================
    def draw_hud(self, frame, active):
        """Draw the always-visible minimal HUD with capture buttons."""
        h, w = frame.shape[:2]
        self.ui_boxes.clear()
        self.slider_boxes.clear()
        
        ui = frame.copy()
        
        # ── Top Bar (thin, subtle) ──
        cv2.rectangle(ui, (0, 0), (w, 36), C_BG, -1)
        cv2.addWeighted(ui, 0.85, frame, 0.15, 0, frame)
        
        # Settings gear button (top-left)
        gx, gy, gw, gh = 8, 4, 28, 28
        self.ui_boxes["SETTINGS"] = (gx, gy, gw, gh)
        gear_hover = gx <= self.mouse_x <= gx + gw and gy <= self.mouse_y <= gy + gh
        gear_c = C_ACCENT if gear_hover else C_TEXT_DIM
        # Draw gear icon (simple ⚙ representation)
        cv2.circle(frame, (gx + 14, gy + 14), 8, gear_c, 1, cv2.LINE_AA)
        cv2.circle(frame, (gx + 14, gy + 14), 3, gear_c, -1, cv2.LINE_AA)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            ox = int(gx + 14 + math.cos(rad) * 10)
            oy = int(gy + 14 + math.sin(rad) * 10)
            cv2.circle(frame, (ox, oy), 2, gear_c, -1, cv2.LINE_AA)
        
        # Mode label (top-center)
        mode_text = f"MODE: {self.mode.upper()}"
        ts = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, mode_text, ((w - ts[0]) // 2, 24), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_ACCENT, 1, cv2.LINE_AA)
        
        # Status (top-right) — small, unobtrusive
        status_text = "ACTIVE" if active else "STANDBY"
        status_c = C_SUCCESS if active else C_TEXT_DIM
        cv2.circle(frame, (w - 80, 18), 4, status_c, -1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (w - 72, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, status_c, 1, cv2.LINE_AA)
        
        # ── Bottom Bar — Always-visible capture controls ──
        bar_h = 50
        bar_y = h - bar_h
        cv2.rectangle(ui, (0, bar_y), (w, h), C_BG, -1)
        cv2.addWeighted(ui[bar_y:], 0.85, frame[bar_y:], 0.15, 0, frame[bar_y:])
        
        # PHOTO button (bottom-center-left)
        px, py, pw, ph = w // 2 - 100, bar_y + 8, 80, 34
        self.ui_boxes["SNAPSHOT"] = (px, py, pw, ph)
        photo_hover = px <= self.mouse_x <= px + pw and py <= self.mouse_y <= py + ph
        photo_bg = C_PANEL_HOVER if photo_hover else C_PANEL
        draw_rounded_rect(frame, px, py, pw, ph, photo_bg, 4)
        draw_rounded_rect(frame, px, py, pw, ph, C_ACCENT, 4, fill=False)
        # Camera icon
        cv2.circle(frame, (px + 16, py + ph // 2), 6, C_ACCENT, 1, cv2.LINE_AA)
        cv2.circle(frame, (px + 16, py + ph // 2), 3, C_ACCENT, -1, cv2.LINE_AA)
        cv2.putText(frame, "PHOTO", (px + 28, py + ph // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_ACCENT, 1, cv2.LINE_AA)
        
        # RECORD button (bottom-center-right)
        rx, ry, rw, rh = w // 2 + 20, bar_y + 8, 80, 34
        self.ui_boxes["RECORD"] = (rx, ry, rw, rh)
        rec_hover = rx <= self.mouse_x <= rx + rw and ry <= self.mouse_y <= ry + rh
        if self.is_recording:
            rec_bg = (40, 30, 60) if rec_hover else (30, 20, 50)
            rec_border = C_DANGER
            rec_text = "STOP"
        else:
            rec_bg = C_PANEL_HOVER if rec_hover else C_PANEL
            rec_border = C_DANGER
            rec_text = "REC"
        draw_rounded_rect(frame, rx, ry, rw, rh, rec_bg, 4)
        draw_rounded_rect(frame, rx, ry, rw, rh, rec_border, 4, fill=False)
        # Record dot
        if self.is_recording and int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (rx + 14, ry + rh // 2), 5, C_DANGER, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (rx + 14, ry + rh // 2), 5, C_DANGER, 1, cv2.LINE_AA)
        cv2.putText(frame, rec_text, (rx + 26, ry + rh // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_DANGER if self.is_recording else C_TEXT, 1, cv2.LINE_AA)
        
        # Timer indicator (bottom-left)
        timer_text = f"TIMER: {self.timer_mode}s" if self.timer_mode > 0 else "TIMER: OFF"
        cv2.putText(frame, timer_text, (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1, cv2.LINE_AA)
        
        # Recording time (if recording)
        if self.is_recording:
            rec_elapsed = int(time.time() - getattr(self, '_rec_start', time.time()))
            rec_str = f"REC {rec_elapsed // 60:02d}:{rec_elapsed % 60:02d}"
            cv2.putText(frame, rec_str, (w - 130, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_DANGER, 1, cv2.LINE_AA)

        # ── Settings Panel ──
        if self.show_settings:
            self._draw_settings_panel(frame, w, h)
            
        # ── Countdown ──
        if self.pending_action:
            rem = int((self.action_start_time + self.timer_mode) - time.time()) + 1
            if rem > 0:
                # Dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                # Big number
                cv2.putText(frame, f"{rem}", (w // 2 - 40, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, C_ACCENT, 6, cv2.LINE_AA)
                cv2.putText(frame, f"{self.pending_action}...", (w // 2 - 60, h // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
                            
        # ── Flash effect ──
        if time.time() < self.flash_until:
            cv2.addWeighted(np.ones_like(frame) * 255, 0.5, frame, 0.5, 0, frame)
    
    def _draw_settings_panel(self, frame, w, h):
        """Draw the tabbed settings panel overlay."""
        # Full-screen dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Panel dimensions
        pw, ph = min(560, w - 40), min(440, h - 60)
        px = (w - pw) // 2
        py = (h - ph) // 2
        
        # Panel background
        draw_rounded_rect(frame, px, py, pw, ph, C_PANEL, 8)
        draw_rounded_rect(frame, px, py, pw, ph, C_ACCENT, 8, fill=False)
        
        # Title bar
        cv2.putText(frame, "NEXUS SETTINGS", (px + 16, py + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ACCENT, 1, cv2.LINE_AA)
        
        # Close button
        cx, cy, cw, ch = px + pw - 32, py + 4, 24, 24
        self.ui_boxes["CLOSE_SETTINGS"] = (cx, cy, cw, ch)
        close_hover = cx <= self.mouse_x <= cx + cw and cy <= self.mouse_y <= cy + ch
        cv2.putText(frame, "X", (cx + 6, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    C_DANGER if close_hover else C_TEXT_DIM, 1, cv2.LINE_AA)
        
        # ── Tab bar ──
        tab_y = py + 36
        tab_w = (pw - 16) // len(self.tab_names)
        for i, name in enumerate(self.tab_names):
            tx = px + 8 + i * tab_w
            tw = tab_w - 4
            self.ui_boxes[f"TAB_{i}"] = (tx, tab_y, tw, 26)
            tab_hover = tx <= self.mouse_x <= tx + tw and tab_y <= self.mouse_y <= tab_y + 26
            is_active = self.settings_tab == i
            bg = C_ACCENT if is_active else (C_PANEL_HOVER if tab_hover else C_BG)
            text_c = C_BG if is_active else C_TEXT
            draw_rounded_rect(frame, tx, tab_y, tw, 26, bg, 4)
            draw_text_centered(frame, name, tx, tab_y, tw, 26, text_c, 0.32)
        
        # ── Tab content area ──
        content_y = tab_y + 36
        content_x = px + 16
        content_w = pw - 32
        
        if self.settings_tab == 0:
            self._draw_effects_tab(frame, content_x, content_y, content_w)
        elif self.settings_tab == 1:
            self._draw_particles_tab(frame, content_x, content_y, content_w)
        elif self.settings_tab == 2:
            self._draw_capture_tab(frame, content_x, content_y, content_w)
        elif self.settings_tab == 3:
            self._draw_camera_tab(frame, content_x, content_y, content_w)
    
    def _draw_effects_tab(self, frame, x, y, w):
        """Effects tab: mode selection, sliders, presets."""
        # Mode buttons (2x2 grid)
        modes = ["hologram", "neon", "matrix", "plasma"]
        mode_w = (w - 8) // 2
        for i, m in enumerate(modes):
            mx = x + (i % 2) * (mode_w + 8)
            my = y + (i // 2) * 34
            self.ui_boxes[f"MODE_{m.upper()}"] = (mx, my, mode_w, 28)
            hover = mx <= self.mouse_x <= mx + mode_w and my <= self.mouse_y <= my + 28
            is_active = self.mode == m
            bg = C_ACCENT if is_active else (C_PANEL_HOVER if hover else C_BG)
            text_c = C_BG if is_active else C_TEXT
            draw_rounded_rect(frame, mx, my, mode_w, 28, bg, 4)
            draw_text_centered(frame, m.upper(), mx, my, mode_w, 28, text_c, 0.35)
        
        # Sliders
        slider_y = y + 78
        sb = draw_slider(frame, x, slider_y, w, self.line_thickness, 1, 10, "Line Thickness")
        self.slider_boxes["line_thickness"] = sb
        
        slider_y += 34
        sb = draw_slider(frame, x, slider_y, w, self.transparency, 0.0, 0.5, "Fill Opacity")
        self.slider_boxes["transparency"] = sb
        
        slider_y += 34
        sb = draw_slider(frame, x, slider_y, w, self.glow_intensity, 0.0, 3.0, "Glow Intensity")
        self.slider_boxes["glow_intensity"] = sb
        
        slider_y += 34
        sb = draw_slider(frame, x, slider_y, w, self.animation_speed, 0.1, 3.0, "Animation Speed")
        self.slider_boxes["animation_speed"] = sb
        
        # Neon color swatches
        slider_y += 38
        cv2.putText(frame, "Neon Color:", (x, slider_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1, cv2.LINE_AA)
        for i, color in enumerate(NEON_PALETTE):
            sx = x + 80 + i * 24
            self.ui_boxes[f"NEON_COLOR_{i}"] = (sx, slider_y, 20, 20)
            cv2.rectangle(frame, (sx, slider_y), (sx + 20, slider_y + 20), color, -1)
            if i == self.neon_color_idx:
                cv2.rectangle(frame, (sx - 1, slider_y - 1), (sx + 21, slider_y + 21), (255, 255, 255), 2)
        
        # X-Ray toggle
        slider_y += 30
        self._draw_toggle(frame, x, slider_y, "X-Ray Mode", "XRAY_TOGGLE", self.xray_enabled)
        
        # Holographic masking toggle
        slider_y += 28
        self._draw_toggle(frame, x, slider_y, "Holo Masking", "MASKING_TOGGLE", self.holo_masking_enabled)
        
        # Holo text button
        slider_y += 28
        bx, by, bw, bh = x, slider_y, w // 2, 26
        self.ui_boxes["HOLO_TEXT_EDIT"] = (bx, by, bw, bh)
        hover = bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh
        draw_rounded_rect(frame, bx, by, bw, bh, C_PANEL_HOVER if hover else C_BG, 4)
        draw_rounded_rect(frame, bx, by, bw, bh, C_ACCENT, 4, fill=False)
        label = f'Text: "{self.holo_text[:15]}"' if self.holo_text else "Set Holo Text..."
        draw_text_centered(frame, label, bx, by, bw, bh, C_ACCENT, 0.3)
        
        # ── Preset Slots ──
        slider_y += 36
        cv2.putText(frame, "PRESETS:", (x, slider_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1, cv2.LINE_AA)
        slider_y += 16
        slot_w = (w - 16) // 9
        for i in range(1, 10):
            sx = x + (i - 1) * (slot_w + 2)
            has_data = self.presets.slots.get(i) is not None
            # Click toggles save/load
            self.ui_boxes[f"PRESET_LOAD_{i}"] = (sx, slider_y, slot_w, 20)
            hover = sx <= self.mouse_x <= sx + slot_w and slider_y <= self.mouse_y <= slider_y + 20
            bg = C_ACCENT if has_data else (C_PANEL_HOVER if hover else C_BG)
            text_c = C_BG if has_data else C_TEXT_DIM
            draw_rounded_rect(frame, sx, slider_y, slot_w, 20, bg, 3)
            draw_text_centered(frame, str(i), sx, slider_y, slot_w, 20, text_c, 0.3)
        # Save row
        slider_y += 24
        cv2.putText(frame, "SAVE TO:", (x, slider_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_TEXT_DIM, 1, cv2.LINE_AA)
        slider_y += 14
        for i in range(1, 10):
            sx = x + (i - 1) * (slot_w + 2)
            self.ui_boxes[f"PRESET_SAVE_{i}"] = (sx, slider_y, slot_w, 18)
            hover = sx <= self.mouse_x <= sx + slot_w and slider_y <= self.mouse_y <= slider_y + 18
            draw_rounded_rect(frame, sx, slider_y, slot_w, 18, C_PANEL_HOVER if hover else C_BG, 3)
            draw_rounded_rect(frame, sx, slider_y, slot_w, 18, C_TEXT_DIM, 3, fill=False)
            draw_text_centered(frame, f"S{i}", sx, slider_y, slot_w, 18, C_TEXT_DIM, 0.25)
    
    def _draw_particles_tab(self, frame, x, y, w):
        """Particles tab: toggle, color picker."""
        self._draw_toggle(frame, x, y, "Particles Enabled", "PARTICLES_TOGGLE", self.particles_enabled)
        
        # Color swatches
        y += 36
        cv2.putText(frame, "Particle Color:", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1, cv2.LINE_AA)
        for i, color in enumerate(NEON_PALETTE):
            sx = x + 100 + i * 24
            self.ui_boxes[f"PARTICLE_COLOR_{i}"] = (sx, y, 20, 20)
            cv2.rectangle(frame, (sx, y), (sx + 20, y + 20), color, -1)
            if i == self.particle_color_idx:
                cv2.rectangle(frame, (sx - 1, y - 1), (sx + 21, y + 21), (255, 255, 255), 2)
    
    def _draw_capture_tab(self, frame, x, y, w):
        """Capture tab: timer, holo-only capture toggle."""
        # Timer buttons
        cv2.putText(frame, "Timer:", (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT_DIM, 1, cv2.LINE_AA)
        timers = [0, 3, 5, 10]
        for i, t in enumerate(timers):
            tx = x + 60 + i * 56
            self.ui_boxes[f"TIMER_{t}"] = (tx, y, 48, 24)
            hover = tx <= self.mouse_x <= tx + 48 and y <= self.mouse_y <= y + 24
            is_active = self.timer_mode == t
            bg = C_ACCENT if is_active else (C_PANEL_HOVER if hover else C_BG)
            text_c = C_BG if is_active else C_TEXT
            draw_rounded_rect(frame, tx, y, 48, 24, bg, 4)
            label = "OFF" if t == 0 else f"{t}s"
            draw_text_centered(frame, label, tx, y, 48, 24, text_c, 0.35)
        
        # Holo-only capture button
        y += 40
        bx, by, bw, bh = x, y, w // 2, 30
        self.ui_boxes["HOLO_CAPTURE"] = (bx, by, bw, bh)
        hover = bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh
        draw_rounded_rect(frame, bx, by, bw, bh, C_PANEL_HOVER if hover else C_BG, 4)
        draw_rounded_rect(frame, bx, by, bw, bh, C_ACCENT2, 4, fill=False)
        draw_text_centered(frame, "HOLOGRAM-ONLY CAPTURE", bx, by, bw, bh, C_ACCENT2, 0.32)
    
    def _draw_camera_tab(self, frame, x, y, w):
        """Camera tab: switch, connect phone, gesture toggle."""
        buttons = [
            ("SWITCH_CAM", "Switch Camera"),
            ("CONNECT_PHONE", "Connect Phone (IP)"),
        ]
        for i, (name, label) in enumerate(buttons):
            bx, by, bw, bh = x, y + i * 38, w // 2, 30
            self.ui_boxes[name] = (bx, by, bw, bh)
            hover = bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh
            draw_rounded_rect(frame, bx, by, bw, bh, C_PANEL_HOVER if hover else C_BG, 4)
            draw_rounded_rect(frame, bx, by, bw, bh, C_ACCENT, 4, fill=False)
            draw_text_centered(frame, label, bx, by, bw, bh, C_ACCENT, 0.32)
        
        # Gesture toggle
        y += 90
        self._draw_toggle(frame, x, y, "Gesture Control (Pinch/Swipe/V)", "GESTURE_TOGGLE", self.gesture.enabled)
    
    def _draw_toggle(self, frame, x, y, label, name, value):
        """Draw an ON/OFF toggle switch."""
        tw, th = 44, 22
        self.ui_boxes[name] = (x, y, tw + 160, th)
        
        # Track
        track_c = C_SUCCESS if value else C_SLIDER_TRACK
        draw_rounded_rect(frame, x, y, tw, th, track_c, th // 2)
        
        # Knob
        knob_x = x + tw - th + 2 if value else x + 2
        cv2.circle(frame, (knob_x + th // 2 - 2, y + th // 2), th // 2 - 3, (255, 255, 255), -1, cv2.LINE_AA)
        
        # Label
        cv2.putText(frame, label, (x + tw + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT, 1, cv2.LINE_AA)

    # ==========================================
    # X-RAY & HOLOGRAPHIC MASKING
    # ==========================================
    def apply_xray(self, frame, quad):
        """Apply X-Ray effect inside the hologram quad."""
        pts_arr = np.array(quad, dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_arr], 255)
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply thermal colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Blend edges into thermal
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        xray = cv2.addWeighted(thermal, 0.7, edges_bgr, 0.5, 0)
        
        # Apply only inside mask
        mask_3ch = cv2.merge([mask, mask, mask])
        frame_masked = np.where(mask_3ch > 0, xray, frame)
        np.copyto(frame, frame_masked)
    
    def apply_holo_masking(self, frame, quad):
        """Apply holographic glitch texture to everything inside the quad."""
        pts_arr = np.array(quad, dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_arr], 255)
        
        h, w = frame.shape[:2]
        
        # Create holographic texture
        holo = np.zeros_like(frame)
        t = self.anim_frame * 0.05
        
        # Scan lines
        for ly in range(0, h, 3):
            intensity = int(40 + 20 * math.sin(ly * 0.1 + t))
            cv2.line(holo, (0, ly), (w, ly), (0, intensity, intensity // 2), 1)
        
        # Color shift
        roi = frame.copy()
        b, g, r = cv2.split(roi)
        shift = int(3 * math.sin(t * 2))
        b = np.roll(b, -shift, axis=1)
        r = np.roll(r, shift, axis=1)
        glitched = cv2.merge([b, g, r])
        
        # Blend
        result = cv2.addWeighted(glitched, 0.7, holo, 0.3, 0)
        
        mask_3ch = cv2.merge([mask, mask, mask])
        frame_masked = np.where(mask_3ch > 0, result, frame)
        np.copyto(frame, frame_masked)

    # ==========================================
    # HOLOGRAM TEXT RENDERING
    # ==========================================
    def draw_holo_text(self, frame, quad):
        """Draw custom text centered inside the hologram."""
        if not self.holo_text:
            return
        pts_arr = np.array(quad, dtype=np.int32)
        cx = int(np.mean(pts_arr[:, 0]))
        cy = int(np.mean(pts_arr[:, 1]))
        
        # Measure text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        ts = cv2.getTextSize(self.holo_text, font, scale, thickness)[0]
        tx = cx - ts[0] // 2
        ty = cy + ts[1] // 2
        
        # Glow layers
        glow_c = tuple(int(v * 0.3) for v in C_ACCENT)
        cv2.putText(frame, self.holo_text, (tx, ty), font, scale, glow_c, thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, self.holo_text, (tx, ty), font, scale, C_ACCENT, thickness, cv2.LINE_AA)

    # ==========================================
    # MAIN LOOP
    # ==========================================
    def run(self):
        window_name = "NEXUS FRAME"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            # ── Check if window was closed via X button ──
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            frame = self.cam.read()
            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA SIGNAL", (440, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_DANGER, 2, cv2.LINE_AA)
                cv2.putText(frame, "Open Settings > Camera > Switch or Connect", (340, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT_DIM, 1, cv2.LINE_AA)
                active = False
            else:
                self.anim_frame += int(self.animation_speed * 1 + 0.5)
            h, w = frame.shape[:2]
            
            # ── MediaPipe Inference ──
            ts = int(time.time() * 1000) - self.start_ms
            if not hasattr(self, 'last_ts'):
                self.last_ts = -1
            if ts <= self.last_ts:
                ts = self.last_ts + 1
            self.last_ts = ts
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.detector.detect_async(mp_img, ts)
            
            # ── Get result ──
            with self.result_lock:
                res = self.latest_result
                
            active = False
            quad = None
            if res and res.hand_landmarks and len(res.hand_landmarks) >= 2:
                smoothed = []
                for idx, lms in enumerate(res.hand_landmarks[:2]):
                    k = f"hand_{idx}"
                    if k not in self.smoothers:
                        self.smoothers[k] = HandSmoother()
                    smoothed.append(self.smoothers[k].smooth(lms, w, h))
                    
                quad = self.get_quad(smoothed[0], smoothed[1])
                if quad:
                    active = True
                    self.last_quad = quad
                    
                    # ── Gesture Detection ──
                    if self.gesture.enabled:
                        # Peace sign on either hand
                        for hand_lms in res.hand_landmarks[:2]:
                            if self.gesture.detect_peace_sign(hand_lms, w, h):
                                if self.gesture.peace_start == 0:
                                    self.gesture.peace_start = time.time()
                                elif time.time() - self.gesture.peace_start > 2.0 and not self.gesture.peace_triggered:
                                    self.gesture.peace_triggered = True
                                    self.action_start_time = time.time()
                                    self.pending_action = "CAPTURE"
                                    self.timer_mode = 3
                            else:
                                self.gesture.peace_start = 0
                                self.gesture.peace_triggered = False
                        
                        # Swipe detection
                        for idx, sm in enumerate(smoothed):
                            direction = self.gesture.detect_swipe(idx, sm)
                            if direction == "right":
                                modes = list(RENDER_FNS.keys())
                                ci = modes.index(self.mode) if self.mode in modes else 0
                                self.mode = modes[(ci + 1) % len(modes)]
                            elif direction == "left":
                                modes = list(RENDER_FNS.keys())
                                ci = modes.index(self.mode) if self.mode in modes else 0
                                self.mode = modes[(ci - 1) % len(modes)]
                    
                    # ── Apply X-Ray or Holo Masking BEFORE effect rendering ──
                    if self.xray_enabled:
                        self.apply_xray(frame, quad)
                    elif self.holo_masking_enabled:
                        self.apply_holo_masking(frame, quad)
                    
                    # ── Render Effect ──
                    render_fn = RENDER_FNS.get(self.mode, render_hologram)
                    render_fn(frame, quad, self.anim_frame, self.p_sys,
                              neon_color=NEON_PALETTE[self.neon_color_idx % len(NEON_PALETTE)],
                              line_thickness=self.line_thickness,
                              transparency=self.transparency,
                              glow_intensity=self.glow_intensity)
                    
                    # ── Hologram Text ──
                    self.draw_holo_text(frame, quad)

            # ── Countdown Timer ──
            if self.pending_action:
                if time.time() > self.action_start_time + self.timer_mode:
                    if self.pending_action == "CAPTURE":
                        self.capture_image = True
                    elif self.pending_action == "RECORD":
                        self.toggle_record()
                    self.pending_action = None
                    
            # ── Particles ──
            if self.particles_enabled:
                self.p_sys.update_and_draw(frame)
            
            # ── Save clean frame BEFORE drawing HUD ──
            clean_frame = frame.copy()
            
            # ── Draw HUD (only on display frame) ──
            self.draw_hud(frame, active)
            
            # ── FPS ──
            self.fps_count += 1
            now = time.time()
            if now - self.fps_timer >= 1.0:
                self.fps = self.fps_count / (now - self.fps_timer)
                self.fps_count = 0
                self.fps_timer = now
                
            # ── Capture (uses clean_frame, no HUD) ──
            if self.capture_image:
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                if self.holo_only_capture and self.last_quad:
                    # Crop to hologram region with alpha mask
                    pts_arr = np.array(self.last_quad, dtype=np.int32)
                    x_r, y_r, bw_r, bh_r = cv2.boundingRect(pts_arr)
                    # Create BGRA with transparency
                    cropped = clean_frame[y_r:y_r+bh_r, x_r:x_r+bw_r].copy()
                    alpha = np.zeros((bh_r, bw_r), dtype=np.uint8)
                    shifted_pts = pts_arr - np.array([x_r, y_r])
                    cv2.fillPoly(alpha, [shifted_pts], 255)
                    bgra = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
                    bgra[:, :, 3] = alpha
                    path = os.path.join(self.media_dir, f"nexus_holo_{ts_str}.png")
                    cv2.imwrite(path, bgra)
                    self.holo_only_capture = False
                else:
                    path = os.path.join(self.media_dir, f"nexus_snap_{ts_str}.png")
                    cv2.imwrite(path, clean_frame)
                self.capture_image = False
                self.flash_until = time.time() + 0.15
                print(f"[SAVED] {path}")
                    
            # ── Record (uses clean_frame, no HUD) ──
            if self.is_recording:
                if self.video_writer is None:
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(self.media_dir, f"nexus_rec_{ts_str}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
                    self._rec_start = time.time()
                self.video_writer.write(clean_frame)
                
            cv2.imshow(window_name, frame)
            
            # ── Key Handling ──
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord('1'):
                self.mode = "hologram"
            elif key == ord('2'):
                self.mode = "neon"
            elif key == ord('3'):
                self.mode = "matrix"
            elif key == ord('4'):
                self.mode = "plasma"
            elif key == ord(' '):
                if self.timer_mode > 0:
                    self.action_start_time = time.time()
                    self.pending_action = "CAPTURE"
                else:
                    self.capture_image = True
            elif key == ord('r') or key == ord('R'):
                self.toggle_record()
            elif key == ord('t') or key == ord('T'):
                timers = [0, 3, 5, 10]
                ci = timers.index(self.timer_mode) if self.timer_mode in timers else 0
                self.timer_mode = timers[(ci + 1) % len(timers)]
            elif key == 0x7A:  # F11 (scancode varies, fallback)
                self._toggle_fullscreen(window_name)
                
        # ── Cleanup ──
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        self.cam.stop()
        self.detector.close()
        cv2.destroyAllWindows()
        
    def _toggle_fullscreen(self, window_name):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

if __name__ == "__main__":
    NexusApp()
