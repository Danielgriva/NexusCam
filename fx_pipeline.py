import cv2
import numpy as np
import random
import math

class ParticleSystem:
    def __init__(self, max_particles=1000):
        self.max_particles = max_particles
        self.count = 0
        # State arrays for vectorization
        self.x = np.zeros(max_particles, dtype=np.float32)
        self.y = np.zeros(max_particles, dtype=np.float32)
        self.vx = np.zeros(max_particles, dtype=np.float32)
        self.vy = np.zeros(max_particles, dtype=np.float32)
        self.life = np.zeros(max_particles, dtype=np.float32)
        self.decay = np.zeros(max_particles, dtype=np.float32)
        self.color = np.zeros((max_particles, 3), dtype=np.uint8)
        self.size = np.zeros(max_particles, dtype=np.float32)

    def spawn(self, x, y, vx, vy, life, decay, color, size):
        if self.count >= self.max_particles:
            return
        idx = self.count
        self.x[idx] = x
        self.y[idx] = y
        self.vx[idx] = vx
        self.vy[idx] = vy
        self.life[idx] = life
        self.decay[idx] = decay
        self.color[idx] = color
        self.size[idx] = size
        self.count += 1

    def spawn_burst(self, x, y, num=20, base_color=(200, 255, 0)):
        for _ in range(min(num, self.max_particles - self.count)):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1.0, 5.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = 1.0
            decay = random.uniform(0.015, 0.04)
            size = random.uniform(1.0, 3.0)
            self.spawn(x, y, vx, vy, life, decay, base_color, size)

    def update_and_draw(self, frame):
        if self.count == 0:
            return

        # Vectorized physics update
        active = self.life[:self.count] > 0
        
        self.x[:self.count][active] += self.vx[:self.count][active]
        self.y[:self.count][active] += self.vy[:self.count][active]
        self.vy[:self.count][active] += 0.1  # Gravity
        
        # Air friction
        self.vx[:self.count][active] *= 0.98
        self.vy[:self.count][active] *= 0.98
        
        self.life[:self.count][active] -= self.decay[:self.count][active]

        # Draw active particles
        for i in range(self.count):
            if self.life[i] > 0:
                cx, cy = int(self.x[i]), int(self.y[i])
                s = max(1, int(self.size[i] * self.life[i]))
                c = tuple(int(v) for v in self.color[i])
                cv2.circle(frame, (cx, cy), s, c, -1, cv2.LINE_AA)

        # Compact array by removing dead particles
        still_alive = self.life[:self.count] > 0
        new_count = np.sum(still_alive)
        if new_count < self.count:
            self.x[:new_count] = self.x[:self.count][still_alive]
            self.y[:new_count] = self.y[:self.count][still_alive]
            self.vx[:new_count] = self.vx[:self.count][still_alive]
            self.vy[:new_count] = self.vy[:self.count][still_alive]
            self.life[:new_count] = self.life[:self.count][still_alive]
            self.decay[:new_count] = self.decay[:self.count][still_alive]
            self.color[:new_count] = self.color[:self.count][still_alive]
            self.size[:new_count] = self.size[:self.count][still_alive]
            self.count = new_count


def draw_electric_arc(frame, pt1, pt2, color=(255, 200, 0), max_displacement=15, segments=5, thickness=2):
    """Draws jagged lightning between two points."""
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 5:
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        return

    pts = [np.array(pt1, dtype=np.float32)]
    
    # Subdivide segment and add noise
    for i in range(1, segments):
        t = i / segments
        base_x = pt1[0] + (pt2[0] - pt1[0]) * t
        base_y = pt1[1] + (pt2[1] - pt1[1]) * t
        
        # Perpendicular normal
        nx = -(pt2[1] - pt1[1]) / dist
        ny = (pt2[0] - pt1[0]) / dist
        
        offset = random.uniform(-max_displacement, max_displacement)
        pts.append(np.array([base_x + nx * offset, base_y + ny * offset], dtype=np.float32))
        
    pts.append(np.array(pt2, dtype=np.float32))
    
    pts_int = np.int32(pts).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_int], False, color, thickness, cv2.LINE_AA)
    # Glow layer
    glow_t = max(1, thickness + 4)
    cv2.polylines(frame, [pts_int], False, tuple(int(v*0.5) for v in color), glow_t, cv2.LINE_AA)

def chromatic_aberration(frame, pts):
    """Apply RGB split glitch on the edges."""
    h, w = frame.shape[:2]
    pts_arr = np.array(pts, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts_arr)
    
    # Pad rect
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    
    if x2 <= x1 or y2 <= y1:
        return frame
        
    roi = frame[y1:y2, x1:x2].copy()
    
    # Shift channels
    shift = 3
    b, g, r = cv2.split(roi)
    
    b_shifted = np.roll(b, -shift, axis=1)
    r_shifted = np.roll(r, shift, axis=1)
    
    glitched = cv2.merge([b_shifted, g, r_shifted])
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_arr], 255)
    
    # Extract only the edges for glitch
    mask_edges = cv2.Canny(mask, 100, 200)
    mask_edges = cv2.dilate(mask_edges, np.ones((5,5), np.uint8), iterations=1)
    
    roi_mask = mask_edges[y1:y2, x1:x2]
    
    # Blend back
    idx = roi_mask > 0
    frame[y1:y2, x1:x2][idx] = glitched[idx]


# ---------------------------------------------------------
# DISTINCT RENDERING MODES (1-4)
# All accept **kwargs for: line_thickness, transparency, glow_intensity
# ---------------------------------------------------------

def render_hologram(frame, pts, anim_frame, p_sys=None, **kwargs):
    """Clean, professional cyan UI frame with customizable parameters."""
    line_thickness = kwargs.get("line_thickness", 2)
    transparency = kwargs.get("transparency", 0.15)
    glow_intensity = kwargs.get("glow_intensity", 1.0)
    color_cyan = (255, 200, 0)  # BGR (Cyan-ish)
    pts_arr = np.array(pts, dtype=np.int32)
    
    # Filled quad with adjustable transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_arr], color_cyan)
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)
    
    # Border with adjustable thickness
    cv2.polylines(frame, [pts_arr], True, color_cyan, line_thickness, cv2.LINE_AA)
    
    # Glow layer (scaled by glow_intensity)
    if glow_intensity > 0.1:
        glow_t = max(1, int(line_thickness * 2 * glow_intensity))
        glow_c = tuple(int(v * 0.3 * glow_intensity) for v in color_cyan)
        cv2.polylines(frame, [pts_arr], True, glow_c, glow_t, cv2.LINE_AA)
    
    # Corner brackets
    for p in pts:
        cx, cy = p
        bracket = int(15 * max(0.5, glow_intensity))
        cv2.line(frame, (cx-bracket, cy), (cx+bracket, cy), color_cyan, line_thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy-bracket), (cx, cy+bracket), color_cyan, line_thickness, cv2.LINE_AA)
    
    # Scan line effect inside quad
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_arr], 255)
    x, y, bw, bh = cv2.boundingRect(pts_arr)
    scan_overlay = np.zeros_like(frame)
    step = max(3, 6 - int(glow_intensity))
    for ly in range(y, y + bh, step):
        cv2.line(scan_overlay, (x, ly), (x + bw, ly), color_cyan, 1)
    scan_masked = cv2.bitwise_and(scan_overlay, scan_overlay, mask=mask)
    cv2.addWeighted(scan_masked, 0.05 * glow_intensity, frame, 1.0, 0, frame)
    
    # Moving sweep line
    sweep_y = y + int((anim_frame * 2) % max(1, bh))
    if y <= sweep_y <= y + bh:
        sweep_overlay = np.zeros_like(frame)
        cv2.line(sweep_overlay, (x, sweep_y), (x + bw, sweep_y), color_cyan, 2)
        sweep_masked = cv2.bitwise_and(sweep_overlay, sweep_overlay, mask=mask)
        cv2.addWeighted(sweep_masked, 0.2 * glow_intensity, frame, 1.0, 0, frame)

def render_neon(frame, pts, anim_frame, p_sys=None, **kwargs):
    """Vibrant glow with customizable colors and particle bursts."""
    line_thickness = kwargs.get("line_thickness", 2)
    transparency = kwargs.get("transparency", 0.15)
    glow_intensity = kwargs.get("glow_intensity", 1.0)
    color = kwargs.get("neon_color", (216, 97, 255))
    pts_arr = np.array(pts, dtype=np.int32)
    
    # Filled quad
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_arr], color)
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)
    
    # Multi-layer glow borders
    cv2.polylines(frame, [pts_arr], True, color, line_thickness, cv2.LINE_AA)
    if glow_intensity > 0.1:
        glow_layers = [
            (int(line_thickness * 3 * glow_intensity), 0.15),
            (int(line_thickness * 2 * glow_intensity), 0.35),
        ]
        for thickness, alpha in glow_layers:
            t = max(1, thickness)
            c = tuple(int(v * alpha * glow_intensity) for v in color)
            cv2.polylines(frame, [pts_arr], True, c, t, cv2.LINE_AA)
    
    # Crosshair at center
    t = anim_frame * 0.04
    cx = int(sum(p[0] for p in pts) / 4)
    cy = int(sum(p[1] for p in pts) / 4)
    cs = int(18 + 6 * math.sin(t * 2))
    cross_c = tuple(int(v * 0.4) for v in color)
    cv2.line(frame, (cx - cs, cy), (cx + cs, cy), cross_c, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - cs), (cx, cy + cs), cross_c, 1, cv2.LINE_AA)
    
    # Rotating arcs
    angle_deg = int((t * 80) % 360)
    cv2.ellipse(frame, (cx, cy), (cs * 2, cs * 2), 0,
                angle_deg, angle_deg + 70, color, line_thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (cs * 2, cs * 2), 0,
                angle_deg + 180, angle_deg + 250, color, line_thickness, cv2.LINE_AA)
    
    if p_sys:
        for p in pts:
            if random.random() < 0.3:
                p_sys.spawn_burst(p[0], p[1], num=4, base_color=color)

def render_matrix(frame, pts, anim_frame, p_sys=None, **kwargs):
    """Digital green with data lines."""
    line_thickness = kwargs.get("line_thickness", 2)
    transparency = kwargs.get("transparency", 0.15)
    glow_intensity = kwargs.get("glow_intensity", 1.0)
    color_green = (0, 255, 0)
    pts_arr = np.array(pts, dtype=np.int32)
    
    # Dark fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_arr], (0, 20, 0))
    cv2.addWeighted(overlay, transparency + 0.05, frame, 1 - transparency - 0.05, 0, frame)
    
    # Grid lines inside
    x, y, w, h = cv2.boundingRect(pts_arr)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_arr], 255)
    
    grid_overlay = np.zeros_like(frame)
    step = 20
    for i in range(x, x+w, step):
        cv2.line(grid_overlay, (i, y), (i, y+h), (0, 50, 0), 1)
    for j in range(y, y+h, step):
        cv2.line(grid_overlay, (x, j), (x+w, j), (0, 50, 0), 1)
    grid_masked = cv2.bitwise_and(grid_overlay, grid_overlay, mask=mask)
    cv2.addWeighted(grid_masked, 0.5, frame, 1.0, 0, frame)
    
    # Border
    cv2.polylines(frame, [pts_arr], True, color_green, line_thickness, cv2.LINE_AA)
    if glow_intensity > 0.1:
        glow_c = tuple(int(v * 0.3 * glow_intensity) for v in color_green)
        cv2.polylines(frame, [pts_arr], True, glow_c, int(line_thickness * 2.5 * glow_intensity), cv2.LINE_AA)
    
    # Glitch occasionally
    if anim_frame % 10 == 0:
        chromatic_aberration(frame, pts)

def render_plasma(frame, pts, anim_frame, p_sys=None, **kwargs):
    """The chaotic electric lightning effect (Cyberpunk)."""
    line_thickness = kwargs.get("line_thickness", 2)
    transparency = kwargs.get("transparency", 0.15)
    glow_intensity = kwargs.get("glow_intensity", 1.0)
    color_cyan = (200, 255, 0)
    color_pink = (216, 97, 255)
    
    if anim_frame % 5 == 0:
        chromatic_aberration(frame, pts)

    overlay = frame.copy()
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], color_cyan)
    alpha = transparency + 0.05 * math.sin(anim_frame * 0.1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        if random.random() < 0.4:
            draw_electric_arc(frame, p1, p2, 
                            color=color_cyan if i%2==0 else color_pink,
                            thickness=line_thickness)
            
    if p_sys:
        for p in pts:
            if random.random() < 0.2:
                p_sys.spawn_burst(p[0], p[1], num=3, base_color=color_cyan)
                
    pts_arr = pts_arr.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_arr], True, color_pink, line_thickness, cv2.LINE_AA)
    
    # Glow
    if glow_intensity > 0.1:
        glow_c = tuple(int(v * 0.3 * glow_intensity) for v in color_pink)
        cv2.polylines(frame, [pts_arr], True, glow_c, int(line_thickness * 2 * glow_intensity), cv2.LINE_AA)
    
    for cx, cy in pts:
        bracket = int(10 * max(0.5, glow_intensity))
        cv2.line(frame, (cx-bracket, cy), (cx+bracket, cy), color_cyan, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy-bracket), (cx, cy+bracket), color_cyan, 1, cv2.LINE_AA)
