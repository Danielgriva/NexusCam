/**
 * NEXUS FRAME - Holographic Hand Tracker v3.0
 * =============================================
 * Uses MediaPipe Hand Landmarker to detect two hands and draws
 * a futuristic holographic panel between the fingertips as a
 * proper quadrilateral (not a bounding box).
 *
 * No hand skeleton is drawn. Effects are GPU-friendly.
 * Depth-aware: uses palm size to estimate hand distance.
 */

import {
    HandLandmarker,
    FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// =============================================
// GLOBALS
// =============================================
let handLandmarker;
let webcamRunning = false;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;
let currentMode = "hologram";
let animationFrame = 0;
let particles = [];
let startTime = Date.now();
let wasFrameActive = false;

// Smoothed quad corners (reduces jitter)
let smoothQuad = null;
const SMOOTH = 0.4;

// Offscreen canvas for matrix chars (performance)
let matrixCanvas = null;
let matrixCtx = null;
let matrixInited = false;

// DOM
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const statusPill = document.getElementById("statusPill");
const statusText = statusPill.querySelector(".status-text");
const fpsCounter = document.getElementById("fpsCounter");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingStatus = document.getElementById("loadingStatus");
const hudTime = document.getElementById("hudTime");
const handsDetected = document.getElementById("handsDetected");
const frameStatus = document.getElementById("frameStatus");
const modeSwitcher = document.getElementById("modeSwitcher");
const captureBtn = document.getElementById("captureBtn");
const cameraWrapper = document.getElementById("cameraWrapper");

// =============================================
// INIT MEDIAPIPE
// =============================================
async function initHandLandmarker() {
    loadingStatus.textContent = "Loading vision module...";

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );

    loadingStatus.textContent = "Loading hand landmark model...";

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    loadingStatus.textContent = "Starting camera...";
    await startCamera();
}

// =============================================
// CAMERA
// =============================================
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user",
            },
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            webcamRunning = true;

            // Offscreen canvas for matrix effect
            matrixCanvas = document.createElement("canvas");
            matrixCanvas.width = canvas.width;
            matrixCanvas.height = canvas.height;
            matrixCtx = matrixCanvas.getContext("2d");

            loadingOverlay.classList.add("hidden");
            statusText.textContent = "ACTIVE";
            statusPill.classList.add("active");

            detectHands();
        });
    } catch (err) {
        loadingStatus.textContent = "Camera access denied. Please allow camera.";
        console.error("Camera error:", err);
    }
}

// =============================================
// ESTIMATE HAND DEPTH (distance from camera)
// Uses palm size as proxy: bigger palm = closer
// Returns a scale factor 0.5..2.0
// =============================================
function estimateDepth(landmarks) {
    // Distance from wrist (0) to middle finger MCP (9)
    const wrist = landmarks[0];
    const mcp = landmarks[9];
    const dx = wrist.x - mcp.x;
    const dy = wrist.y - mcp.y;
    const palmSize = Math.sqrt(dx * dx + dy * dy);
    // Typical palm size in normalized coords when hand is ~50cm from camera: ~0.2
    // Close (~30cm): ~0.35, Far (~100cm): ~0.1
    const refSize = 0.2;
    return Math.max(0.4, Math.min(2.5, palmSize / refSize));
}

// =============================================
// DETECTION LOOP
// =============================================
function detectHands() {
    if (!webcamRunning) return;

    const now = performance.now();

    // FPS
    frameCount++;
    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
        fpsCounter.textContent = fps + " FPS";
    }

    // HUD time
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    hudTime.textContent =
        String(Math.floor(elapsed / 3600)).padStart(2, "0") + ":" +
        String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0") + ":" +
        String(elapsed % 60).padStart(2, "0");

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Detect
    const results = handLandmarker.detectForVideo(video, now);
    animationFrame++;

    let frameActive = false;

    if (results.landmarks && results.landmarks.length > 0) {
        const n = results.landmarks.length;
        handsDetected.textContent = `${n} HAND${n > 1 ? "S" : ""} DETECTED`;

        if (n >= 2) {
            const hand1 = results.landmarks[0];
            const hand2 = results.landmarks[1];
            const depth1 = estimateDepth(hand1);
            const depth2 = estimateDepth(hand2);
            const avgDepth = (depth1 + depth2) / 2;

            drawFrameEffect(hand1, hand2, avgDepth);
            frameActive = true;
            frameStatus.textContent = "FRAME: ACTIVE ⚡";
            frameStatus.style.color = "#00ffc8";
        } else {
            frameStatus.textContent = "FRAME: 1 HAND — NEED 2";
            frameStatus.style.color = "#ffaa33";
        }
    } else {
        handsDetected.textContent = "NO HANDS DETECTED";
        frameStatus.textContent = "FRAME: STANDBY";
        frameStatus.style.color = "#8888aa";
    }

    // Burst on activation
    if (frameActive && !wasFrameActive) spawnBurst();
    wasFrameActive = frameActive;

    // Particles
    updateParticles();

    requestAnimationFrame(detectHands);
}

// =============================================
// BUILD QUAD from two hands
// Returns 4 corners in order: TL, TR, BR, BL
// =============================================
function buildQuad(hand1, hand2) {
    const w = canvas.width;
    const h = canvas.height;

    // For each hand: index tip (8) is "top", thumb tip (4) is "bottom"
    const h1_top = { x: (1 - hand1[8].x) * w, y: hand1[8].y * h };
    const h1_bot = { x: (1 - hand1[4].x) * w, y: hand1[4].y * h };
    const h2_top = { x: (1 - hand2[8].x) * w, y: hand2[8].y * h };
    const h2_bot = { x: (1 - hand2[4].x) * w, y: hand2[4].y * h };

    // Figure out which hand is on the left by comparing index tip x
    let leftTop, leftBot, rightTop, rightBot;
    if (h1_top.x < h2_top.x) {
        leftTop = h1_top; leftBot = h1_bot;
        rightTop = h2_top; rightBot = h2_bot;
    } else {
        leftTop = h2_top; leftBot = h2_bot;
        rightTop = h1_top; rightBot = h1_bot;
    }

    // Quad corners: TL, TR, BR, BL
    return [leftTop, rightTop, rightBot, leftBot];
}

function smoothQuadPoints(raw) {
    if (!smoothQuad) {
        smoothQuad = raw.map((p) => ({ x: p.x, y: p.y }));
    } else {
        for (let i = 0; i < 4; i++) {
            smoothQuad[i].x += (raw[i].x - smoothQuad[i].x) * SMOOTH;
            smoothQuad[i].y += (raw[i].y - smoothQuad[i].y) * SMOOTH;
        }
    }
    return smoothQuad;
}

// =============================================
// DRAW FRAME EFFECT
// =============================================
function drawFrameEffect(hand1, hand2, depth) {
    const rawQuad = buildQuad(hand1, hand2);
    const quad = smoothQuadPoints(rawQuad);

    // Quad dimensions for internal effects
    const minX = Math.min(quad[0].x, quad[1].x, quad[2].x, quad[3].x);
    const maxX = Math.max(quad[0].x, quad[1].x, quad[2].x, quad[3].x);
    const minY = Math.min(quad[0].y, quad[1].y, quad[2].y, quad[3].y);
    const maxY = Math.max(quad[0].y, quad[1].y, quad[2].y, quad[3].y);
    const qw = maxX - minX;
    const qh = maxY - minY;

    if (qw < 25 || qh < 25) return;

    ctx.save();

    switch (currentMode) {
        case "hologram":
            drawHologram(quad, qw, qh, depth);
            break;
        case "neon":
            drawNeon(quad, qw, qh, depth);
            break;
        case "matrix":
            drawMatrix(quad, qw, qh, depth);
            break;
        case "plasma":
            drawPlasma(quad, qw, qh, depth);
            break;
    }

    // Corner brackets + energy dots
    drawCorners(quad, depth);

    // Spawn particles along edges
    if (animationFrame % 3 === 0) spawnEdgeParticle(quad);

    ctx.restore();
}

// =============================================
// Helper: draw quad path
// =============================================
function quadPath(q) {
    ctx.beginPath();
    ctx.moveTo(q[0].x, q[0].y);
    ctx.lineTo(q[1].x, q[1].y);
    ctx.lineTo(q[2].x, q[2].y);
    ctx.lineTo(q[3].x, q[3].y);
    ctx.closePath();
}

function quadCenter(q) {
    return {
        x: (q[0].x + q[1].x + q[2].x + q[3].x) / 4,
        y: (q[0].y + q[1].y + q[2].y + q[3].y) / 4,
    };
}

// =============================================
// EFFECT: HOLOGRAM
// =============================================
function drawHologram(quad, qw, qh, depth) {
    const t = animationFrame * 0.03;
    const center = quadCenter(quad);
    // Opacity scales with depth — closer = more opaque
    const depthAlpha = Math.min(1, 0.6 + depth * 0.2);

    // --- Fill ---
    quadPath(quad);
    const grad = ctx.createLinearGradient(quad[0].x, quad[0].y, quad[2].x, quad[2].y);
    grad.addColorStop(0, `rgba(0, 255, 200, ${(0.13 + 0.04 * Math.sin(t)) * depthAlpha})`);
    grad.addColorStop(0.4, `rgba(80, 120, 255, ${(0.09 + 0.03 * Math.sin(t + 1)) * depthAlpha})`);
    grad.addColorStop(1, `rgba(0, 255, 200, ${(0.13 + 0.04 * Math.sin(t + 2)) * depthAlpha})`);
    ctx.fillStyle = grad;
    ctx.fill();

    // --- Scan lines (clipped to quad) ---
    ctx.save();
    quadPath(quad);
    ctx.clip();

    // Horizontal scan lines — draw fewer for perf
    const minY = Math.min(quad[0].y, quad[1].y, quad[2].y, quad[3].y);
    const maxY = Math.max(quad[0].y, quad[1].y, quad[2].y, quad[3].y);
    const minX = Math.min(quad[0].x, quad[1].x, quad[2].x, quad[3].x);
    const maxX = Math.max(quad[0].x, quad[1].x, quad[2].x, quad[3].x);

    ctx.strokeStyle = `rgba(0, 255, 200, ${0.06 * depthAlpha})`;
    ctx.lineWidth = 1;
    for (let ly = minY; ly < maxY; ly += 4) {
        ctx.beginPath();
        ctx.moveTo(minX, ly);
        ctx.lineTo(maxX, ly);
        ctx.stroke();
    }

    // 2 bright sweep lines
    for (let i = 0; i < 2; i++) {
        const sweep = minY + ((animationFrame * 2 + i * qh * 0.5) % qh);
        ctx.strokeStyle = `rgba(0, 255, 200, ${(0.22 + 0.1 * Math.sin(t + i)) * depthAlpha})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(minX, sweep);
        ctx.lineTo(maxX, sweep);
        ctx.stroke();
    }

    // HUD text
    ctx.font = "11px 'Share Tech Mono', monospace";
    ctx.fillStyle = `rgba(0, 255, 200, ${0.4 * depthAlpha})`;
    ctx.textAlign = "left";
    const info = [
        "SYS://NEXUS_FRAME",
        `DIM: ${Math.round(qw)}×${Math.round(qh)}`,
        `DEPTH: ${depth.toFixed(2)}x`,
        "STATUS: TRACKING ✓",
    ];
    for (let i = 0; i < info.length; i++) {
        ctx.fillText(info[i], minX + 10, minY + 18 + i * 15);
    }

    // Timestamp at bottom center
    if (animationFrame % 40 < 32) {
        ctx.textAlign = "center";
        ctx.fillStyle = `rgba(0, 255, 200, ${0.3 * depthAlpha})`;
        ctx.fillText(new Date().toISOString().substr(11, 12), center.x, maxY - 8);
    }

    ctx.restore();

    // --- Border ---
    quadPath(quad);
    ctx.strokeStyle = `rgba(0, 255, 200, ${(0.4 + 0.15 * Math.sin(t)) * depthAlpha})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

// =============================================
// EFFECT: NEON
// =============================================
function drawNeon(quad, qw, qh, depth) {
    const t = animationFrame * 0.04;
    const center = quadCenter(quad);
    const da = Math.min(1, 0.6 + depth * 0.2);

    // Fill
    quadPath(quad);
    ctx.fillStyle = `rgba(123, 97, 255, ${(0.10 + 0.05 * Math.sin(t)) * da})`;
    ctx.fill();

    // Multi-layer borders (no shadowBlur — use double-stroke trick for glow)
    // Thick soft border
    quadPath(quad);
    ctx.strokeStyle = `rgba(255, 97, 216, ${0.25 * da})`;
    ctx.lineWidth = 8;
    ctx.stroke();

    // Medium
    quadPath(quad);
    ctx.strokeStyle = `rgba(255, 97, 216, ${(0.55 + 0.2 * Math.sin(t)) * da})`;
    ctx.lineWidth = 3;
    ctx.stroke();

    // Thin bright
    quadPath(quad);
    ctx.strokeStyle = `rgba(255, 220, 255, ${0.7 * da})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Crosshair at center
    const cs = 15 + 5 * Math.sin(t * 2);
    ctx.strokeStyle = `rgba(255, 97, 216, ${0.35 * da})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(center.x - cs, center.y);
    ctx.lineTo(center.x + cs, center.y);
    ctx.moveTo(center.x, center.y - cs);
    ctx.lineTo(center.x, center.y + cs);
    ctx.stroke();

    // Rotating arc
    ctx.beginPath();
    ctx.arc(center.x, center.y, cs * 2, t, t + 0.8);
    ctx.strokeStyle = `rgba(255, 97, 216, ${0.3 * da})`;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(center.x, center.y, cs * 2, t + Math.PI, t + Math.PI + 0.8);
    ctx.stroke();

    // Label
    ctx.save();
    quadPath(quad);
    ctx.clip();
    ctx.font = "bold 11px 'Share Tech Mono', monospace";
    ctx.fillStyle = `rgba(255, 97, 216, ${0.4 * da})`;
    ctx.textAlign = "center";
    const minY = Math.min(quad[0].y, quad[1].y, quad[2].y, quad[3].y);
    ctx.fillText("◈ NEON FRAME ◈", center.x, minY + 18);
    ctx.restore();
}

// =============================================
// EFFECT: MATRIX
// =============================================
function drawMatrix(quad, qw, qh, depth) {
    const da = Math.min(1, 0.6 + depth * 0.2);
    const chars = "アイウエオカキクケコサシスセソ0123456789";
    const fontSize = 13;

    // Dark fill
    quadPath(quad);
    ctx.fillStyle = `rgba(0, 12, 0, ${0.18 * da})`;
    ctx.fill();

    // Clip to quad
    ctx.save();
    quadPath(quad);
    ctx.clip();

    const minX = Math.min(quad[0].x, quad[1].x, quad[2].x, quad[3].x);
    const maxX = Math.max(quad[0].x, quad[1].x, quad[2].x, quad[3].x);
    const minY = Math.min(quad[0].y, quad[1].y, quad[2].y, quad[3].y);
    const maxY = Math.max(quad[0].y, quad[1].y, quad[2].y, quad[3].y);

    ctx.font = `${fontSize}px 'Share Tech Mono', monospace`;

    // Only draw ~15 columns max for performance
    const colW = Math.max(fontSize, qw / 15);
    const numCols = Math.min(15, Math.ceil(qw / colW));

    for (let i = 0; i < numCols; i++) {
        const cx = minX + i * colW + colW / 2;
        // Each column scrolls at different speed based on column index
        const speed = 1.5 + (i * 7 % 5);
        const offset = (animationFrame * speed) % (qh + fontSize * 10);
        const headY = minY + offset;

        // Head char (bright)
        if (headY >= minY && headY <= maxY) {
            const ch = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillStyle = `rgba(120, 255, 160, ${0.9 * da})`;
            ctx.fillText(ch, cx, headY);
        }

        // Trail (6 chars fading)
        for (let t = 1; t <= 6; t++) {
            const ty = headY - t * fontSize;
            if (ty < minY || ty > maxY) continue;
            const ch = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillStyle = `rgba(0, 255, 80, ${Math.max(0, (0.5 - t * 0.07)) * da})`;
            ctx.fillText(ch, cx, ty);
        }
    }

    ctx.restore();

    // Green border (thick for fake glow, no shadowBlur)
    quadPath(quad);
    ctx.strokeStyle = `rgba(0, 255, 100, ${0.2 * da})`;
    ctx.lineWidth = 6;
    ctx.stroke();
    quadPath(quad);
    ctx.strokeStyle = `rgba(0, 255, 100, ${0.55 * da})`;
    ctx.lineWidth = 2;
    ctx.stroke();
}

// =============================================
// EFFECT: PLASMA
// =============================================
function drawPlasma(quad, qw, qh, depth) {
    const t = animationFrame * 0.025;
    const center = quadCenter(quad);
    const da = Math.min(1, 0.6 + depth * 0.2);

    ctx.save();
    quadPath(quad);
    ctx.clip();

    const minX = Math.min(quad[0].x, quad[1].x, quad[2].x, quad[3].x);
    const minY = Math.min(quad[0].y, quad[1].y, quad[2].y, quad[3].y);

    // 3 plasma blobs
    const blobs = [
        { color1: "255, 97, 216", color2: "123, 97, 255", phase: 0 },
        { color1: "0, 255, 200", color2: "61, 155, 255", phase: 2.1 },
        { color1: "255, 140, 66", color2: "255, 97, 216", phase: 4.2 },
    ];

    for (const blob of blobs) {
        const angle = t + blob.phase;
        const bx = center.x + Math.cos(angle) * qw * 0.25;
        const by = center.y + Math.sin(angle) * qh * 0.25;
        const r = Math.max(qw, qh) * 0.5;

        const grad = ctx.createRadialGradient(bx, by, 0, bx, by, r);
        grad.addColorStop(0, `rgba(${blob.color1}, ${(0.18 + 0.06 * Math.sin(t + blob.phase)) * da})`);
        grad.addColorStop(0.5, `rgba(${blob.color2}, ${0.04 * da})`);
        grad.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = grad;
        ctx.fillRect(minX, minY, qw, qh);
    }

    // Ripple
    const rippleR = (animationFrame * 1.5) % 150;
    const rippleA = Math.max(0, 0.2 - rippleR * 0.0013);
    ctx.beginPath();
    ctx.arc(center.x, center.y, rippleR, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255, 97, 216, ${rippleA * da})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.restore();

    // Rainbow border
    const hue = (animationFrame * 2.5) % 360;
    // Thick for glow
    quadPath(quad);
    ctx.strokeStyle = `hsla(${hue}, 100%, 68%, ${0.2 * da})`;
    ctx.lineWidth = 7;
    ctx.stroke();
    quadPath(quad);
    ctx.strokeStyle = `hsla(${hue}, 100%, 68%, ${0.6 * da})`;
    ctx.lineWidth = 2;
    ctx.stroke();
}

// =============================================
// CORNERS (brackets + dots)
// =============================================
function drawCorners(quad, depth) {
    const da = Math.min(1, 0.6 + depth * 0.2);
    const modeC = {
        hologram: [0, 255, 200],
        neon: [255, 97, 216],
        matrix: [0, 255, 100],
        plasma: [255, 140, 66],
    };
    const c = modeC[currentMode] || modeC.hologram;
    const pulse = 0.7 + 0.3 * Math.sin(animationFrame * 0.06);
    const a = pulse * da;

    // For each corner, draw an L-bracket pointing inward
    for (let i = 0; i < 4; i++) {
        const curr = quad[i];
        const prev = quad[(i + 3) % 4];
        const next = quad[(i + 1) % 4];

        // Direction toward prev and next
        const toPrev = normalize(prev.x - curr.x, prev.y - curr.y);
        const toNext = normalize(next.x - curr.x, next.y - curr.y);
        const bLen = 22;

        ctx.strokeStyle = `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(curr.x + toPrev.x * bLen, curr.y + toPrev.y * bLen);
        ctx.lineTo(curr.x, curr.y);
        ctx.lineTo(curr.x + toNext.x * bLen, curr.y + toNext.y * bLen);
        ctx.stroke();

        // Dot at corner
        ctx.beginPath();
        ctx.arc(curr.x, curr.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
        ctx.fill();

        // Soft glow ring (cheap: no shadowBlur)
        ctx.beginPath();
        ctx.arc(curr.x, curr.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${0.08 * a})`;
        ctx.fill();
    }

    // Energy dots moving along edges
    const numDots = 8;
    const t = animationFrame * 0.05;
    for (let i = 0; i < numDots; i++) {
        const progress = ((t + (i / numDots)) % 1);
        // Walk along the 4 edges
        const totalP = progress * 4;
        const edgeIdx = Math.floor(totalP) % 4;
        const edgeFrac = totalP - Math.floor(totalP);
        const p1 = quad[edgeIdx];
        const p2 = quad[(edgeIdx + 1) % 4];
        const px = p1.x + (p2.x - p1.x) * edgeFrac;
        const py = p1.y + (p2.y - p1.y) * edgeFrac;

        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${(0.5 + 0.4 * Math.sin(t * 5 + i)) * da})`;
        ctx.fill();
    }
}

function normalize(x, y) {
    const len = Math.sqrt(x * x + y * y) || 1;
    return { x: x / len, y: y / len };
}

// =============================================
// PARTICLES
// =============================================
function spawnEdgeParticle(quad) {
    const edge = Math.floor(Math.random() * 4);
    const p1 = quad[edge];
    const p2 = quad[(edge + 1) % 4];
    const frac = Math.random();
    const px = p1.x + (p2.x - p1.x) * frac;
    const py = p1.y + (p2.y - p1.y) * frac;

    const mc = {
        hologram: "0, 255, 200",
        neon: Math.random() > 0.5 ? "255, 97, 216" : "123, 97, 255",
        matrix: "0, 255, 100",
        plasma: `${128 + Math.floor(Math.random() * 127)}, ${Math.floor(Math.random() * 200)}, 255`,
    };

    particles.push({
        x: px, y: py,
        vx: (Math.random() - 0.5) * 1.5,
        vy: -0.5 - Math.random() * 1.5,
        life: 1,
        decay: 0.02 + Math.random() * 0.02,
        size: 1 + Math.random() * 2,
        rgb: mc[currentMode] || mc.hologram,
    });
}

function spawnBurst() {
    if (!smoothQuad) return;
    const c = quadCenter(smoothQuad);
    for (let i = 0; i < 25; i++) {
        const angle = (Math.PI * 2 * i) / 25;
        const speed = 2 + Math.random() * 4;
        particles.push({
            x: c.x, y: c.y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 1,
            decay: 0.02 + Math.random() * 0.01,
            size: 1.5 + Math.random() * 2.5,
            rgb: Math.random() > 0.5 ? "0, 255, 200" : "123, 97, 255",
        });
    }
}

function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.97;
        p.vy *= 0.97;
        p.life -= p.decay;

        if (p.life <= 0) {
            particles.splice(i, 1);
            continue;
        }

        ctx.globalAlpha = p.life * 0.7;
        ctx.fillStyle = `rgba(${p.rgb}, ${p.life})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (particles.length > 120) particles.splice(0, particles.length - 120);
}

// =============================================
// MODE SWITCHING
// =============================================
modeSwitcher.addEventListener("click", (e) => {
    const btn = e.target.closest(".mode-btn");
    if (!btn) return;
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    particles = [];
});

// =============================================
// TIMER SELECTOR
// =============================================
let timerDelay = 0; // seconds (0 = off)

const timerSelector = document.getElementById("timerSelector");
timerSelector.addEventListener("click", (e) => {
    const btn = e.target.closest(".timer-btn");
    if (!btn) return;
    document.querySelectorAll(".timer-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    timerDelay = parseInt(btn.dataset.timer, 10);
});

// =============================================
// COUNTDOWN + SCREENSHOT
// =============================================
const countdownOverlay = document.getElementById("countdownOverlay");
const countdownNumber = document.getElementById("countdownNumber");
let countdownActive = false;

function doCapture() {
    const sc = document.getElementById("screenshotCanvas");
    sc.width = canvas.width;
    sc.height = canvas.height;
    const sctx = sc.getContext("2d");
    sctx.save();
    sctx.translate(canvas.width, 0);
    sctx.scale(-1, 1);
    sctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    sctx.restore();
    sctx.drawImage(canvas, 0, 0);
    const link = document.createElement("a");
    link.download = `nexus_frame_${Date.now()}.png`;
    link.href = sc.toDataURL("image/png");
    link.click();
    cameraWrapper.classList.add("flash");
    setTimeout(() => cameraWrapper.classList.remove("flash"), 350);
}

function startCountdown(seconds) {
    if (countdownActive) return;
    countdownActive = true;
    countdownOverlay.classList.remove("hidden");
    let remaining = seconds;
    countdownNumber.textContent = remaining;

    const interval = setInterval(() => {
        remaining--;
        if (remaining <= 0) {
            clearInterval(interval);
            countdownOverlay.classList.add("hidden");
            countdownActive = false;
            doCapture();
        } else {
            countdownNumber.textContent = remaining;
        }
    }, 1000);
}

captureBtn.addEventListener("click", () => {
    if (countdownActive) return;
    if (timerDelay > 0) {
        startCountdown(timerDelay);
    } else {
        doCapture();
    }
});

// =============================================
// VIDEO RECORDING
// =============================================
const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const recIcon = document.getElementById("recIcon");
const recordingIndicator = document.getElementById("recordingIndicator");
const recTimeEl = document.getElementById("recTime");
const compositeCanvas = document.getElementById("compositeCanvas");

let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recStartTime = 0;
let recTimerInterval = null;

// Composite canvas: draws video + effects together for recording
function startCompositing() {
    compositeCanvas.width = canvas.width;
    compositeCanvas.height = canvas.height;
}

function compositeFrame() {
    if (!isRecording) return;
    const cctx = compositeCanvas.getContext("2d");
    // Draw mirrored video
    cctx.save();
    cctx.translate(compositeCanvas.width, 0);
    cctx.scale(-1, 1);
    cctx.drawImage(video, 0, 0, compositeCanvas.width, compositeCanvas.height);
    cctx.restore();
    // Draw effects overlay
    cctx.drawImage(canvas, 0, 0);
    requestAnimationFrame(compositeFrame);
}

function startRecording() {
    startCompositing();

    // Get stream from composite canvas
    const stream = compositeCanvas.captureStream(30); // 30fps

    // Try to add audio from microphone (optional, won't fail if denied)
    recordedChunks = [];

    // Determine supported MIME type
    const mimeTypes = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
        "video/mp4",
    ];
    let mimeType = "";
    for (const mt of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mt)) {
            mimeType = mt;
            break;
        }
    }

    mediaRecorder = new MediaRecorder(stream, {
        mimeType: mimeType || undefined,
        videoBitsPerSecond: 4000000, // 4 Mbps
    });

    mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
            recordedChunks.push(e.data);
        }
    };

    mediaRecorder.onstop = () => {
        const ext = mimeType.includes("mp4") ? "mp4" : "webm";
        const blob = new Blob(recordedChunks, { type: mimeType || "video/webm" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.download = `nexus_frame_${Date.now()}.${ext}`;
        link.href = url;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
    };

    mediaRecorder.start(100); // collect data every 100ms
    isRecording = true;
    recStartTime = Date.now();

    // Start compositing loop
    compositeFrame();

    // UI updates
    recordBtn.classList.add("recording");
    recordLabel.textContent = "STOP";
    recordingIndicator.classList.remove("hidden");

    // Recording timer display
    recTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
        const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const s = String(elapsed % 60).padStart(2, "0");
        recTimeEl.textContent = `${m}:${s}`;
    }, 500);
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    isRecording = false;
    clearInterval(recTimerInterval);

    // UI updates
    recordBtn.classList.remove("recording");
    recordLabel.textContent = "RECORD";
    recordingIndicator.classList.add("hidden");
    recTimeEl.textContent = "00:00";
}

recordBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// =============================================
// INIT
// =============================================
initHandLandmarker().catch((err) => {
    console.error("Init failed:", err);
    loadingStatus.textContent = "Error: " + err.message;
});

