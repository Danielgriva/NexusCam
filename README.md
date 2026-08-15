# 🌌 NexusFrame & NexusCam

### *Next-Generation Edge-AI Spatial Holographic Camera System*

**NexusFrame** is an open-source, privacy-first, real-time holographic camera platform that bridges your Android phone and desktop PC to create futuristic augmented-reality media.

By combining low-latency wireless video streaming with cutting-edge MediaPipe hand tracking, NexusFrame projects custom, fully interactive holographic panels between your fingertips — transforming simple hand gestures into sci-fi visual experiences.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖐️ **AI Hand Tracking** | Sub-millimeter precision tracking via MediaPipe. Two-hand detection creates holographic quads in real-time. |
| 📱 **NexusCam (Android)** | Turn your phone into a wireless Pro Camera with manual ISO, White Balance, Exposure, Contrast, Flash, and Zoom. |
| 🎨 **4 Effect Modes** | Hologram, Neon, Matrix, and Plasma — each fully customizable. |
| ⚙️ **Deep Customization** | Sliders for Line Thickness, Transparency, Glow Intensity, Animation Speed. 9 saveable preset slots. |
| 🔮 **X-Ray & Holo Masking** | Toggle X-Ray thermal vision or holographic glitch textures inside the hologram region. |
| ✋ **Gesture Control** | Pinch-to-zoom, swipe to cycle effects, peace sign to trigger snapshot — all toggleable. |
| 📸 **Clean Capture** | Dual-buffer rendering ensures zero UI clutter in saved photos and videos. |
| 🎯 **Hologram-Only Capture** | Save just the hologram region as a transparent PNG. |
| 💬 **Holographic Text** | Type a custom quote or message that floats inside the hologram. |
| 🔒 **Privacy-First** | 100% offline. Zero cloud, zero telemetry. Your video never leaves your local network. |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** with pip
- **Windows 10/11** (tested)
- A webcam or Android phone running NexusCam

### 1. Clone & Install

```bash
git clone https://github.com/Danielgriva/nexuscam.git
cd nexuscam
pip install opencv-python numpy mediapipe
```

### 2. Download the Hand Tracking Model

Download `hand_landmarker.task` from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place it in the project root.

### 3. Run NexusFrame (PC)

```bash
python nexus_engine.py
```

### 4. (Optional) Run NexusCam (Android)

1. Build the APK from `NexusCam/` using Android Studio or Gradle
2. Install on your Android phone
3. Both devices must be on the same WiFi network
4. In NexusFrame, click **Settings > Camera > Connect Phone** and enter the IP shown on your phone

---

## 🎮 Controls

### Keyboard Shortcuts
| Key | Action |
|---|---|
| `1-4` | Switch effect mode |
| `T` | Cycle timer (OFF/3s/5s/10s) |
| `SPACE` | Take photo |
| `R` | Start/Stop recording |
| `F11` | Toggle fullscreen |
| `ESC` / `Q` | Quit |

### Mouse
- Click the **⚙ gear icon** (top-left) to open Settings
- **PHOTO** and **REC** buttons are always visible at the bottom

### Gesture Control (toggleable in Settings > Camera)
| Gesture | Action |
|---|---|
| ✌️ Peace sign (2s hold) | Triggers 3s countdown snapshot |
| 👋 Swipe left/right | Cycle through effect modes |

---

## 📁 Project Structure

```
nexuscam/
├── nexus_engine.py      # Main PC application (OpenCV + MediaPipe)
├── fx_pipeline.py       # Effect rendering engine (4 modes)
├── nexus_frame.py       # Legacy standalone version
├── index.html           # Web-based demo UI
├── app.js               # Web app logic
├── style.css            # Web app styles
├── NexusCam/            # Android app (Kotlin + CameraX)
│   ├── app/src/main/
│   │   ├── java/.../MainActivity.kt
│   │   ├── java/.../MjpegServer.kt
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── .gitignore
├── LICENSE              # MIT License
└── README.md
```

---

## 🔒 Privacy & Security

NexusFrame is designed with a **zero-trust, offline-first** architecture:

- ❌ No cloud services
- ❌ No telemetry or analytics
- ❌ No external API calls
- ✅ All processing happens locally on your CPU/GPU
- ✅ Phone-to-PC streaming uses local WiFi only (never leaves your network)
- ✅ All media is saved locally to your desktop

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-effect`)
3. Commit your changes (`git commit -m 'Add amazing effect'`)
4. Push to the branch (`git push origin feature/amazing-effect`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) — Google's ML framework for hand tracking
- [OpenCV](https://opencv.org/) — Computer vision library
- [CameraX](https://developer.android.com/training/camerax) — Android camera framework
- [NanoHTTPD](https://github.com/NanoHttpd/nanohttpd) — Lightweight HTTP server for Android streaming
