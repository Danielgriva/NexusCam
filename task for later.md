# 🚀 NexusCam & NexusFrame: Next Generation Blueprint

This document serves as the master prompt and blueprint for the next phase of development. It outlines 13 core upgrades requested by the user, plus an exclusive AI-proposed feature, aiming to transform the project into a highly polished, open-source, pro-grade application. 

*Core Philosophy 1: Ultimate User Freedom. Every single feature outlined below must have a master toggle switch in the settings. The user is in complete control of what features are active.*

*Core Philosophy 2: Extreme Optimization (Low-End Devices). We must anticipate users with very low specs, old CPUs, and limited RAM. Every visual effect, neural network model, and background process must be strictly memory-profiled and heavily multithreaded to guarantee a smooth, lag-free experience on budget hardware.*

## Phase 1: Core Functionality & Cleanliness
**1. Clean Media Capture & Organized Gallery**
- **Objective**: Ensure that when a snapshot or video is recorded, **zero** UI elements (buttons, FPS counters, "tracker locked" text) are visible in the final saved media, and the files are cleanly organized on the user's phone. 
- **Details**: 
  - Implement a dual-buffer rendering system. One buffer draws the raw frame + holographic effects (saved to disk), while the second buffer overlays the UI buttons and diagnostics (displayed to the user). The FPS counter should be removed or minimized to a microscopic corner element.
  - **Dedicated Gallery Album**: When media is saved on the Android device, the app must automatically create a new dedicated album/folder (e.g., "NexusCam") in the phone's gallery so everything is organized, instead of dumping files into the general camera roll.

**2. PC Window & Hotkey Management**
- **Objective**: Make the PC application behave like a standard Windows application.
- **Details**:
  - Fix the window 'X' (close button) event so it properly terminates the Python process without requiring the `ESC` key.
  - Bind the `F11` key to toggle borderless fullscreen / maximized mode.
  - Relocate the `[SNAPSHOT]` and `[RECORD]` buttons to the main HUD (always visible), mimicking standard camera apps, so users don't have to open the settings menu to capture media.

## Phase 2: Android "Pro Camera" & Connectivity
**3. Pro Camera Mode (Android)**
- **Objective**: Expand the Android APK far beyond a simple streaming client.
- **Details**: Utilize `Camera2` / `CameraX` advanced extensions to build a full "Pro Camera Mode". 
  - Add manual sliders for **Contrast, ISO (AV), White Balance, and Exposure**.
  - Implement smooth, pinch-to-zoom functionality.
  - Expose all available hardware camera lenses (Ultrawide, Telephoto).

**4. Frictionless Connectivity & Smart Parsing**
- **Objective**: Eliminate the hassle of typing long IP addresses.
- **Details**: 
  - The Android camera feed should remain paused/blacked out until an active connection from the PC is established to save battery and reduce thermal throttling.
  - **Auto-Pairing**: Implement a QR Code scanner in the Android app that scans a QR code displayed on the PC screen to instantly link them.
  - **Smart Input**: If manual entry is used, the PC UI should only ask for the base IP digits (e.g., `192.168.1.15`). The code will automatically prepend `http://` and append `:8080/video`.

## Phase 3: The Standalone Android FX App
**5. The 3rd APK: On-Device Edge Processing**
- **Objective**: Port the entire OpenCV/MediaPipe holographic engine directly to a standalone Android app for users without a PC.
- **Details**: This requires migrating the Python effect pipeline to Kotlin/C++ using the MediaPipe Android SDK. 
  - It must run completely on the edge (on-device) with heavy multi-threading to ensure video recording is buttery smooth and lag-free.
  - This app will inherently include the "Pro Camera Mode" features from Phase 2.

## Phase 4: Ultimate UI/UX Overhaul & Deep Customization
**6. Professional, High-End Design Language**
- **Objective**: Strip away the "sloppy AI design" and implement a breathtaking, professional UI.
- **Details**: 
  - Conduct research on high-end camera UI/UX (e.g., Sony Alpha interfaces, RED cinema cameras, or modern cyberpunk HUDs like in *Cyberpunk 2077*).
  - Select a strict, harmonious color palette (e.g., deep OLED blacks, subtle charcoal greys, and striking neon cyan/magenta accents).
  - Ensure all text, padding, and alignments are mathematically perfect.

**7. Deep Customization Engine & Effect Slots**
- **Objective**: Give the user absolute, granular creative control over how their effects look, down to the pixel.
- **Details**: 
  - **Granular Controls**: Introduce dedicated sliders for **Line Thickness, Particle Color, Transparency/Opacity, and Bloom/Glow Intensity**. 
  - **Custom Backgrounds**: Allow users to replace the real-world background with solid colors, gradients, or custom images (chroma-key style background replacement).
  - **Holographic Quotes/Text**: Add a text input field so users can type a custom quote or message. This text will be rendered in a futuristic font floating perfectly in the center of the hologram.
  - **Animation Speed**: Sliders to control the rotation speed and flicker rate of the holograms.
  - Create **Slots 1 through 9** where users can save, load, and tweak all these custom parameters.
  - The PC settings UI must be dramatically simplified into an understandable, tabbed layout to accommodate these new sliders gracefully.

## Phase 5: Next-Gen Holographic Rendering
**8. Spatial Awareness & X-Ray Shaders**
- **Objective**: Make the hologram interact with the real world.
- **Details**: 
  - **Holographic Masking**: When the user moves their face or an object *behind* the holographic projection, the object itself should inherit a glowing, glitchy, holographic texture.
  - **X-Ray Mode**: Implement a shader that acts as a localized X-Ray. Whatever the hologram physically covers on screen is rendered as a stylized wireframe or thermal map, creating the illusion of scanning the environment.

**9. Hologram-Exclusive Capture**
- **Objective**: A completely new media format.
- **Details**: Add a capture mode (on both PC and Phone) that ignores the background entirely. It crops the image down to the exact boundaries of the drawn hologram, masking out the real world, and saves *only* the holographic panel and whatever is interacting inside it as a transparent PNG or masked video.

## Phase 6: Security & Open Source
**10. "NASA-Grade" Privacy & Security**
- **Objective**: Guarantee absolute data sovereignty.
- **Details**: 
  - Audit the codebase to ensure **zero** external network calls, telemetry, or crash reporting.
  - The app must only bind to the `localhost` or local WLAN `192.168.x.x` subnet.
  - Explicitly block internet access in the Android Manifest (removing `<uses-permission android:name="android.permission.INTERNET" />` if not strictly required for local LAN, or isolating the NanoHTTPD server tightly).

**11. Open-Source Github Preparation**
- **Objective**: Release the project to the world.
- **Details**: 
  - Modularize the codebase into clean directories (`/android`, `/pc_engine`, `/effects`).
  - Write a comprehensive `README.md` with installation instructions, contribution guidelines, and architecture diagrams.
  - Add an MIT or GPLv3 License.

---

## 🌟 Bonus Features: Spatial Control & Audio
**12. Minority Report UI Navigation (Toggleable)**
- **Objective**: Since we are already running high-precision MediaPipe hand tracking, why touch a mouse or screen at all?
- **Details**: 
  - Implement a **Gesture Control Engine**. 
  - **Pinch and Drag**: Zoom the camera by pinching two hands together and pulling apart.
  - **Swipe**: Swipe your hand left or right in the air to cycle between the 9 custom effect slots.
  - **Peace Sign (V)**: Hold up a peace sign for 2 seconds to trigger a 3-second countdown for a clean snapshot.
  - *Must include a master ON/OFF toggle in settings so users are never forced to use gestures if they prefer manual control.*

**13. Built-In Music Streamer & Audio Reactivity**
- **Objective**: Make the visuals dance to the music without "listening" to your microphone or other apps.
- **Details**: 
  - Instead of listening to the PC's system audio or microphone, we will build an **Internal Music Player** directly into the app. 
  - *How do open-source apps do this for free?* Open-source music apps (like Spotube or ViMusic) use public YouTube Music audio streams mapped to Spotify metadata. We will integrate a similar backend (like `ytmusicapi`). This allows users to search for any song or playlist directly inside NexusFrame and stream the high-quality audio for free, completely internally!
  - **For Videos (Reactivity)**: The music will physically drive the hologram's glow intensity, line thickness, and particle speed, reacting dynamically to the bass and treble. *This will be heavily optimized using multithreading to ensure absolutely zero lag while recording.*
  - **For Snapshots (Lyric & Art Integration)**: 
    - Search for a song, pick specific lyrics, and the app will embed the album art and lyrics into a stylish widget on your screen.
    - **Draggable & Removable**: You can grab the lyric/thumbnail widget and drag it anywhere on the screen. While dragging, a "Trash Can" icon will appear; dropping the widget on the trash can deletes it.
    - **Widget Customization**: The widget will have sleek, curved edges. By default, its background color will automatically match the dominant color of the song's album art (thumbnail).
    - **Custom Thumbnails**: Users can replace the album art with their own images (requires local file storage permissions).
  - *Must include an ON/OFF toggle for the visualizer and a full search UI for the built-in music player.*
