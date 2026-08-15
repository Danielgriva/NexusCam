package com.example.nexuscam

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.SystemClock
import android.text.format.Formatter
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// Color palette
val NeonCyan = Color(0xFF00FFCC)
val NeonMagenta = Color(0xFFFF61D8)
val DarkBg = Color(0xFF0C0C0E)
val PanelBg = Color(0xFF14161C)
val TextDim = Color(0xFF505A64)

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private var server: MjpegServer? = null
    
    private var ipAddress by mutableStateOf("Loading IP...")
    private var hasCameraPermission by mutableStateOf(false)
    private var activeConnections by mutableStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        @Suppress("DEPRECATION")
        val ip = Formatter.formatIpAddress(wifiManager.connectionInfo.ipAddress)
        ipAddress = if (ip == "0.0.0.0") "No WiFi Connection" else "http://$ip:8080/video"
        
        server = MjpegServer(8080)
        server?.onConnectionStateChanged = { connections ->
            activeConnections = connections
        }
        server?.start(fi.iki.elonen.NanoHTTPD.SOCKET_READ_TIMEOUT, false)

        hasCameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            hasCameraPermission = isGranted
            if (!isGranted) {
                Toast.makeText(this, "Camera permission required.", Toast.LENGTH_LONG).show()
            }
        }

        if (!hasCameraPermission) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    background = DarkBg,
                    primary = NeonCyan,
                    secondary = NeonMagenta,
                    surface = PanelBg
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    NexusCamApp(
                        ipAddress = ipAddress,
                        hasCameraPermission = hasCameraPermission,
                        server = server,
                        activeConnections = activeConnections
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        server?.stop()
        cameraExecutor.shutdown()
    }
}

@Composable
fun NexusCamApp(
    ipAddress: String,
    hasCameraPermission: Boolean,
    server: MjpegServer?,
    activeConnections: Int
) {
    var showProControls by remember { mutableStateOf(false) }
    
    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Header Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(PanelBg)
                    .padding(horizontal = 16.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                        text = "NEXUS CAM",
                        style = MaterialTheme.typography.titleMedium,
                        color = NeonCyan,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 2.sp
                    )
                    Text(
                        text = ipAddress,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextDim,
                        fontSize = 11.sp
                    )
                }
                // Pro Mode Toggle
                TextButton(
                    onClick = { showProControls = !showProControls }
                ) {
                    Text(
                        text = if (showProControls) "AUTO" else "PRO",
                        color = if (showProControls) NeonMagenta else NeonCyan,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }
            }
            
            // Camera Preview or Standby
            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                if (hasCameraPermission) {
                    if (BuildConfig.FLAVOR == "cameraOnly" && activeConnections == 0) {
                        Text(
                            text = "STANDBY\nWaiting for PC connection...",
                            color = Color.White,
                            modifier = Modifier.align(Alignment.Center),
                            textAlign = TextAlign.Center
                        )
                    } else {
                        var fxResult by remember { mutableStateOf<com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult?>(null) }
                        val particleSystem = remember { ParticleSystem(maxParticles = 250) }
                        var canvasWidth by remember { mutableStateOf(0) }
                        var canvasHeight by remember { mutableStateOf(0) }
                        
                        val context = androidx.compose.ui.platform.LocalContext.current
                        val handTracker = remember {
                            if (BuildConfig.FLAVOR == "standalone") {
                                HandTracker(
                                    context = context,
                                    onResult = { result, w, h ->
                                        fxResult = result
                                    }
                                )
                            } else null
                        }

                        // FX Overlay
                        if (BuildConfig.FLAVOR == "standalone") {
                            androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                                canvasWidth = size.width.toInt()
                                canvasHeight = size.height.toInt()
                                renderNeonFX(
                                    drawScope = this,
                                    result = fxResult,
                                    particles = particleSystem,
                                    w = canvasWidth,
                                    h = canvasHeight,
                                    isFrontCamera = false
                                )
                            }
                        }
                        
                        ProCameraPreview(
                            showProControls = showProControls,
                            onFrame = { proxy ->
                                if (BuildConfig.FLAVOR == "standalone" && handTracker != null) {
                                    // Throttle AI processing to save RAM/CPU
                                    if (System.currentTimeMillis() % 2 == 0L) {
                                        val bmp = imageProxyToBitmap(proxy)
                                        handTracker.processFrame(bmp, SystemClock.uptimeMillis())
                                    }
                                }
                                server?.updateImage(proxy)
                            }
                        )
                    }
                } else {
                    Text(
                        text = "Please allow Camera permissions to stream...",
                        color = Color.White,
                        modifier = Modifier.align(Alignment.Center),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
fun ProCameraPreview(
    showProControls: Boolean,
    onFrame: (ImageProxy) -> Unit
) {
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    var isBackCamera by remember { mutableStateOf(true) }
    var isTorchOn by remember { mutableStateOf(false) }
    var zoomRatio by remember { mutableFloatStateOf(1f) }
    var exposureIndex by remember { mutableIntStateOf(0) }
    
    val cameraRef = remember { mutableStateOf<Camera?>(null) }
    val cameraInfoRef = remember { mutableStateOf<CameraInfo?>(null) }
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Camera View
        key(isBackCamera) {
            AndroidView(
                factory = { context ->
                    val previewView = PreviewView(context)
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                    
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        
                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            .also {
                                it.setAnalyzer(ContextCompat.getMainExecutor(context)) { image ->
                                    onFrame(image)
                                }
                            }
                            
                        val cameraSelector = if (isBackCamera) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
                        
                        try {
                            cameraProvider.unbindAll()
                            val camera = cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageAnalysis
                            )
                            cameraRef.value = camera
                            cameraInfoRef.value = camera.cameraInfo
                            camera.cameraControl.enableTorch(isTorchOn)
                            
                            // Setup Tap-to-Focus
                            previewView.setOnTouchListener { view, event ->
                                if (event.action == android.view.MotionEvent.ACTION_UP) {
                                    val factory = previewView.meteringPointFactory
                                    val point = factory.createPoint(event.x, event.y)
                                    val action = FocusMeteringAction.Builder(point).build()
                                    camera.cameraControl.startFocusAndMetering(action)
                                    view.performClick()
                                }
                                true
                            }
                            
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                        
                    }, ContextCompat.getMainExecutor(context))
                    
                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )
        }
        
        // ── Bottom Controls Bar ──
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            // ── Pro Camera Sliders ──
            if (showProControls) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(DarkBg.copy(alpha = 0.85f))
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    // Zoom Slider
                    ProSlider(
                        label = "ZOOM",
                        value = zoomRatio,
                        valueRange = 1f..10f,
                        valueText = String.format("%.1fx", zoomRatio),
                        onValueChange = { 
                            zoomRatio = it
                            cameraRef.value?.cameraControl?.setZoomRatio(it)
                        }
                    )
                    
                    // Exposure Compensation Slider
                    val exposureRange = cameraInfoRef.value?.exposureState?.exposureCompensationRange
                    val minExp = exposureRange?.lower ?: -12
                    val maxExp = exposureRange?.upper ?: 12
                    
                    ProSlider(
                        label = "EXPOSURE",
                        value = exposureIndex.toFloat(),
                        valueRange = minExp.toFloat()..maxExp.toFloat(),
                        valueText = if (exposureIndex >= 0) "+$exposureIndex" else "$exposureIndex",
                        onValueChange = {
                            exposureIndex = it.toInt()
                            cameraRef.value?.cameraControl?.setExposureCompensationIndex(it.toInt())
                        }
                    )
                }
            }
            
            // ── Action Buttons Row ──
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(PanelBg)
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Flip Camera
                IconActionButton(
                    text = "FLIP",
                    onClick = { isBackCamera = !isBackCamera }
                )
                
                // Flash Toggle
                IconActionButton(
                    text = if (isTorchOn) "FLASH ON" else "FLASH",
                    isActive = isTorchOn,
                    onClick = {
                        isTorchOn = !isTorchOn
                        cameraRef.value?.cameraControl?.enableTorch(isTorchOn)
                    }
                )
            }
        }
    }
}

@Composable
fun ProSlider(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    valueText: String,
    onValueChange: (Float) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            color = TextDim,
            fontSize = 10.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.width(64.dp),
            letterSpacing = 1.sp
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            modifier = Modifier.weight(1f),
            colors = SliderDefaults.colors(
                thumbColor = NeonCyan,
                activeTrackColor = NeonCyan,
                inactiveTrackColor = TextDim.copy(alpha = 0.3f)
            )
        )
        Text(
            text = valueText,
            color = NeonCyan,
            fontSize = 11.sp,
            modifier = Modifier.width(42.dp),
            textAlign = TextAlign.End
        )
    }
}

@Composable
fun IconActionButton(
    text: String,
    isActive: Boolean = false,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        shape = RoundedCornerShape(8.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isActive) NeonCyan.copy(alpha = 0.2f) else PanelBg,
        ),
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 8.dp)
    ) {
        Text(
            text = text,
            color = if (isActive) NeonCyan else Color.White,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.sp
        )
    }
}
