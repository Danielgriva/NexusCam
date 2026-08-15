package com.example.nexuscam

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.text.format.Formatter
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private var server: MjpegServer? = null
    
    private var ipAddress by mutableStateOf("Loading IP...")
    private var hasCameraPermission by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val ip = Formatter.formatIpAddress(wifiManager.connectionInfo.ipAddress)
        ipAddress = if (ip == "0.0.0.0") "No WiFi Connection" else "http://$ip:8080/video"
        
        server = MjpegServer(8080)
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
                    background = Color.Black,
                    primary = Color(0xFF00FFCC)
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "NEXUS CAM",
                            style = MaterialTheme.typography.headlineMedium,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(16.dp)
                        )
                        
                        Text(
                            text = "Enter this in Nexus Frame on your PC:",
                            style = MaterialTheme.typography.bodyLarge,
                            color = Color.White
                        )
                        
                        Text(
                            text = ipAddress,
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(bottom = 16.dp)
                        )
                        
                        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                            if (hasCameraPermission) {
                                CameraPreview(
                                    onFrame = { proxy ->
                                        server?.updateImage(proxy)
                                    }
                                )
                            } else {
                                Text(
                                    text = "Please allow Camera permissions to stream...",
                                    color = Color.White,
                                    modifier = Modifier.align(Alignment.Center)
                                )
                            }
                        }
                    }
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
fun CameraPreview(onFrame: (ImageProxy) -> Unit) {
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    var isBackCamera by remember { mutableStateOf(true) }
    var isTorchOn by remember { mutableStateOf(false) }
    val cameraControlRef = remember { mutableStateOf<CameraControl?>(null) }
    
    Box(modifier = Modifier.fillMaxSize()) {
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
                            cameraControlRef.value = camera.cameraControl
                            cameraControlRef.value?.enableTorch(isTorchOn)
                            
                            // Setup Tap-to-Focus
                            previewView.setOnTouchListener { view, event ->
                                if (event.action == android.view.MotionEvent.ACTION_UP) {
                                    val factory = previewView.meteringPointFactory
                                    val point = factory.createPoint(event.x, event.y)
                                    val action = FocusMeteringAction.Builder(point).build()
                                    cameraControlRef.value?.startFocusAndMetering(action)
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
        
        // UI Controls Overlay
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = { isBackCamera = !isBackCamera }) {
                Text("Flip Camera")
            }
            Button(onClick = {
                isTorchOn = !isTorchOn
                cameraControlRef.value?.enableTorch(isTorchOn)
            }) {
                Text(if (isTorchOn) "Flash: ON" else "Flash: OFF")
            }
        }
    }
}
