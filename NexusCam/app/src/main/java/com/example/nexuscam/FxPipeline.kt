package com.example.nexuscam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.ByteArrayOutputStream
import kotlin.random.Random

fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
    val yBuffer = imageProxy.planes[0].buffer
    val uBuffer = imageProxy.planes[1].buffer
    val vBuffer = imageProxy.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 50, out)
    val imageBytes = out.toByteArray()
    
    val options = BitmapFactory.Options()
    options.inSampleSize = 2 // Downscale to save memory on 4GB RAM devices
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
}

// --- Particle Engine ---
class Particle(var x: Float, var y: Float) {
    var vx = (Random.nextFloat() - 0.5f) * 4f
    var vy = (Random.nextFloat() - 0.5f) * 4f
    var life = 1.0f
    var decay = 0.02f + Random.nextFloat() * 0.05f
    val color = Color(0xFF00FFCC)

    fun update() {
        x += vx
        y += vy
        life -= decay
    }
}

class ParticleSystem(private val maxParticles: Int = 250) {
    val particles = mutableListOf<Particle>()

    fun emit(x: Float, y: Float, count: Int = 2) {
        if (particles.size < maxParticles) {
            for (i in 0 until count) {
                particles.add(Particle(x, y))
            }
        }
    }

    fun update() {
        val iter = particles.iterator()
        while (iter.hasNext()) {
            val p = iter.next()
            p.update()
            if (p.life <= 0) iter.remove()
        }
    }
}

// --- MediaPipe Helper ---
class HandTracker(
    val context: Context,
    val onResult: (HandLandmarkerResult, Int, Int) -> Unit
) {
    private var handLandmarker: HandLandmarker? = null

    init {
        setup()
    }

    private fun setup() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()
            
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.4f)
            .setMinTrackingConfidence(0.4f)
            .setResultListener { result, image ->
                onResult(result, image.width, image.height)
            }
            .setErrorListener { error ->
                error.printStackTrace()
            }
            .build()
            
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    fun processFrame(bitmap: Bitmap, timestamp: Long) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker?.detectAsync(mpImage, timestamp)
    }

    fun close() {
        handLandmarker?.close()
        handLandmarker = null
    }
}

// --- Renderers ---
fun renderNeonFX(
    drawScope: DrawScope,
    result: HandLandmarkerResult?,
    particles: ParticleSystem,
    w: Int,
    h: Int,
    isFrontCamera: Boolean
) {
    result?.let {
        for (hand in it.landmarks()) {
            // Draw lines connecting joints
            val connections = HandLandmarker.HAND_CONNECTIONS
            val path = Path()
            
            for (c in connections) {
                val start = hand[c.start()]
                val end = hand[c.end()]
                val sx = if(isFrontCamera) w - (start.x() * w) else start.x() * w
                val sy = start.y() * h
                val ex = if(isFrontCamera) w - (end.x() * w) else end.x() * w
                val ey = end.y() * h
                
                path.moveTo(sx, sy)
                path.lineTo(ex, ey)
            }
            
            // Draw glowing outline
            drawScope.drawPath(path, color = Color(0xFF00FFCC).copy(alpha = 0.6f), style = Stroke(width = 8f))
            drawScope.drawPath(path, color = Color.White, style = Stroke(width = 3f))
            
            // Emit particles at fingertips (joint 8, 12, 16, 20)
            val tips = listOf(8, 12, 16, 20)
            for (tip in tips) {
                if(tip < hand.size) {
                    val lm = hand[tip]
                    val px = if(isFrontCamera) w - (lm.x() * w) else lm.x() * w
                    val py = lm.y() * h
                    particles.emit(px, py, 1)
                }
            }
        }
    }
    
    // Draw and update particles
    particles.update()
    for (p in particles.particles) {
        drawScope.drawCircle(
            color = p.color.copy(alpha = p.life),
            radius = 3f * p.life,
            center = Offset(p.x, p.y)
        )
    }
}
