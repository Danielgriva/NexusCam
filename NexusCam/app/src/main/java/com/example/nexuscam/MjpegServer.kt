package com.example.nexuscam

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import fi.iki.elonen.NanoHTTPD
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.atomic.AtomicReference

class MjpegServer(port: Int) : NanoHTTPD(port) {

    private val latestJpeg = AtomicReference<ByteArray?>(null)
    var activeConnections = 0
    var onConnectionStateChanged: ((Int) -> Unit)? = null

    fun updateImage(imageProxy: ImageProxy) {
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
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 70, out)
        
        latestJpeg.set(out.toByteArray())
        imageProxy.close()
    }

    override fun serve(session: IHTTPSession): Response {
        if (session.uri == "/video") {
            val stream = object : InputStream() {
                var currentFrame: ByteArray? = null
                var currentPos = 0

                init {
                    activeConnections++
                    onConnectionStateChanged?.invoke(activeConnections)
                }

                override fun close() {
                    activeConnections--
                    if(activeConnections < 0) activeConnections = 0
                    onConnectionStateChanged?.invoke(activeConnections)
                    super.close()
                }

                override fun read(): Int {
                    return -1 // Fallback
                }
                
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    if (currentFrame == null || currentPos >= currentFrame!!.size) {
                        val jpeg = latestJpeg.get() ?: run {
                            Thread.sleep(16)
                            return 0
                        }
                        
                        val header = "--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.size}\r\n\r\n".toByteArray()
                        val footer = "\r\n".toByteArray()
                        
                        val fullPayload = ByteArray(header.size + jpeg.size + footer.size)
                        System.arraycopy(header, 0, fullPayload, 0, header.size)
                        System.arraycopy(jpeg, 0, fullPayload, header.size, jpeg.size)
                        System.arraycopy(footer, 0, fullPayload, header.size + jpeg.size, footer.size)
                        
                        currentFrame = fullPayload
                        currentPos = 0
                        Thread.sleep(16) // Throttle to ~60 FPS
                    }
                    
                    val bytesToRead = Math.min(len, currentFrame!!.size - currentPos)
                    System.arraycopy(currentFrame!!, currentPos, b, off, bytesToRead)
                    currentPos += bytesToRead
                    return bytesToRead
                }
            }
            return newChunkedResponse(Response.Status.OK, "multipart/x-mixed-replace; boundary=--boundary", stream)
        }
        return newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "Not Found")
    }
}
