/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.image_segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.Log
import androidx.core.graphics.scale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.nio.FloatBuffer
import androidx.core.graphics.createBitmap
import org.tensorflow.lite.DataType

class ImageSegmentationHelper(private val context: Context) {
    companion object {
        private const val TAG = "ImageSegmentation"
    }

    private val _segmentation = MutableSharedFlow<SegmentationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val segmentation: SharedFlow<SegmentationResult> get() = _segmentation

    private val _error = MutableSharedFlow<Throwable?>()
    val error: SharedFlow<Throwable?> get() = _error

    private val type: TFFile = TFFile.SemSegm(context)

    private var interpreter: Interpreter? = null

    suspend fun initClassifier(delegate: Delegate = Delegate.CPU) {
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, type.file)

            Log.i(TAG, "Done creating LiteRT buffer from ${type.file}")
            val options = Interpreter.Options().apply {
                numThreads = 4
                useNNAPI = delegate == Delegate.NNAPI
            }
            Interpreter(litertBuffer, options)
        } catch (e: Exception) {
            Log.i(TAG, "Create LiteRT from ${type.file} is failed ${e.message}")
            _error.emit(e)
            null
        }
    }

    suspend fun segment(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(Dispatchers.IO) {
                val interpreter = interpreter ?: return@withContext
                if (isActive) {
                    val result = segment(interpreter, bitmap, type.bitmap, rotationDegrees)
                    _segmentation.emit(SegmentationResult(result))
                }
            }
        } catch (e: Exception) {
            Log.i(TAG, "Image segment error occurred: ${e.message}")
            _error.emit(e)
        }
    }

    private fun segment(
        interpreter: Interpreter,
        input: Bitmap,
        background: Bitmap,
        rotationDegrees: Int
    ): Bitmap {
        val (i, h, w, c) = interpreter.getOutputTensor(0).shape()
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
            .add(Rot90Op(-rotationDegrees / 90))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(input)
        tensorImage = imageProcessor.process(tensorImage)

        val outputBuffer = FloatBuffer.allocate(i * h * w * c)
        interpreter.run(tensorImage.buffer, outputBuffer)

        return generateGenericMaskFromBuffer(background, outputBuffer, w, h, c)
    }

    private fun generateGenericMaskFromBuffer(
        resBackground: Bitmap,
        buffer: FloatBuffer,
        w: Int,
        h: Int,
        c: Int,
        targetChannel: Int = 1
    ): Bitmap {
        buffer.rewind()
        val actualChannel = if (c == 1) 0 else targetChannel
        val smallMask = createBitmap(w, h)
        val maskPixels = IntArray(w * h)
        for (i in 0 until (w * h)) {
            val confidence = buffer.get(i * c + actualChannel)
            val alpha = (confidence * 255).toInt().coerceIn(0, 255)
            maskPixels[i] = Color.argb(alpha, 0, 0, 0)
        }
        smallMask.setPixels(maskPixels, 0, w, 0, 0, w, h)
        val fullResMask = smallMask.scale(resBackground.width, resBackground.height)
        val result = createBitmap(resBackground.width, resBackground.height)
        val canvas = Canvas(result)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        canvas.drawBitmap(fullResMask, 0f, 0f, paint)
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
        canvas.drawBitmap(resBackground, 0f, 0f, paint)
        return result
    }

    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    enum class Delegate {
        CPU, NNAPI
    }

    data class SegmentationResult(
        val bitmap: Bitmap,
    )

    sealed interface TFFile {
        val file: String
        val labels: List<String>
        val bitmap: Bitmap

        data class SemSegm(private val context: Context) : TFFile {
            override val file = "semsegm_of8000_latency_16fp.tflite"
            override val labels = listOf("background", "person")
            override val bitmap: Bitmap = context.assets.open("background.png").use {
                BitmapFactory.decodeStream(it)
            }
        }
    }
}
