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
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.mapNotNull
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(
    private val imageSegmentationHelper: ImageSegmentationHelper
) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageSegmentationHelper = ImageSegmentationHelper(context)
                return MainViewModel(imageSegmentationHelper) as T
            }
        }
    }

    init {
        viewModelScope.launch { imageSegmentationHelper.initClassifier() }
    }

    private var segmentJob: Job? = null

    private val segmentationUiShareFlow = MutableStateFlow<Bitmap?>(null).also { flow ->
        viewModelScope.launch {
            imageSegmentationHelper.segmentation
                .mapNotNull { it.bitmap }.collect {
                    flow.emit(it)
                }
        }
    }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            imageSegmentationHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        segmentationUiShareFlow,
        errorMessage,
    ) { bitmap, error ->
        UiState(
            bitmap = bitmap,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState(null, null))


    /** Start segment an image.
     *  @param imageProxy contain `imageBitMap` and imageInfo as `image rotation degrees`.
     *
     */
    fun segment(imageProxy: ImageProxy) {
        segmentJob = viewModelScope.launch {
            imageSegmentationHelper.segment(
                imageProxy.toBitmap(),
                imageProxy.imageInfo.rotationDegrees,
            )
            imageProxy.close()
        }
    }

    /** Stop current segmentation */
    fun stopSegment() {
        viewModelScope.launch {
            segmentJob?.cancel()
            segmentationUiShareFlow.emit(null)
        }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}