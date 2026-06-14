/**
 * =====================================================================
 * SmartbarViewModel.kt
 * Solenz AI Keyboard — Smartbar State Yönetimi
 * =====================================================================
 * Smartbar UI'sinin tüm state'ini yöneten ViewModel.
 * GhostTextController ile UI arasındaki köprü.
 *
 * FlorisBoard'un mevcut ViewModel yapısına entegre edilir.
 */

package dev.solenz.keyboard.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.solenz.keyboard.ai.ghost.GhostTextController
import dev.solenz.keyboard.ai.ghost.GhostTextControllerState
import dev.solenz.keyboard.ui.components.SmartbarUiState
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class SmartbarViewModel(application: Application) : AndroidViewModel(application) {

    private val ghostController = GhostTextController(
        context = application.applicationContext,
        isDarkTheme = true,
        onStateChanged = { state -> onControllerStateChanged(state) },
    )

    private val _uiState = MutableStateFlow(SmartbarUiState())
    val uiState: StateFlow<SmartbarUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val loaded = ghostController.initialize()
            _uiState.update { it.copy(isModelLoaded = loaded) }
        }
    }

    private fun onControllerStateChanged(state: GhostTextControllerState) {
        _uiState.update { current ->
            current.copy(
                controllerState = state,
                averageLatencyMs = ghostController.averageLatencyMs,
            )
        }
    }

    fun toggleAi(enabled: Boolean) {
        ghostController.setAiEnabled(enabled)
        _uiState.update { it.copy(isAiEnabled = enabled) }
    }

    fun onGhostAccept() {
        ghostController.commit()
    }

    fun onGhostDismiss() {
        ghostController.dismiss()
    }

    // Ghost controller referansı — IME servisine bağlanmak için
    fun getController(): GhostTextController = ghostController

    override fun onCleared() {
        ghostController.close()
        super.onCleared()
    }
}
