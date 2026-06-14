/**
 * =====================================================================
 * AiKeyboardService.kt
 * Solenz AI Keyboard — FlorisBoard Entegrasyonu
 * =====================================================================
 * FlorisBoard'un FlorisImeService sınıfını extend eden ve
 * GhostTextEngine'i IME katmanına bağlayan ana servis.
 *
 * Bu sınıf FlorisBoard'un mevcut servisine entegre edilecek —
 * kendi başına standalone bir IME servisi olarak da çalışabilir.
 *
 * Manifest'e eklenmesi gereken entry:
 * ```xml
 * <service
 *     android:name=".AiKeyboardService"
 *     android:exported="true"
 *     android:label="@string/app_name"
 *     android:permission="android.permission.BIND_INPUT_METHOD">
 *     <intent-filter>
 *         <action android:name="android.view.InputMethod" />
 *     </intent-filter>
 *     <meta-data
 *         android:name="android.view.im"
 *         android:resource="@xml/method" />
 * </service>
 * ```
 */

package dev.solenz.keyboard.ai.ime

import android.inputmethodservice.InputMethodService
import android.util.Log
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import dev.solenz.keyboard.ai.engine.GhostTextEngine
import dev.solenz.keyboard.ai.engine.GhostTextEngineConfig
import dev.solenz.keyboard.ai.engine.GhostTextState
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

/**
 * Ghost Text mantığını InputMethodService'e bağlayan köprü sınıf.
 *
 * FlorisBoard entegrasyonu için bu sınıfın metodlarını
 * FlorisImeService içine taşıyın veya composition ile kullanın.
 */
class AiKeyboardService : InputMethodService() {

    companion object {
        private const val TAG = "AiKeyboardService"
    }

    // ─── Bileşenler ───────────────────────────────────────────────────
    private lateinit var ghostEngine: GhostTextEngine

    // Servis lifecycle scope'u
    private val serviceScope = CoroutineScope(
        Dispatchers.Main + SupervisorJob() + CoroutineName("AiKeyboardService")
    )

    // Anlık composing text (kullanıcının yazdığı ama henüz onaylanmayan)
    private val composingBuffer = StringBuilder()

    // Ghost text'in aktif olup olmadığı
    private var isGhostActive = false
    private var currentGhostText = ""

    // ─── Yaşam Döngüsü ───────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "IME Servisi başlatılıyor...")

        ghostEngine = GhostTextEngine(
            context = applicationContext,
            config = GhostTextEngineConfig(
                debouncingDelayMs = 280L,
                maxNewTokens = 8,
                confidenceThreshold = 0.05f,
            ),
        )

        // Motoru arka planda başlat
        serviceScope.launch {
            val success = ghostEngine.initialize()
            Log.i(TAG, "GhostTextEngine başlatma: ${if (success) "Başarılı ✓" else "Başarısız ✗"}")
        }

        // State akışını dinle
        serviceScope.launch {
            ghostEngine.stateFlow.collectLatest { state ->
                handleGhostState(state)
            }
        }
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        composingBuffer.clear()
        ghostEngine.dismiss()
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        super.onFinishInputView(finishingInput)
        clearGhostText()
        composingBuffer.clear()
    }

    override fun onDestroy() {
        serviceScope.cancel()
        ghostEngine.close()
        super.onDestroy()
    }

    // ─── Ghost Text Durum Yönetimi ────────────────────────────────────

    /**
     * GhostTextEngine'den gelen durum değişikliklerini işle.
     * Main dispatcher'da çalışır — UI güncellemeleri güvenli.
     */
    private fun handleGhostState(state: GhostTextState) {
        when (state) {
            is GhostTextState.Loading -> {
                // İsteğe bağlı: Küçük bir yükleme indikatörü gösterilebilir
            }

            is GhostTextState.Ready -> {
                currentGhostText = state.result.completion
                isGhostActive = true
                showGhostText(state.result.completion)
                Log.d(TAG, "Ghost ready: \"${state.result.completion}\" (${state.result.latencyMs}ms)")
            }

            is GhostTextState.Empty -> {
                isGhostActive = false
                currentGhostText = ""
                clearGhostText()
            }

            is GhostTextState.Committed -> {
                isGhostActive = false
                currentGhostText = ""
            }

            is GhostTextState.Error -> {
                Log.e(TAG, "Ghost Text hatası: ${state.message}")
                isGhostActive = false
            }
        }
    }

    // ─── Karakter Girişi ─────────────────────────────────────────────

    /**
     * Klavyeden karakter girişi geldiğinde çağrılır.
     * FlorisBoard'un kendi event sistemine bağlanacak.
     *
     * @param char Basılan karakter
     */
    fun onCharacterInput(char: Char) {
        when {
            // Space veya Tab — ghost text'i kabul et
            (char == ' ' || char == '\t') && isGhostActive -> {
                commitGhostText()
            }

            // Normal karakter — ghost text'i iptal et ve devam et
            else -> {
                if (isGhostActive) {
                    clearGhostText()
                    ghostEngine.dismiss()
                }
                composingBuffer.append(char)
                updateComposing()
                ghostEngine.onTextChanged(composingBuffer.toString())
            }
        }
    }

    /**
     * Backspace tuşu işleme.
     */
    fun onBackspace() {
        if (isGhostActive) {
            clearGhostText()
            ghostEngine.dismiss()
            return
        }
        if (composingBuffer.isNotEmpty()) {
            composingBuffer.deleteCharAt(composingBuffer.length - 1)
            updateComposing()
            if (composingBuffer.isNotEmpty()) {
                ghostEngine.onTextChanged(composingBuffer.toString())
            } else {
                ghostEngine.dismiss()
            }
        }
    }

    /**
     * Enter tuşu — ghost text'i reddet, satır oluştur.
     */
    fun onEnter() {
        if (isGhostActive) {
            clearGhostText()
            ghostEngine.dismiss()
        }
        commitText("\n")
        composingBuffer.clear()
    }

    // ─── InputConnection Metodları ────────────────────────────────────
    // (Adım 3'te detaylandırılacak, burada soyutlama gösterilmektedir)

    /**
     * Ghost text'i imlecin yanına gri renkte göster.
     * Detaylı implementasyon Adım 3'te: setComposingText() ile.
     */
    private fun showGhostText(ghostText: String) {
        val ic = currentInputConnection ?: return
        // Tam implementasyon Adım 3'te —
        // Mevcut composing + ghost text birleşimi setComposingText ile basılır
        Log.d(TAG, "showGhostText: \"$ghostText\"")
    }

    /**
     * Ghost text'i kalıcı hale getir.
     * Detaylı implementasyon Adım 3'te: commitText() ile.
     */
    private fun commitGhostText() {
        val ic = currentInputConnection ?: return
        val accepted = ghostEngine.commit()
        if (accepted.isNotEmpty()) {
            // Composing'i kapat, ghost + space kalıcı yaz
            ic.finishComposingText()
            ic.commitText("$accepted ", 1)
            composingBuffer.append("$accepted ")
            Log.i(TAG, "Ghost text commit: \"$accepted\"")
        } else {
            // Normal space
            ic.commitText(" ", 1)
            composingBuffer.append(" ")
        }
        isGhostActive = false
        currentGhostText = ""
    }

    /**
     * Ghost text'i temizle, composing'e geri dön.
     */
    private fun clearGhostText() {
        val ic = currentInputConnection ?: return
        if (composingBuffer.isNotEmpty()) {
            // Sadece composing text'i güncelle (ghost olmadan)
            ic.setComposingText(composingBuffer.toString(), 1)
        } else {
            ic.finishComposingText()
        }
        isGhostActive = false
        currentGhostText = ""
    }

    /**
     * Composing text'i güncelle (ghost text olmadan).
     */
    private fun updateComposing() {
        val ic = currentInputConnection ?: return
        if (composingBuffer.isEmpty()) {
            ic.finishComposingText()
        } else {
            ic.setComposingText(composingBuffer.toString(), 1)
        }
    }

    /**
     * Metni doğrudan commit et (composing olmadan).
     */
    private fun commitText(text: String) {
        currentInputConnection?.commitText(text, 1)
    }

    // ─── Public API (FlorisBoard entegrasyonu için) ───────────────────

    /**
     * AI özelliğini aç/kapat.
     * Smartbar toggle butonu tarafından çağrılır.
     */
    fun toggleAi(enabled: Boolean) {
        ghostEngine.setEnabled(enabled)
    }

    /**
     * Anlık performans bilgisi.
     * Smartbar debug paneli için.
     */
    fun getPerformanceStats(): Map<String, Any> = mapOf(
        "averageLatencyMs" to ghostEngine.averageLatencyMs,
        "isGhostActive" to isGhostActive,
        "currentGhost" to currentGhostText,
    )
}
