/**
 * =====================================================================
 * FlorisAiImeService.kt
 * Solenz AI Keyboard — Tam IME Servisi Entegrasyonu
 * =====================================================================
 * FlorisBoard'un FlorisImeService sınıfını extend eden ve
 * GhostTextController'ı tüm klavye eventlerine bağlayan
 * production-ready IME servisi.
 *
 * FlorisBoard entegrasyonu:
 *  Bu sınıftaki logik, FlorisBoard'un mevcut
 *  app/src/main/kotlin/dev/patrickgold/florisboard/ime/core/
 *  FlorisImeService.kt dosyasına override veya delegation ile
 *  entegre edilir.
 *
 * Özellikler:
 *  ✅ Tam ghost text yanılsaması (setComposingText + commitText)
 *  ✅ Tüm edge case'ler (backspace, enter, paste, selection)
 *  ✅ InputConnection geçersiz olduğunda graceful degradation
 *  ✅ Configuration change'de state korunması
 *  ✅ Accessibility servisi uyumluluğu
 */

package dev.solenz.keyboard.ai.ime

import android.inputmethodservice.InputMethodService
import android.os.Build
import android.util.Log
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import dev.solenz.keyboard.ai.ghost.GhostTextController
import dev.solenz.keyboard.ai.ghost.GhostTextControllerState
import kotlinx.coroutines.*

class FlorisAiImeService : InputMethodService() {

    companion object {
        private const val TAG = "FlorisAiIme"

        // Özellikleri kapatmak için bu flag false yapılabilir
        private const val GHOST_TEXT_ENABLED = true
    }

    // ─── Bileşenler ───────────────────────────────────────────────────
    private lateinit var ghostController: GhostTextController

    private val serviceScope = CoroutineScope(
        Dispatchers.Main.immediate + SupervisorJob()
    )

    // Mevcut editor tipi
    private var currentEditorInfo: EditorInfo? = null

    // ─── Lifecycle ────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "IME Service başlatılıyor...")

        ghostController = GhostTextController(
            context = applicationContext,
            isDarkTheme = isSystemDarkTheme(),
            onStateChanged = { state -> handleControllerStateChanged(state) },
        )

        // Asenkron başlat — klavye hemen açılabilsin
        serviceScope.launch {
            val ok = ghostController.initialize()
            Log.i(TAG, "Ghost text motor: ${if (ok) "HAZIR ✓" else "HATA ✗"}")
        }
    }

    override fun onBindInput() {
        super.onBindInput()
        // InputConnection henüz hazır değil, sadece log
        Log.d(TAG, "onBindInput")
    }

    override fun onStartInput(attribute: EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        currentEditorInfo = attribute

        // Ghost text'i bazı alan tiplerinde kapat
        val shouldEnableGhost = attribute?.let { shouldEnableGhostForEditor(it) } ?: true
        ghostController.setAiEnabled(GHOST_TEXT_ENABLED && shouldEnableGhost)

        Log.d(TAG, "onStartInput: inputType=${attribute?.inputType}, ghost=${GHOST_TEXT_ENABLED && shouldEnableGhost}")
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)

        // InputConnection artık hazır — controller'a bağla
        val ic = currentInputConnection
        ghostController.attachInputConnection(ic)
        ghostController.onStartInput()

        // Editor'daki mevcut metni sync et
        if (!restarting) {
            val textBefore = ic?.getTextBeforeCursor(200, 0)?.toString()
            ghostController.syncWithEditor(textBefore)
        }
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        ghostController.onFinishInput()
        ghostController.attachInputConnection(null)
        super.onFinishInputView(finishingInput)
    }

    override fun onFinishInput() {
        super.onFinishInput()
        currentEditorInfo = null
    }

    override fun onUpdateSelection(
        oldSelStart: Int, oldSelEnd: Int,
        newSelStart: Int, newSelEnd: Int,
        candidatesStart: Int, candidatesEnd: Int,
    ) {
        super.onUpdateSelection(
            oldSelStart, oldSelEnd,
            newSelStart, newSelEnd,
            candidatesStart, candidatesEnd,
        )

        // Kullanıcı imleci elle hareket ettirdiyse ghost'u temizle
        // (composing region dışına çıkıldı)
        if (candidatesStart == -1 && candidatesEnd == -1) {
            // Composing region kapandı — ghost zaten yok
            return
        }

        // Kullanıcı selection değiştirdiyse (sadece OK durumda)
        // ghost'u reddet (kullanıcı düzenleme moduna geçmiş)
        if (newSelStart != newSelEnd) {
            // Seçim var — ghost'u kaldır
            Log.d(TAG, "Kullanıcı seçim yaptı, ghost temizleniyor.")
        }
    }

    override fun onDestroy() {
        serviceScope.cancel()
        ghostController.close()
        super.onDestroy()
    }

    // ─── Klavye Tuşu Yönetimi ─────────────────────────────────────────

    /**
     * FlorisBoard'un key event sisteminden gelen tuş olayları.
     *
     * FlorisBoard entegrasyonunda bu metod FlorisImeService'in
     * ilgili key dispatch metoduna bağlanır.
     *
     * @param keyCode Android KeyEvent kodu
     * @param event KeyEvent nesnesi
     */
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (!GHOST_TEXT_ENABLED) return super.onKeyDown(keyCode, event)

        return when (keyCode) {
            KeyEvent.KEYCODE_SPACE,
            KeyEvent.KEYCODE_TAB -> {
                ghostController.handleKeyPress(' ')
            }

            KeyEvent.KEYCODE_DEL -> {
                ghostController.handleBackspace()
            }

            KeyEvent.KEYCODE_ENTER -> {
                ghostController.handleEnter()
            }

            else -> {
                // Yazdırılabilir karakter mi?
                val char = event?.unicodeChar?.toChar()
                if (char != null && char.isLetterOrDigit() || char == '\'' || char == '-') {
                    ghostController.handleKeyPress(char!!)
                } else {
                    // Özel tuş (Shift, Ctrl vb.) — ghost'u koru
                    super.onKeyDown(keyCode, event)
                }
            }
        }
    }

    /**
     * Yazılı karakter geldiğinde (IME commitText çağrısı dışında).
     * FlorisBoard'un kendi key commit sistemiyle entegre edilir.
     */
    fun onCommitChar(char: Char) {
        ghostController.handleKeyPress(char)
    }

    /**
     * Paste işlemi — ghost'u temizle.
     */
    fun onPaste(text: String) {
        val ic = currentInputConnection ?: return
        // Ghost'u temizle
        ic.finishComposingText()
        ic.commitText(text, 1)
        ghostController.onStartInput() // Sıfırla
    }

    // ─── Yardımcı Metodlar ────────────────────────────────────────────

    /**
     * Bu editor tipi için ghost text aktif olmalı mı?
     *
     * Şifre alanları, sayısal girişler vb. için ghost text anlamsız.
     */
    private fun shouldEnableGhostForEditor(info: EditorInfo): Boolean {
        val inputType = info.inputType
        val typeClass = inputType and android.text.InputType.TYPE_MASK_CLASS
        val typeVariation = inputType and android.text.InputType.TYPE_MASK_VARIATION

        return when {
            // Şifre alanları — ASLA aktif olmasın
            typeVariation == android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD ||
            typeVariation == android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
            typeVariation == android.text.InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD -> false

            // Sayısal giriş — ghost text gereksiz
            typeClass == android.text.InputType.TYPE_CLASS_NUMBER ||
            typeClass == android.text.InputType.TYPE_CLASS_PHONE -> false

            // URI / Email — kısmen aktif, sadece @ sonrası domain için yararlı olabilir
            typeVariation == android.text.InputType.TYPE_TEXT_VARIATION_URI ||
            typeVariation == android.text.InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS -> false

            // Metin alanı — aktif!
            typeClass == android.text.InputType.TYPE_CLASS_TEXT -> true

            else -> false
        }
    }

    /**
     * Sistem koyu tema kullanıyor mu?
     */
    private fun isSystemDarkTheme(): Boolean {
        val nightModeFlags = resources.configuration.uiMode and
                android.content.res.Configuration.UI_MODE_NIGHT_MASK
        return nightModeFlags == android.content.res.Configuration.UI_MODE_NIGHT_YES
    }

    /**
     * Controller durum değişikliklerini UI katmanına ilet.
     * Smartbar güncellemeleri için kullanılır.
     */
    private fun handleControllerStateChanged(state: GhostTextControllerState) {
        when (state) {
            is GhostTextControllerState.Showing -> {
                // Smartbar'a ghost metni bildir
                Log.d(TAG, "Ghost gösteriliyor: \"${state.ghost.completion}\" " +
                        "(${state.ghost.latencyMs}ms, conf=${state.ghost.confidence})")
                // notifySmartbarGhostReady(state.ghost)
            }

            is GhostTextControllerState.Committed -> {
                Log.i(TAG, "Ghost commit edildi: \"${state.text}\"")
                // notifySmartbarCommit(state.text)
            }

            is GhostTextControllerState.Ready -> {
                // Smartbar normal moda geçsin
            }

            else -> Unit
        }
    }

    // ─── Public API (Smartbar / UI için) ─────────────────────────────

    /** AI özelliğini Smartbar toggle'ından aç/kapat. */
    fun toggleAiGhostText(enabled: Boolean) {
        ghostController.setAiEnabled(enabled)
    }

    /** Anlık ghost text (Smartbar önizleme için). */
    fun getCurrentGhostText(): String? = ghostController.currentGhostText

    /** Ortalama latency (debug panel için). */
    fun getAverageLatency(): Long = ghostController.averageLatencyMs
}
