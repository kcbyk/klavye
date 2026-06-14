/**
 * =====================================================================
 * GhostTextController.kt
 * Solenz AI Keyboard — Ana Ghost Text Kontrol Katmanı
 * =====================================================================
 * InputConnection API'sini yöneten, ghost text yanılsamasının
 * tüm yaşam döngüsünü yöneten kritik sınıf.
 *
 * ═══════════════════════════════════════════════════════════════════
 * GHOST TEXT YAŞAM DÖNGÜSÜ
 * ═══════════════════════════════════════════════════════════════════
 *
 *  [1] KULLANICI YAZAR: "merhaba nası"
 *      composingBuffer = "merhaba nası"
 *      setComposingText("merhaba nası", cursor=1)
 *                                     ↑ imleç metnin sonunda
 *
 *  [2] GhostTextEngine tahmin üretir: "lsın bugün"
 *      setComposingText(
 *          SpannableString("merhaba nası [lsın bugün]"),
 *          cursor=1
 *      )
 *      [merhaba nası] = normal composing
 *      [lsın bugün]   = GhostTextSpan → gri + italik
 *                                     ↑ imleç kullanıcı metninin sonunda kalır!
 *
 *  [3a] KULLANICI SPACE/TAB BASAR → KABUL
 *       finishComposingText()           ← composing'i kapat
 *       commitText("merhaba nası ", 1)  ← kullanıcı metnini yaz
 *       commitText("lsın bugün ", 1)    ← ghost text'i arkasına yaz
 *       composingBuffer.clear()
 *
 *  [3b] KULLANICI FARKLI TUŞA BASAR → REDDET
 *       setComposingText("merhaba nası" + yeniKarakter, 1)
 *       → Ghost text kaybolur, sadece kullanıcı metni kalır
 *
 *  [3c] KULLANICI BACKSPACE BASAR → GERİ AL
 *       Ghost aktifse: sadece ghost'u kaldır (composing'e geri dön)
 *       Ghost yoksa: composing'den son karakteri sil
 *
 * ═══════════════════════════════════════════════════════════════════
 */

package dev.solenz.keyboard.ai.ghost

import android.content.Context
import android.text.SpannableString
import android.text.Spanned
import android.util.Log
import android.view.inputmethod.InputConnection
import dev.solenz.keyboard.ai.engine.GhostTextEngine
import dev.solenz.keyboard.ai.engine.GhostTextEngineConfig
import dev.solenz.keyboard.ai.engine.GhostTextState
import dev.solenz.keyboard.ai.engine.PredictionResult
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

/**
 * InputConnection ile GhostTextEngine'i bağlayan tam controller.
 *
 * FlorisBoard entegrasyonunda bu sınıfın bir instance'ı
 * FlorisImeService içinde tutulur ve her key event'te
 * ilgili metodlar çağrılır.
 */
class GhostTextController(
    private val context: Context,
    private val isDarkTheme: Boolean = true,
    private val onStateChanged: ((GhostTextControllerState) -> Unit)? = null,
) {
    companion object {
        private const val TAG = "GhostTextController"

        /** Kabul tetikleyici karakterler */
        private val COMMIT_TRIGGERS = setOf(' ', '\t')

        /** Cümle sonu — ghost temizle */
        private val SENTENCE_ENDERS = setOf('.', '!', '?', '\n')
    }

    // ─── Bileşenler ───────────────────────────────────────────────────
    private val engine = GhostTextEngine(
        context = context,
        config = GhostTextEngineConfig(
            debouncingDelayMs = 280L,
            maxNewTokens = 8,
            confidenceThreshold = 0.05f,
        ),
    )

    private val scope = CoroutineScope(
        Dispatchers.Main + SupervisorJob() + CoroutineName("GhostController")
    )

    // ─── Durum ────────────────────────────────────────────────────────

    /** InputConnection referansı — her bağlantıda güncellenir */
    private var inputConnection: InputConnection? = null

    /** Kullanıcının yazdığı composing buffer */
    private val composingBuffer = StringBuilder()

    /** Aktif ghost text */
    private var activeGhost: ActiveGhost? = null

    /** Controller'ın dış dünyaya bildirdiği durum */
    private var currentState = GhostTextControllerState.Idle
        set(value) {
            field = value
            onStateChanged?.invoke(value)
        }

    // ─── Başlatma ────────────────────────────────────────────────────

    init {
        // GhostTextEngine state akışını dinle
        scope.launch {
            engine.stateFlow.collectLatest { state ->
                handleEngineState(state)
            }
        }
    }

    /**
     * Motoru başlat. FlorisBoard.onCreate() içinde çağrılmalı.
     */
    suspend fun initialize(): Boolean {
        val ok = engine.initialize()
        currentState = if (ok) GhostTextControllerState.Ready else GhostTextControllerState.Error
        return ok
    }

    /**
     * Aktif InputConnection'ı güncelle.
     * FlorisBoard her yeni edit field'a geçişte bu metodu çağırır.
     */
    fun attachInputConnection(ic: InputConnection?) {
        inputConnection = ic
        if (ic == null) {
            clearGhostInternal()
        }
    }

    // ─── Ana Tuş İşleme Metodları ────────────────────────────────────

    /**
     * Karakteri işle — ghost text kabulü veya normal giriş.
     *
     * @param char Basılan karakter
     * @return true: karakter işlendi, false: IME normal davranışa devam etmeli
     */
    fun handleKeyPress(char: Char): Boolean {
        return when {
            // ── Space / Tab → Ghost'u KABUL ET ──────────────────────
            char in COMMIT_TRIGGERS && activeGhost != null -> {
                commitGhostAndSpace()
                true
            }

            // ── Cümle sonu → Ghost'u REDDET, karakteri yaz ──────────
            char in SENTENCE_ENDERS -> {
                dismissGhost()
                typeCharacter(char)
                true
            }

            // ── Normal karakter → Ghost'u iptal et, yaz, yeni tahmin ─
            else -> {
                if (activeGhost != null) dismissGhost()
                typeCharacter(char)
                triggerPrediction()
                true
            }
        }
    }

    /**
     * Backspace işle.
     *
     * Ghost aktifse: sadece ghost'u kaldır (composing'e geri dön)
     * Ghost yoksa: composing'den son karakteri sil
     *
     * @return true: işlendi
     */
    fun handleBackspace(): Boolean {
        val ic = inputConnection ?: return false

        return when {
            // Ghost aktif → sadece ghost'u kaldır
            activeGhost != null -> {
                dismissGhost()
                // Composing'i ghost olmadan tekrar göster
                refreshComposingWithoutGhost()
                true
            }

            // Composing buffer doluysa → son karakteri sil
            composingBuffer.isNotEmpty() -> {
                composingBuffer.deleteCharAt(composingBuffer.length - 1)
                engine.dismiss()

                if (composingBuffer.isEmpty()) {
                    ic.finishComposingText()
                } else {
                    ic.setComposingText(composingBuffer.toString(), 1)
                    triggerPrediction() // Kalan metni yeniden tahmin et
                }
                true
            }

            // Composing boş → standart backspace
            else -> false
        }
    }

    /**
     * Enter işle — ghost'u reddet, satır ekle.
     */
    fun handleEnter(): Boolean {
        dismissGhost()
        commitComposingAndClear()
        inputConnection?.commitText("\n", 1)
        return true
    }

    /**
     * Kelime tamamlama (Smartbar önerisi seçildi) — tam kelimeyi commit et.
     */
    fun handleWordCompletion(word: String) {
        dismissGhost()
        val ic = inputConnection ?: return
        ic.finishComposingText()
        ic.commitText("$word ", 1)
        composingBuffer.clear()
        composingBuffer.append("$word ")
        engine.dismiss()
    }

    // ─── Ghost Text InputConnection Operasyonları ─────────────────────

    /**
     * ★ TEMEL METOD ★
     *
     * Ghost text'i ekranda göster.
     * setComposingText() ile kullanıcı metni + gri ghost birleştirilerek
     * imlecin sağ tarafına basılır, imleç kullanıcı metninin sonunda kalır.
     *
     * Görsel:
     *   "merhaba nası|lsın bugün"
     *                 ↑ imleç burada, solda kullanıcı metni, sağda ghost
     */
    private fun showGhost(ghost: ActiveGhost) {
        val ic = inputConnection ?: return

        val userText = composingBuffer.toString()
        val ghostText = ghost.completion

        // Composite SpannableString oluştur
        val spannable = GhostSpanFactory.buildCompositeSpannable(
            userText = userText,
            ghostText = ghostText,
            isDarkTheme = isDarkTheme,
            confidence = ghost.confidence,
        )

        // setComposingText(text, newCursorPosition)
        //   newCursorPosition = 1 → metnin en sonuna git
        //   Ama biz imleci kullanıcı metninin sonunda istiyoruz!
        //   Bu yüzden özel bir cursor position hesaplıyoruz.
        val cursorPosition = -(ghostText.length + 1) // negatif = metinden geriye say
        ic.setComposingText(spannable, cursorPosition)

        currentState = GhostTextControllerState.Showing(ghost)
        Log.d(TAG, "Ghost gösteriliyor: \"$ghostText\" (cursor offset: $cursorPosition)")
    }

    /**
     * ★ KABUL METODU ★
     *
     * Kullanıcı Space/Tab bastığında ghost text'i kalıcı hale getir.
     *
     * Sıra:
     *  1. Composing region'ı kapat (finishComposingText)
     *  2. Kullanıcı metnini commit et
     *  3. Ghost text + boşluk'u commit et
     *  4. State'i temizle
     */
    private fun commitGhostAndSpace() {
        val ic = inputConnection ?: return
        val ghost = activeGhost ?: return
        val userText = composingBuffer.toString()
        val ghostText = ghost.completion

        Log.i(TAG, "Ghost kabul ediliyor: \"$ghostText\"")

        // 1. Composing region'ı kapat
        ic.finishComposingText()

        // 2. Kullanıcı metnini commit et (zaten editörde var, sadece kalıcılaştır)
        // NOT: finishComposingText zaten bunu yapıyor, ama bazı editörler
        // için explicit commit gerekebilir
        // ic.commitText(userText, 1)  // ← Eğer çift yazma olursa bunu yoruma al

        // 3. Ghost completion + space'i commit et
        ic.commitText("$ghostText ", 1)

        // 4. İç durumu güncelle
        composingBuffer.clear()
        composingBuffer.append("$userText$ghostText ")
        activeGhost = null
        engine.commit()

        currentState = GhostTextControllerState.Committed(ghostText)

        // Kısa gecikme sonra composing'i yeniden başlat
        // (yeni tahminler için)
        scope.launch {
            delay(50)
            composingBuffer.clear()
            currentState = GhostTextControllerState.Ready
        }
    }

    /**
     * ★ RED METODU ★
     *
     * Ghost text'i kaldır, sadece composing text'i bırak.
     * Yeni karakter yazıldığında veya ghost reddedildiğinde çağrılır.
     */
    private fun dismissGhost() {
        if (activeGhost == null) return

        activeGhost = null
        engine.dismiss()
        refreshComposingWithoutGhost()
        currentState = GhostTextControllerState.Ready

        Log.d(TAG, "Ghost reddedildi.")
    }

    /**
     * Ghost olmadan sadece composing text'i göster.
     * Dismiss veya backspace sonrasında çağrılır.
     */
    private fun refreshComposingWithoutGhost() {
        val ic = inputConnection ?: return
        if (composingBuffer.isEmpty()) {
            ic.finishComposingText()
        } else {
            ic.setComposingText(composingBuffer.toString(), 1)
        }
    }

    /**
     * Composing text'i commit et ve buffer'ı temizle.
     * Enter basıldığında veya kelime tamamlamada kullanılır.
     */
    private fun commitComposingAndClear() {
        val ic = inputConnection ?: return
        if (composingBuffer.isNotEmpty()) {
            ic.finishComposingText()
            composingBuffer.clear()
        }
    }

    /**
     * Karakter yaz ve composing text'i güncelle.
     */
    private fun typeCharacter(char: Char) {
        val ic = inputConnection ?: return
        composingBuffer.append(char)
        ic.setComposingText(composingBuffer.toString(), 1)
    }

    /**
     * GhostTextEngine'i mevcut composing buffer ile tetikle.
     */
    private fun triggerPrediction() {
        if (composingBuffer.length >= 3) {
            engine.onTextChanged(composingBuffer.toString())
        }
    }

    /**
     * Ghost text'i iç olarak temizle (IC güncellemesi olmadan).
     */
    private fun clearGhostInternal() {
        activeGhost = null
        composingBuffer.clear()
        currentState = GhostTextControllerState.Idle
    }

    // ─── Engine State Handler ─────────────────────────────────────────

    /**
     * GhostTextEngine'den gelen durum değişikliklerini işle.
     * Main thread'de çalışır — UI ve IC operasyonları güvenli.
     */
    private fun handleEngineState(state: GhostTextState) {
        when (state) {
            is GhostTextState.Loading -> {
                // İmleç yanında küçük bir "..." indikatörü eklenebilir
                // Şimdilik sadece log
                Log.v(TAG, "Engine: yükleniyor...")
            }

            is GhostTextState.Ready -> {
                val result = state.result
                if (result.completion.isBlank()) return

                // Yeni ghost oluştur
                activeGhost = ActiveGhost(
                    completion = result.completion,
                    confidence = result.confidence,
                    latencyMs = result.latencyMs,
                )

                showGhost(activeGhost!!)
            }

            is GhostTextState.Empty -> {
                if (activeGhost != null) {
                    activeGhost = null
                    refreshComposingWithoutGhost()
                }
            }

            is GhostTextState.Committed -> {
                // Engine tarafında commit işlendi
            }

            is GhostTextState.Error -> {
                Log.e(TAG, "Engine hatası: ${state.message}")
                activeGhost = null
            }
        }
    }

    // ─── Input Context Değişiklikleri ────────────────────────────────

    /**
     * Yeni bir input field'a geçildi — sıfırla.
     */
    fun onStartInput() {
        composingBuffer.clear()
        activeGhost = null
        engine.dismiss()
        currentState = GhostTextControllerState.Ready
    }

    /**
     * Input field kapandı.
     */
    fun onFinishInput() {
        clearGhostInternal()
        engine.dismiss()
    }

    /**
     * Mevcut editor'deki metni (imleç öncesi) al ve sync et.
     * FlorisBoard yeniden bağlandığında context'i kurtarmak için.
     */
    fun syncWithEditor(textBeforeCursor: String?) {
        if (textBeforeCursor.isNullOrEmpty()) return

        // Son kelimeyi composing buffer'a al
        val lastWord = textBeforeCursor.trimEnd().split(" ").lastOrNull() ?: return
        if (lastWord.length >= 3) {
            composingBuffer.clear()
            composingBuffer.append(textBeforeCursor)
            engine.onTextChanged(textBeforeCursor)
        }
    }

    // ─── AI Toggle ───────────────────────────────────────────────────

    /** AI özelliğini aç/kapat (Smartbar toggle için). */
    fun setAiEnabled(enabled: Boolean) {
        engine.setEnabled(enabled)
        if (!enabled) {
            dismissGhost()
        }
    }

    /** Mevcut ghost text. Smartbar gösterimi için. */
    val currentGhostText: String? get() = activeGhost?.completion

    /** Ortalama latency. Smartbar debug için. */
    val averageLatencyMs: Long get() = engine.averageLatencyMs

    // ─── Lifecycle ───────────────────────────────────────────────────

    fun close() {
        scope.cancel()
        engine.close()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Yardımcı Veri Sınıfları
// ─────────────────────────────────────────────────────────────────────

/**
 * Aktif ghost text'in anlık snapshot'ı.
 */
data class ActiveGhost(
    val completion: String,
    val confidence: Float,
    val latencyMs: Long,
)

/**
 * Controller'ın dışarıya bildirdiği durum (Smartbar UI için).
 */
sealed class GhostTextControllerState {
    object Idle : GhostTextControllerState()
    object Ready : GhostTextControllerState()
    object Error : GhostTextControllerState()
    data class Showing(val ghost: ActiveGhost) : GhostTextControllerState()
    data class Committed(val text: String) : GhostTextControllerState()
}
