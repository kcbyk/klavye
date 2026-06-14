/**
 * =====================================================================
 * GhostTextEngine.kt
 * Solenz AI Keyboard — FlorisBoard Entegrasyonu
 * =====================================================================
 * GhostTextInterpreter'ın üstünde çalışan yüksek seviyeli motor.
 *
 * Görevleri:
 *  ✅ Debouncing — peş peşe gelen tuş vuruşlarını birleştirip
 *     sadece "kullanıcı durdu" anında modeli tetikler (300ms)
 *  ✅ Önbellek (LRU Cache) — aynı prefix için tekrar model çalıştırmaz
 *  ✅ İptal yönetimi — yeni karakter geldiğinde eski çıkarımı iptal eder
 *  ✅ StateFlow ile UI katmanına reaktif sinyal gönderir
 *  ✅ Latency metrikleri toplar (debug panel için)
 */

package dev.solenz.keyboard.ai.engine

import android.content.Context
import android.util.Log
import android.util.LruCache
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * GhostText motorunun dışarıya yayınladığı durum.
 */
sealed class GhostTextState {
    /** Model tahmin üretiyor */
    object Loading : GhostTextState()

    /** Tahmin hazır, UI güncellenebilir */
    data class Ready(val result: PredictionResult) : GhostTextState()

    /** Tahmin yok (prefix çok kısa, güven düşük vb.) */
    object Empty : GhostTextState()

    /** Kullanıcı ghost text'i kabul etti (space/tab) */
    object Committed : GhostTextState()

    /** Hata durumu */
    data class Error(val message: String) : GhostTextState()
}

/**
 * Motor ayarları.
 */
data class GhostTextEngineConfig(
    /** Tuş vuruşu sonrası model tetiklenmeden önce bekleme süresi (ms) */
    val debouncingDelayMs: Long = 300L,

    /** LRU cache boyutu (prefix sayısı) */
    val cacheSize: Int = 50,

    /** Minimum prefix kelime sayısı */
    val minPrefixWords: Int = 1,

    /** Maksimum üretilecek token sayısı */
    val maxNewTokens: Int = 8,

    /** Güven skoru eşiği — altı gösterilmez */
    val confidenceThreshold: Float = 0.05f,

    /** AI özelliği aktif mi */
    val isEnabled: Boolean = true,
)

/**
 * Ana GhostText motoru. FlorisBoard'un IME katmanından kullanılır.
 *
 * Kullanım:
 * ```kotlin
 * val engine = GhostTextEngine(context)
 * engine.initialize()
 *
 * // UI'da state'i dinle
 * lifecycleScope.launch {
 *     engine.stateFlow.collect { state -> updateUI(state) }
 * }
 *
 * // Her tuş vuruşunda
 * engine.onTextChanged("merhaba nasıl")
 *
 * // Space/Tab'da kabul et
 * engine.commit()
 *
 * // IME kapanırken
 * engine.close()
 * ```
 */
class GhostTextEngine(
    private val context: Context,
    private var config: GhostTextEngineConfig = GhostTextEngineConfig(),
) : AutoCloseable {

    companion object {
        private const val TAG = "GhostTextEngine"
    }

    // ─── Bileşenler ───────────────────────────────────────────────────
    private val interpreter = GhostTextInterpreter(
        context = context,
        maxNewTokens = config.maxNewTokens,
        minConfidence = config.confidenceThreshold,
    )

    // LRU Cache — prefix → tahmin sonucu
    private val cache = object : LruCache<String, PredictionResult>(config.cacheSize) {}

    // Coroutine scope — motor kapanınca tüm işler iptal edilir
    private val scope = CoroutineScope(
        Dispatchers.Default + SupervisorJob() + CoroutineName("GhostTextEngine")
    )

    // Debounce için aktif job referansı
    private var pendingJob: Job? = null

    // ─── Public StateFlow ─────────────────────────────────────────────

    private val _stateFlow = MutableStateFlow<GhostTextState>(GhostTextState.Empty)

    /** UI katmanının dinlediği durum akışı. */
    val stateFlow: StateFlow<GhostTextState> = _stateFlow.asStateFlow()

    /** Mevcut onaylı tamamlama metni (commitText için kullanılır). */
    var currentCompletion: String = ""
        private set

    /** Latency istatistikleri (Smartbar debug paneli için). */
    private val latencyHistory = ArrayDeque<Long>(20)
    val averageLatencyMs: Long get() = if (latencyHistory.isEmpty()) 0L else latencyHistory.average().toLong()

    // ─── Yaşam Döngüsü ───────────────────────────────────────────────

    /**
     * Motoru başlat. FlorisBoard'un onCreate() içinde çağrılmalı.
     * Tamamlandığında stateFlow aracılığıyla UI bilgilendirilir.
     */
    suspend fun initialize(): Boolean {
        val success = interpreter.initialize()
        if (!success) {
            _stateFlow.value = GhostTextState.Error("Model başlatılamadı")
        }
        return success
    }

    // ─── Ana API ─────────────────────────────────────────────────────

    /**
     * Kullanıcı bir tuşa bastığında veya metin değiştiğinde çağrılır.
     *
     * [debouncingDelayMs] ms debounce uygulanır:
     * Hızlı yazım sırasında her tuş için model çalışmaz,
     * sadece kullanıcı durduğunda (300ms sessizlik) tetiklenir.
     *
     * @param currentText Klavyenin mevcut composing + committed metni
     */
    fun onTextChanged(currentText: String) {
        if (!config.isEnabled) return

        // Eski bekleyen iş varsa iptal et
        pendingJob?.cancel()

        // Prefix'i normalize et
        val prefix = currentText.trim()

        // Çok kısa ise temizle
        if (prefix.split(" ").size < config.minPrefixWords || prefix.length < 3) {
            clearGhostText()
            return
        }

        // Debounce — kullanıcı yazmayı durdurunca çalış
        pendingJob = scope.launch {
            delay(config.debouncingDelayMs)

            // İptal edildiyse çalışma
            if (!isActive) return@launch

            _stateFlow.value = GhostTextState.Loading

            // Cache'i kontrol et
            val cachedResult = cache.get(prefix)
            if (cachedResult != null) {
                Log.d(TAG, "Cache hit: \"$prefix\"")
                deliverResult(cachedResult)
                return@launch
            }

            // Model çıkarımı yap
            val result = interpreter.predict(prefix)

            if (!isActive) return@launch // İptal edildiyse sonucu yayınlama

            if (result == EmptyPrediction || result.completion.isBlank()) {
                _stateFlow.value = GhostTextState.Empty
                currentCompletion = ""
                return@launch
            }

            // Cache'e ekle
            cache.put(prefix, result)

            // Latency geçmişini güncelle
            latencyHistory.addLast(result.latencyMs)
            if (latencyHistory.size > 20) latencyHistory.removeFirst()

            deliverResult(result)
        }
    }

    /**
     * Kullanıcı Space veya Tab'a bastı — ghost text'i kabul et.
     * Çağıran katman bu komuttan sonra [currentCompletion]'ı
     * InputConnection.commitText() ile kalıcı hale getirir.
     *
     * @return Kabul edilen tamamlama metni ("" ise ghost text yoktu)
     */
    fun commit(): String {
        pendingJob?.cancel()
        val completion = currentCompletion
        currentCompletion = ""
        _stateFlow.value = GhostTextState.Committed
        Log.d(TAG, "Commit: \"$completion\"")
        return completion
    }

    /**
     * Kullanıcı ghost text'i reddetti (farklı bir tuşa bastı).
     * Composing text temizlenir.
     */
    fun dismiss() {
        pendingJob?.cancel()
        clearGhostText()
    }

    /**
     * AI özelliğini aç/kapat (Smartbar toggle için).
     */
    fun setEnabled(enabled: Boolean) {
        config = config.copy(isEnabled = enabled)
        if (!enabled) {
            pendingJob?.cancel()
            clearGhostText()
        }
        Log.i(TAG, "AI Ghost Text: ${if (enabled) "Aktif" else "Pasif"}")
    }

    /**
     * Önbelleği temizle (dil değişikliği, ayar güncellemesi vb.).
     */
    fun clearCache() {
        cache.evictAll()
        Log.d(TAG, "Cache temizlendi.")
    }

    // ─── Yardımcı Metodlar ────────────────────────────────────────────

    private fun deliverResult(result: PredictionResult) {
        currentCompletion = result.completion
        _stateFlow.value = GhostTextState.Ready(result)
        Log.d(TAG, "Ghost: \"${result.completion}\" | ${result.latencyMs}ms | conf=${result.confidence}")
    }

    private fun clearGhostText() {
        currentCompletion = ""
        _stateFlow.value = GhostTextState.Empty
    }

    override fun close() {
        scope.cancel()
        interpreter.close()
        Log.i(TAG, "GhostTextEngine kapatıldı.")
    }
}
