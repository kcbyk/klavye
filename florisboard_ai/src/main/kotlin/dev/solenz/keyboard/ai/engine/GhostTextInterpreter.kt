/**
 * =====================================================================
 * GhostTextInterpreter.kt
 * Solenz AI Keyboard — FlorisBoard Entegrasyonu
 * =====================================================================
 * TFLite modelini çalıştıran ana Interpreter köprüsü.
 *
 * Sorumluluklar:
 *  - assets/ghost_text_model.tflite dosyasını MappedByteBuffer ile yükler
 *  - GPU Delegate → NNAPI Delegate → CPU fallback zinciri
 *  - Her tuş vuruşunda < 50ms'de next-token tahmin eder
 *  - Greedy decoding ile ghost text completion üretir
 *  - Coroutine tabanlı asenkron çalışma (Main thread'i bloke etmez)
 *
 * Bağımlılıklar (build.gradle):
 *   implementation("org.tensorflow:tensorflow-lite:2.14.0")
 *   implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
 *   implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
 */

package dev.solenz.keyboard.ai.engine

import android.content.Context
import android.util.Log
import dev.solenz.keyboard.ai.tokenizer.TFLiteTokenizer
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.system.measureTimeMillis

/**
 * TFLite model çıkarım sonucu.
 *
 * @property tokens Tahmin edilen kelime/token listesi
 * @property completion Birleştirilmiş tamamlama metni
 * @property latencyMs Çıkarım süresi (milisaniye)
 * @property confidence En yüksek token'ın softmax skoru
 */
data class PredictionResult(
    val tokens: List<String>,
    val completion: String,
    val latencyMs: Long,
    val confidence: Float,
)

/**
 * Boş/hata durumunu temsil eden sentinel nesne.
 */
val EmptyPrediction = PredictionResult(
    tokens = emptyList(),
    completion = "",
    latencyMs = 0L,
    confidence = 0f,
)

/**
 * TFLite modeli için hızlandırıcı seçeneği.
 */
enum class AcceleratorMode {
    AUTO,    // GPU → NNAPI → CPU (otomatik fallback)
    GPU,     // Sadece GPU Delegate
    NNAPI,   // Sadece NNAPI (Android Neural Networks API)
    CPU,     // Sadece CPU (en stabil, daha yavaş)
}

/**
 * Ana TFLite Interpreter sınıfı.
 *
 * Thread-Safety: [predict] metodu kendi Dispatcher'ında çalışır,
 * dışarıdan herhangi bir thread'den çağrılabilir.
 */
class GhostTextInterpreter(
    private val context: Context,
    private val acceleratorMode: AcceleratorMode = AcceleratorMode.AUTO,
    private val maxSeqLength: Int = 64,
    private val maxNewTokens: Int = 8,
    private val temperature: Float = 0.7f,
    private val minConfidence: Float = 0.05f, // Bu skorun altı gösterilmez
) : AutoCloseable {

    companion object {
        private const val TAG = "GhostTextInterpreter"
        private const val MODEL_ASSET_PATH = "ghost_text_model.tflite"
        private const val NUM_THREADS = 4

        // Quantized model için giriş veri tipi
        private const val INPUT_TYPE_INT32 = 0
    }

    // ─── İç Bileşenler ───────────────────────────────────────────────
    private var interpreter: Interpreter? = null
    private var tokenizer: TFLiteTokenizer? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    // Çıkarım için özel coroutine dispatcher (tek thread — sıralı işlem)
    private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

    // Durum bayrakları
    @Volatile private var isInitialized = false
    @Volatile private var isClosing = false

    // ─── Başlatma ────────────────────────────────────────────────────

    /**
     * Modeli ve tokenizer'ı asenkron olarak başlat.
     * FlorisBoard'un InputMethodService.onCreate() içinde çağrılmalı.
     *
     * @return Başlatma başarılı mı?
     */
    suspend fun initialize(): Boolean = withContext(inferenceDispatcher) {
        if (isInitialized) return@withContext true

        try {
            val elapsed = measureTimeMillis {
                // 1. Tokenizer yükle
                tokenizer = TFLiteTokenizer.fromAssets(context)
                Log.i(TAG, "Tokenizer yüklendi. Vocab boyutu: ${tokenizer?.vocabSize}")

                // 2. Model dosyasını MappedByteBuffer ile yükle (zero-copy)
                val modelBuffer = loadModelFromAssets()

                // 3. Interpreter oluştur
                interpreter = createInterpreter(modelBuffer)

                // 4. Tensör boyutlarını logla
                logTensorInfo()

                isInitialized = true
            }
            Log.i(TAG, "✓ GhostTextInterpreter başlatıldı. Süre: ${elapsed}ms")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Başlatma hatası: ${e.message}", e)
            false
        }
    }

    /**
     * Model dosyasını assets'ten memory-mapped buffer olarak yükle.
     * MappedByteBuffer; dosyayı RAM'e kopyalamadan doğrudan diskten okur.
     */
    private fun loadModelFromAssets(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_ASSET_PATH)
        val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength,
        )
    }

    /**
     * Hızlandırıcı öncelik sırası ile Interpreter oluştur.
     * GPU → NNAPI → CPU fallback zinciri.
     */
    private fun createInterpreter(modelBuffer: MappedByteBuffer): Interpreter {
        val options = Interpreter.Options().apply {
            numThreads = NUM_THREADS
            useXNNPACK = true  // ARM NEON optimizasyonu
        }

        return when (acceleratorMode) {
            AcceleratorMode.GPU -> tryGpuDelegate(modelBuffer, options)
            AcceleratorMode.NNAPI -> tryNnapiDelegate(modelBuffer, options)
            AcceleratorMode.CPU -> Interpreter(modelBuffer, options)
            AcceleratorMode.AUTO -> {
                tryGpuDelegate(modelBuffer, options)
                    ?: tryNnapiDelegate(modelBuffer, options)
                    ?: run {
                        Log.i(TAG, "CPU modu kullanılıyor.")
                        Interpreter(modelBuffer, options)
                    }
            }
        }
    }

    private fun tryGpuDelegate(buffer: MappedByteBuffer, options: Interpreter.Options): Interpreter? {
        return try {
            val gpuOptions = GpuDelegate.Options().apply {
                setPrecisionLossAllowed(true) // Hız için hafif doğruluk tradeoff
            }
            gpuDelegate = GpuDelegate(gpuOptions)
            options.addDelegate(gpuDelegate!!)
            val interp = Interpreter(buffer, options)
            Log.i(TAG, "✓ GPU Delegate aktif.")
            interp
        } catch (e: Exception) {
            Log.w(TAG, "GPU Delegate başarısız, NNAPI deneniyor: ${e.message}")
            gpuDelegate?.close()
            gpuDelegate = null
            null
        }
    }

    private fun tryNnapiDelegate(buffer: MappedByteBuffer, options: Interpreter.Options): Interpreter? {
        return try {
            val nnapiOptions = NnApiDelegate.Options().apply {
                executionPreference = NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
            }
            nnapiDelegate = NnApiDelegate(nnapiOptions)
            options.addDelegate(nnapiDelegate!!)
            val interp = Interpreter(buffer, options)
            Log.i(TAG, "✓ NNAPI Delegate aktif.")
            interp
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI Delegate başarısız: ${e.message}")
            nnapiDelegate?.close()
            nnapiDelegate = null
            null
        }
    }

    private fun logTensorInfo() {
        val interp = interpreter ?: return
        val inputCount = interp.inputTensorCount
        val outputCount = interp.outputTensorCount

        for (i in 0 until inputCount) {
            val tensor = interp.getInputTensor(i)
            Log.d(TAG, "Input[$i]: name=${tensor.name()}, shape=${tensor.shape().toList()}, dtype=${tensor.dataType()}")
        }
        for (i in 0 until outputCount) {
            val tensor = interp.getOutputTensor(i)
            Log.d(TAG, "Output[$i]: name=${tensor.name()}, shape=${tensor.shape().toList()}, dtype=${tensor.dataType()}")
        }
    }

    // ─── Çıkarım (Inference) ─────────────────────────────────────────

    /**
     * Ana tahmin metodu — Ghost Text completion üretir.
     *
     * Greedy decoding ile [maxNewTokens] kadar token üretilir.
     * Her token bir öncekinin çıktısına eklenerek otoregressif
     * şekilde tamamlama oluşturulur.
     *
     * @param prefix Kullanıcının klavyeye yazdığı metin (anlık)
     * @return [PredictionResult] veya [EmptyPrediction] (hata durumunda)
     */
    suspend fun predict(prefix: String): PredictionResult = withContext(inferenceDispatcher) {
        if (!isInitialized || isClosing) return@withContext EmptyPrediction
        val tkn = tokenizer ?: return@withContext EmptyPrediction
        val interp = interpreter ?: return@withContext EmptyPrediction

        if (prefix.trim().length < 2) return@withContext EmptyPrediction

        val generatedTokens = mutableListOf<String>()
        var currentText = prefix
        var totalLatency = 0L
        var firstTokenConfidence = 0f

        try {
            for (step in 0 until maxNewTokens) {
                val stepLatency = measureTimeMillis {
                    // 1. Tokenize
                    val inputIds = tkn.encode(currentText, maxLength = maxSeqLength)
                    val attentionMask = tkn.attentionMask(inputIds)

                    // 2. Giriş tensörlerini hazırla
                    val inputIdsTensor = createInt32Buffer(inputIds)
                    val attentionMaskTensor = createInt32Buffer(attentionMask)

                    // 3. Çıkış tensörünü hazırla
                    // [batch=1, seq_len, vocab_size]
                    val outputShape = interp.getOutputTensor(0).shape()
                    val vocabSize = outputShape[outputShape.size - 1]
                    val outputBuffer = ByteBuffer.allocateDirect(
                        1 * maxSeqLength * vocabSize * 4 // float32 → 4 byte
                    ).apply { order(ByteOrder.nativeOrder()) }

                    // 4. Çıkarım yap
                    interp.runForMultipleInputsOutputs(
                        arrayOf(inputIdsTensor, attentionMaskTensor),
                        mapOf(0 to outputBuffer),
                    )

                    // 5. Son gerçek token'ın logit'lerini al
                    outputBuffer.rewind()
                    val seqLen = attentionMask.sum()
                    val logits = FloatArray(vocabSize)
                    val offset = (seqLen - 1) * vocabSize
                    outputBuffer.position(offset * 4)
                    outputBuffer.asFloatBuffer().get(logits)

                    // 6. Temperature sampling ile sonraki token'ı seç
                    val (nextTokenId, confidence) = sampleToken(logits)
                    val nextToken = tkn.decode(intArrayOf(nextTokenId))

                    if (step == 0) firstTokenConfidence = confidence

                    // 7. Durdurma koşulları
                    if (confidence < minConfidence) return@measureTimeMillis
                    if (nextToken.isBlank()) return@measureTimeMillis
                    if (nextToken.trim() in setOf(".", "!", "?", "...")) {
                        generatedTokens.add(nextToken.trim())
                        return@measureTimeMillis
                    }
                    // EOS token kontrolü
                    if (nextTokenId == tkn.eosTokenId || nextTokenId == tkn.padTokenId) {
                        return@measureTimeMillis
                    }

                    generatedTokens.add(nextToken)
                    currentText = "$currentText $nextToken"
                }
                totalLatency += stepLatency

                // Eğer bu adımda token üretilmediyse dur
                if (generatedTokens.size <= step) break
            }
        } catch (e: Exception) {
            Log.e(TAG, "Çıkarım hatası: ${e.message}", e)
            return@withContext EmptyPrediction
        }

        if (generatedTokens.isEmpty()) return@withContext EmptyPrediction

        PredictionResult(
            tokens = generatedTokens,
            completion = generatedTokens.joinToString(" ").trim(),
            latencyMs = totalLatency,
            confidence = firstTokenConfidence,
        )
    }

    /**
     * Logit dizisinden temperature sampling ile token seç.
     *
     * Temperature:
     *  - 0.0: Greedy (en yüksek logit)
     *  - 0.5: Az çeşitlilik
     *  - 1.0: Ham softmax
     *  - >1.0: Daha rastgele
     *
     * @return Pair(token_id, confidence_score)
     */
    private fun sampleToken(logits: FloatArray): Pair<Int, Float> {
        if (temperature <= 0f) {
            // Greedy decoding
            val maxIdx = logits.indices.maxByOrNull { logits[it] } ?: 0
            return Pair(maxIdx, softmax(logits)[maxIdx])
        }

        // Temperature scaling
        val scaled = FloatArray(logits.size) { logits[it] / temperature }
        val probs = softmax(scaled)

        // Multinomial sampling
        val random = Math.random().toFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (random <= cumulative) {
                return Pair(i, probs[i])
            }
        }
        val fallback = probs.indices.maxByOrNull { probs[it] } ?: 0
        return Pair(fallback, probs[fallback])
    }

    /**
     * Numerically stable softmax.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxVal = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - maxVal).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    /**
     * IntArray'i TFLite'ın beklediği Int32 ByteBuffer'a çevir.
     */
    private fun createInt32Buffer(data: IntArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(data.size * 4)
            .apply { order(ByteOrder.nativeOrder()) }
        data.forEach { buffer.putInt(it) }
        buffer.rewind()
        return buffer
    }

    private fun IntArray.sum(): Int = fold(0) { acc, i -> acc + i }

    // ─── Lifecycle ───────────────────────────────────────────────────

    override fun close() {
        isClosing = true
        interpreter?.close()
        gpuDelegate?.close()
        nnapiDelegate?.close()
        interpreter = null
        gpuDelegate = null
        nnapiDelegate = null
        isInitialized = false
        Log.i(TAG, "GhostTextInterpreter kapatıldı.")
    }
}
