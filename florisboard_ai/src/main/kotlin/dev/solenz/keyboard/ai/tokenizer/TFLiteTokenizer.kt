/**
 * =====================================================================
 * TFLiteTokenizer.kt
 * Solenz AI Keyboard — FlorisBoard Entegrasyonu
 * =====================================================================
 * Python tarafında üretilen tokenizer.json dosyasını Android'de
 * çalıştıran BPE (Byte-Pair Encoding) tokenizer implementasyonu.
 *
 * Görevleri:
 *  1. assets/tokenizer/tokenizer.json dosyasını belleğe yükler
 *  2. encode(text) → IntArray (token ID listesi)
 *  3. decode(ids) → String (token ID'lerini metne çevirir)
 *  4. Qwen2 özel tokenları (ChatML) yönetir
 *
 * Kütüphane bağımlılığı gerektirmez — saf Kotlin implementasyonu.
 */

package dev.solenz.keyboard.ai.tokenizer

import android.content.Context
import android.content.res.AssetManager
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

/**
 * Qwen2 BPE Tokenizer'ının Android implementasyonu.
 *
 * Thread-safe: Tüm public metodlar read-only işlem yapar,
 * yükleme bir kez init'te gerçekleşir.
 */
class TFLiteTokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val merges: List<Pair<String, String>>,
    private val specialTokens: Map<String, Int>,
    private val idToToken: Map<Int, String>,
) {

    // ─── Özel Token ID'leri ───────────────────────────────────────────
    val padTokenId: Int = specialTokens["<|endoftext|>"] ?: 0
    val eosTokenId: Int = specialTokens["<|im_end|>"] ?: padTokenId
    val unkTokenId: Int = specialTokens["<unk>"] ?: 0

    companion object {
        private const val TOKENIZER_ASSET_PATH = "tokenizer/tokenizer.json"
        private const val BYTE_LEVEL_OFFSET = 256

        /**
         * Factory method — AssetManager üzerinden tokenizer yükle.
         * Bu işlem IO içerdiğinden mutlaka arka planda çağrılmalıdır.
         */
        fun fromAssets(context: Context): TFLiteTokenizer {
            val json = loadJsonFromAssets(context.assets, TOKENIZER_ASSET_PATH)
            return parse(json)
        }

        private fun loadJsonFromAssets(assets: AssetManager, path: String): JSONObject {
            val reader = BufferedReader(
                InputStreamReader(assets.open(path), StandardCharsets.UTF_8)
            )
            val content = reader.use { it.readText() }
            return JSONObject(content)
        }

        private fun parse(json: JSONObject): TFLiteTokenizer {
            val vocab = mutableMapOf<String, Int>()
            val idToToken = mutableMapOf<Int, String>()
            val specialTokens = mutableMapOf<String, Int>()
            val merges = mutableListOf<Pair<String, String>>()

            // vocab.json içindeki token → ID eşlemeleri
            val modelObj = json.optJSONObject("model")
            val vocabObj = modelObj?.optJSONObject("vocab")
            if (vocabObj != null) {
                val keys = vocabObj.keys()
                while (keys.hasNext()) {
                    val token = keys.next()
                    val id = vocabObj.getInt(token)
                    vocab[token] = id
                    idToToken[id] = token
                }
            }

            // Merge kuralları
            val mergesArray = modelObj?.optJSONArray("merges")
            if (mergesArray != null) {
                for (i in 0 until mergesArray.length()) {
                    val merge = mergesArray.getString(i).split(" ")
                    if (merge.size == 2) {
                        merges.add(Pair(merge[0], merge[1]))
                    }
                }
            }

            // Özel tokenlar
            val addedTokens = json.optJSONArray("added_tokens")
            if (addedTokens != null) {
                for (i in 0 until addedTokens.length()) {
                    val tokenObj = addedTokens.getJSONObject(i)
                    val content = tokenObj.getString("content")
                    val id = tokenObj.getInt("id")
                    specialTokens[content] = id
                    vocab[content] = id
                    idToToken[id] = content
                }
            }

            return TFLiteTokenizer(vocab, merges, specialTokens, idToToken)
        }
    }

    // ─── Merge öncelik tablosu (merge sırası = öncelik) ──────────────
    private val mergeRanks: Map<Pair<String, String>, Int> by lazy {
        merges.mapIndexed { index, pair -> pair to index }.toMap()
    }

    // ─── Byte-Level BPE Encoding ──────────────────────────────────────

    /**
     * Metni token ID dizisine çevirir.
     *
     * @param text Encode edilecek metin
     * @param maxLength Maksimum token sayısı (padding/truncation uygulanır)
     * @param addSpecialTokens ChatML özel token'larını ekle
     * @return IntArray: [input_ids]
     */
    fun encode(
        text: String,
        maxLength: Int = 64,
        addSpecialTokens: Boolean = true,
    ): IntArray {
        val tokens = mutableListOf<Int>()

        if (addSpecialTokens) {
            // <|im_start|>user\n
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
            tokens.addAll(encodeText("user\n"))
        }

        tokens.addAll(encodeText(text))

        if (addSpecialTokens) {
            // <|im_end|>\n<|im_start|>assistant\n
            eosTokenId.let { tokens.add(it) }
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
            tokens.addAll(encodeText("assistant\n"))
        }

        // Truncate
        val truncated = if (tokens.size > maxLength) tokens.take(maxLength) else tokens

        // Pad
        val padded = IntArray(maxLength) { padTokenId }
        truncated.forEachIndexed { i, id -> padded[i] = id }

        return padded
    }

    /**
     * Token ID dizisini metne çevirir.
     */
    fun decode(ids: IntArray, skipSpecialTokens: Boolean = true): String {
        val sb = StringBuilder()
        for (id in ids) {
            val token = idToToken[id] ?: continue
            if (skipSpecialTokens && specialTokens.containsValue(id)) continue
            // Ġ (U+0120) → boşluk, BPE encoding convention
            sb.append(token.replace("Ġ", " ").replace("Ċ", "\n"))
        }
        return sb.toString().trim()
    }

    /**
     * Attention mask üret (padding konumları 0, gerçek tokenlar 1).
     */
    fun attentionMask(inputIds: IntArray): IntArray {
        return IntArray(inputIds.size) { i ->
            if (inputIds[i] != padTokenId) 1 else 0
        }
    }

    // ─── İç Yardımcı Metodlar ────────────────────────────────────────

    private fun encodeText(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()

        // Kelime bazlı tokenization
        val words = preTokenize(text)
        val tokenIds = mutableListOf<Int>()

        for (word in words) {
            val wordTokens = bpeEncode(word)
            tokenIds.addAll(wordTokens.mapNotNull { vocab[it] ?: unkTokenId })
        }

        return tokenIds
    }

    /**
     * Metni kelimelere böl (boşluk bazlı, Türkçe uyumlu).
     * Qwen2 byte-level BPE için boşlukları Ġ ile işaretle.
     */
    private fun preTokenize(text: String): List<String> {
        val words = mutableListOf<String>()
        var i = 0
        val chars = text.toList()

        while (i < chars.size) {
            val sb = StringBuilder()
            // İlk karakter veya boşluk sonrası
            if (i == 0 || chars[i - 1] == ' ') {
                if (chars[i] != ' ') {
                    sb.append("Ġ")
                }
            }
            while (i < chars.size && chars[i] != ' ') {
                sb.append(chars[i])
                i++
            }
            if (sb.isNotEmpty()) words.add(sb.toString())
            if (i < chars.size && chars[i] == ' ') i++ // boşluğu atla
        }
        return words
    }

    /**
     * Tek kelime için BPE merge işlemi.
     */
    private fun bpeEncode(word: String): List<String> {
        if (word.isEmpty()) return emptyList()

        // Kelimeyi karakterlere böl
        var symbols = word.map { it.toString() }.toMutableList()

        // Merge kurallarını uygula
        while (symbols.size > 1) {
            // Mevcut sembol çiftleri arasında en düşük rank'lı merge'ü bul
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1

            for (i in 0 until symbols.size - 1) {
                val pair = Pair(symbols[i], symbols[i + 1])
                val rank = mergeRanks[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if (bestIdx == -1) break // Uygulanacak merge kalmadı

            // Merge uygula
            val merged = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols = (symbols.take(bestIdx) + merged + symbols.drop(bestIdx + 2)).toMutableList()
        }

        return symbols
    }

    /**
     * Vocabulary boyutu.
     */
    val vocabSize: Int get() = vocab.size
}
