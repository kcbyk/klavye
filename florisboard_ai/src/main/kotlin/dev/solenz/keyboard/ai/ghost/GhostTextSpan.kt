/**
 * =====================================================================
 * GhostTextSpan.kt
 * Solenz AI Keyboard — Ghost Text Görsel Katmanı
 * =====================================================================
 * Android'in Spannable sistemi kullanılarak ghost text'in
 * "gri/soluk" görünümünü oluşturan özel span tanımları.
 *
 * Ghost Text görsel hiyerarşisi:
 *  [Kullanıcı metni — beyaz/siyah]  [Ghost text — gri/soluk + italik]
 *       ↑ commitText ile kalıcı           ↑ setComposingText ile geçici
 *
 * Hangi renk uygulanır?
 *  - Koyu tema: #80FFFFFF (yarı saydam beyaz)
 *  - Açık tema: #80000000 (yarı saydam siyah)
 *  - Neon accent: #4D6EE7FF (Solenz marka rengi, confidence yüksekse)
 */

package dev.solenz.keyboard.ai.ghost

import android.graphics.Color
import android.text.TextPaint
import android.text.style.CharacterStyle
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.text.style.UpdateAppearance
import android.graphics.Typeface

/**
 * Ghost text için özel span — renk + opaklık + italik.
 *
 * ForegroundColorSpan'ı extend edemeyiz (final), bu yüzden
 * CharacterStyle'dan türetip updateDrawState'i override ediyoruz.
 */
class GhostTextSpan(
    private val color: Int = GHOST_COLOR_DARK,
    private val alpha: Int = 140,      // 0-255 (140 ≈ %55 opaklık)
    private val italic: Boolean = true,
) : CharacterStyle(), UpdateAppearance {

    companion object {
        /** Koyu tema ghost rengi — soğuk gri */
        val GHOST_COLOR_DARK = Color.argb(140, 180, 180, 195)

        /** Açık tema ghost rengi — koyu gri */
        val GHOST_COLOR_LIGHT = Color.argb(140, 80, 80, 90)

        /** Yüksek confidence ghost — solenz neon mavisi */
        val GHOST_COLOR_ACCENT = Color.argb(160, 110, 130, 231)

        /**
         * Tema ve confidence'a göre uygun span oluştur.
         */
        fun create(isDarkTheme: Boolean, confidence: Float): GhostTextSpan {
            val color = when {
                confidence > 0.7f -> GHOST_COLOR_ACCENT   // Çok güvenli → neon
                isDarkTheme -> GHOST_COLOR_DARK            // Koyu tema → gri
                else -> GHOST_COLOR_LIGHT                  // Açık tema → koyu gri
            }
            val alpha = (100 + (confidence * 100).toInt()).coerceIn(100, 200)
            return GhostTextSpan(color = color, alpha = alpha)
        }
    }

    override fun updateDrawState(tp: TextPaint) {
        // Renk ve opaklık uygula
        tp.color = color
        tp.alpha = alpha

        // İtalik stil — ghost text'i normal metinden ayırt et
        if (italic) {
            tp.typeface = Typeface.create(tp.typeface, Typeface.ITALIC)
        }
    }
}

/**
 * Composing text için kullanılacak span kombinasyonunu döndür.
 *
 * Android'in setComposingText() metodu, composing region'a
 * otomatik bir underline span ekler. Bunu override etmek için
 * kendi SpannableString'imizi kullanıyoruz.
 */
object GhostSpanFactory {

    /**
     * Composing + Ghost birleşik SpannableString oluştur.
     *
     * @param userText Kullanıcının yazdığı kısım
     * @param ghostText Model tahmininin gösterileceği kısım
     * @param isDarkTheme Tema modu
     * @param confidence Model güven skoru
     */
    fun buildCompositeSpannable(
        userText: String,
        ghostText: String,
        isDarkTheme: Boolean = true,
        confidence: Float = 0.5f,
    ): android.text.SpannableString {
        val fullText = "$userText $ghostText"
        val spannable = android.text.SpannableString(fullText)

        val ghostStart = userText.length + 1  // +1 boşluk için
        val ghostEnd = fullText.length

        if (ghostStart < ghostEnd) {
            // Ghost renk span'ı
            spannable.setSpan(
                GhostTextSpan.create(isDarkTheme, confidence),
                ghostStart,
                ghostEnd,
                android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
            )
        }

        return spannable
    }
}
