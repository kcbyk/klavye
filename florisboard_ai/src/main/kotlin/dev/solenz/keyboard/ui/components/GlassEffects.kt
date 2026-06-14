/**
 * =====================================================================
 * GlassEffects.kt
 * Solenz AI Keyboard — Glassmorphism Effect System
 * =====================================================================
 * Tüm cam efektleri, neon glow ve blur modifier'larını içerir.
 *
 * Android'de gerçek BlurEffect API 31+ gerektirir.
 * Bu dosya:
 *  - API 31+: RenderEffect ile gerçek Gaussian Blur
 *  - API 26-30: Simüle edilmiş cam efekti (şeffaf katman + border)
 */

package dev.solenz.keyboard.ui.components

import android.graphics.RenderEffect
import android.graphics.Shader
import android.os.Build
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.solenz.keyboard.ui.theme.SolenzColors
import kotlin.math.cos
import kotlin.math.sin

// ─────────────────────────────────────────────────────────────────────
// GLASSMORPHISM MODİFİER
// ─────────────────────────────────────────────────────────────────────

/**
 * Bileşene cam efekti ekler.
 *
 * Görsel bileşenler:
 *  1. Yarı saydam koyu arka plan (frosted glass tabanı)
 *  2. İnce neon border (kenarlık parlaması)
 *  3. API 31+: Gaussian blur (gerçek cam)
 *  4. API <31: Gradient overlay (simüle)
 *
 * @param cornerRadius Köşe yuvarlaklığı
 * @param backgroundAlpha Arka plan opaklığı (0.0 - 1.0)
 * @param borderColor Kenarlık rengi (varsayılan: neon mavi)
 * @param borderWidth Kenarlık genişliği
 * @param blurRadius Blur miktarı (API 31+)
 */
fun Modifier.glassmorphism(
    cornerRadius: Dp = 16.dp,
    backgroundAlpha: Float = 0.15f,
    borderColor: Color = SolenzColors.BorderGlow,
    borderWidth: Dp = 1.dp,
    blurRadius: Float = 20f,
): Modifier = composed {
    val shape = RoundedCornerShape(cornerRadius)
    val bgColor = SolenzColors.Surface.copy(alpha = backgroundAlpha)

    this
        .clip(shape)
        .background(bgColor)
        .border(
            width = borderWidth,
            brush = Brush.linearGradient(
                colors = listOf(
                    borderColor.copy(alpha = 0.6f),
                    borderColor.copy(alpha = 0.1f),
                    borderColor.copy(alpha = 0.4f),
                ),
            ),
            shape = shape,
        )
        .then(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                // API 31+: Gerçek blur efekti
                Modifier.graphicsLayer {
                    renderEffect = RenderEffect
                        .createBlurEffect(blurRadius, blurRadius, Shader.TileMode.CLAMP)
                        .asComposeRenderEffect()
                }
            } else {
                // API <31: Gradient simülasyon
                Modifier.drawBehind {
                    drawGlassOverlay(backgroundAlpha)
                }
            }
        )
}

private fun DrawScope.drawGlassOverlay(alpha: Float) {
    drawRect(
        brush = Brush.verticalGradient(
            colors = listOf(
                Color.White.copy(alpha = alpha * 0.3f),
                Color.Transparent,
            ),
        ),
    )
}

// ─────────────────────────────────────────────────────────────────────
// NEON GLOW MODİFİER
// ─────────────────────────────────────────────────────────────────────

/**
 * Bileşenin etrafına neon parlama efekti ekler.
 *
 * Çoklu katmanlı shadow yaklaşımı:
 *  - İç katman: yoğun, küçük glow
 *  - Dış katman: soluk, geniş aura
 *
 * @param glowColor Parlama rengi
 * @param glowRadius Parlama yarıçapı
 * @param intensity Parlama yoğunluğu (0.0 - 1.0)
 */
fun Modifier.neonGlow(
    glowColor: Color = SolenzColors.NeonPrimary,
    glowRadius: Dp = 12.dp,
    intensity: Float = 0.6f,
): Modifier = this.drawBehind {
    val radiusPx = glowRadius.toPx()

    // Dış aura (geniş, soluk)
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                glowColor.copy(alpha = intensity * 0.3f),
                glowColor.copy(alpha = 0f),
            ),
            radius = radiusPx * 2.5f,
        ),
        radius = radiusPx * 2.5f,
        center = center,
    )

    // İç yoğun glow
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                glowColor.copy(alpha = intensity * 0.7f),
                glowColor.copy(alpha = 0f),
            ),
            radius = radiusPx,
        ),
        radius = radiusPx,
        center = center,
    )
}

/**
 * Dikdörtgen bileşen için neon border glow efekti.
 */
fun Modifier.neonBorderGlow(
    glowColor: Color = SolenzColors.NeonPrimary,
    cornerRadius: Dp = 16.dp,
    glowSpread: Dp = 8.dp,
    intensity: Float = 0.5f,
): Modifier = this.drawBehind {
    val spreadPx = glowSpread.toPx()
    val cornerPx = cornerRadius.toPx()

    // Dış glow katmanı
    for (i in 1..3) {
        val alpha = intensity * (0.4f / i)
        val spread = spreadPx * i

        drawRoundRect(
            color = glowColor.copy(alpha = alpha),
            topLeft = androidx.compose.ui.geometry.Offset(-spread, -spread),
            size = androidx.compose.ui.geometry.Size(
                size.width + spread * 2,
                size.height + spread * 2,
            ),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(
                cornerPx + spread,
                cornerPx + spread,
            ),
            style = Stroke(width = spreadPx * 0.5f),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// ANIMASYONLU NEON PULSE
// ─────────────────────────────────────────────────────────────────────

/**
 * AI aktif durumda neon pulsating animasyon.
 * Composable içinde kullanılır, animasyon state döndürür.
 */
@Composable
fun rememberPulseAnimation(
    targetAlpha: Float = 0.8f,
    initialAlpha: Float = 0.3f,
    durationMs: Int = 1500,
): androidx.compose.animation.core.Animatable<Float, *> {
    val alpha = androidx.compose.animation.core.remember {
        androidx.compose.animation.core.Animatable(initialAlpha)
    }

    androidx.compose.runtime.LaunchedEffect(Unit) {
        while (true) {
            alpha.animateTo(
                targetValue = targetAlpha,
                animationSpec = androidx.compose.animation.core.tween(
                    durationMillis = durationMs,
                    easing = androidx.compose.animation.core.FastOutSlowInEasing,
                ),
            )
            alpha.animateTo(
                targetValue = initialAlpha,
                animationSpec = androidx.compose.animation.core.tween(
                    durationMillis = durationMs,
                    easing = androidx.compose.animation.core.FastOutSlowInEasing,
                ),
            )
        }
    }

    return alpha
}
