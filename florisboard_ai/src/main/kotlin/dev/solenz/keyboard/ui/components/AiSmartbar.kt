/**
 * =====================================================================
 * AiSmartbar.kt
 * Solenz AI Keyboard — Premium AI Smartbar Composable
 * =====================================================================
 * FlorisBoard'un üst "Smartbar" bölümünü tamamen yeniden tasarlayan
 * glassmorphism + neon AI kontrol paneli.
 *
 * Bileşenler (soldan sağa):
 *  [AI Toggle] [Ghost Text Preview ←→→] [Confidence] [Latency] [Settings]
 *
 * Animasyonlar:
 *  - Ghost text slide-in (soldan)
 *  - AI toggle neon pulse
 *  - Confidence bar fill animasyonu
 *  - Commit checkmark burst
 *  - Loading shimmer
 */

package dev.solenz.keyboard.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.*
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.solenz.keyboard.ai.ghost.ActiveGhost
import dev.solenz.keyboard.ai.ghost.GhostTextControllerState
import dev.solenz.keyboard.ui.theme.SolenzColors
import dev.solenz.keyboard.ui.theme.SolenzTypography
import kotlinx.coroutines.delay

// ─────────────────────────────────────────────────────────────────────
// SMARTBAR STATE (ViewModel tarafından sağlanır)
// ─────────────────────────────────────────────────────────────────────

data class SmartbarUiState(
    val isAiEnabled: Boolean = true,
    val controllerState: GhostTextControllerState = GhostTextControllerState.Idle,
    val averageLatencyMs: Long = 0L,
    val isModelLoaded: Boolean = false,
)

// ─────────────────────────────────────────────────────────────────────
// ANA SMARTBAR COMPOSABLE
// ─────────────────────────────────────────────────────────────────────

/**
 * Premium AI Smartbar — FlorisBoard'un üst bar yerine kullanılır.
 *
 * @param uiState Smartbar'ın mevcut durumu
 * @param onToggleAi AI özelliğini aç/kapat callback
 * @param onGhostAccept Ghost text'i manuel kabul et
 * @param onGhostDismiss Ghost text'i reddet
 * @param onSettingsClick Ayarlar sayfasını aç
 */
@Composable
fun AiSmartbar(
    uiState: SmartbarUiState,
    onToggleAi: (Boolean) -> Unit = {},
    onGhostAccept: () -> Unit = {},
    onGhostDismiss: () -> Unit = {},
    onSettingsClick: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(52.dp)
            // Glassmorphism arka plan
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        Color(0xE60D0D1A),   // Sol → koyu
                        Color(0xE6121225),   // Sağ → hafif mor ton
                    )
                )
            )
            // Üst neon çizgi
            .drawBehind {
                drawLine(
                    brush = Brush.horizontalGradient(
                        colors = listOf(
                            Color.Transparent,
                            SolenzColors.NeonPrimary.copy(alpha = 0.6f),
                            SolenzColors.NeonSecondary.copy(alpha = 0.6f),
                            Color.Transparent,
                        )
                    ),
                    start = androidx.compose.ui.geometry.Offset(0f, 0f),
                    end = androidx.compose.ui.geometry.Offset(size.width, 0f),
                    strokeWidth = 1.dp.toPx(),
                )
            },
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {

            // ── 1. AI TOGGLE BUTONU ──────────────────────────────────
            AiToggleButton(
                isEnabled = uiState.isAiEnabled,
                isModelLoaded = uiState.isModelLoaded,
                onClick = { onToggleAi(!uiState.isAiEnabled) },
            )

            // ── 2. GHOST TEXT ÖNIZLEME ALANI (esnek) ────────────────
            GhostTextPreviewArea(
                state = uiState.controllerState,
                isAiEnabled = uiState.isAiEnabled,
                onAccept = onGhostAccept,
                onDismiss = onGhostDismiss,
                modifier = Modifier.weight(1f),
            )

            // ── 3. CONFIDENCE + LATENCY PANEL ───────────────────────
            AnimatedVisibility(
                visible = uiState.isAiEnabled && uiState.controllerState is GhostTextControllerState.Showing,
                enter = fadeIn() + scaleIn(),
                exit = fadeOut() + scaleOut(),
            ) {
                val showing = uiState.controllerState as? GhostTextControllerState.Showing
                if (showing != null) {
                    MetricsPanel(
                        confidence = showing.ghost.confidence,
                        latencyMs = uiState.averageLatencyMs,
                    )
                }
            }

            // ── 4. AYARLAR BUTONU ────────────────────────────────────
            SettingsIconButton(onClick = onSettingsClick)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// AI TOGGLE BUTONU
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun AiToggleButton(
    isEnabled: Boolean,
    isModelLoaded: Boolean,
    onClick: () -> Unit,
) {
    // Pulse animasyonu — AI aktifken
    val pulseAlpha by rememberInfiniteTransition(label = "pulse")
        .animateFloat(
            initialValue = 0.4f,
            targetValue = if (isEnabled) 1.0f else 0.4f,
            animationSpec = infiniteRepeatable(
                animation = tween(1200, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse,
            ),
            label = "pulseAlpha",
        )

    // Renk geçişi — aktif/pasif
    val buttonColor by animateColorAsState(
        targetValue = if (isEnabled) SolenzColors.NeonPrimary else SolenzColors.TextHint,
        animationSpec = tween(300),
        label = "btnColor",
    )

    Box(
        modifier = Modifier
            .size(36.dp)
            .clip(CircleShape)
            .background(
                color = if (isEnabled)
                    SolenzColors.NeonPrimary.copy(alpha = 0.15f)
                else
                    SolenzColors.Surface.copy(alpha = 0.5f)
            )
            .then(
                if (isEnabled) Modifier.border(
                    width = 1.dp,
                    brush = Brush.linearGradient(
                        listOf(
                            SolenzColors.NeonPrimary.copy(alpha = pulseAlpha),
                            SolenzColors.NeonSecondary.copy(alpha = pulseAlpha * 0.5f),
                        )
                    ),
                    shape = CircleShape,
                ) else Modifier.border(
                    width = 1.dp,
                    color = SolenzColors.BorderSubtle,
                    shape = CircleShape,
                )
            )
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        // Yükleniyor — shimmer ring
        if (!isModelLoaded) {
            LoadingRing(modifier = Modifier.size(20.dp))
        } else {
            // AI ikonu — spark emoji yerine çizgi tabanlı ikon
            AiSparkIcon(
                color = buttonColor,
                glowing = isEnabled,
                pulseAlpha = pulseAlpha,
            )
        }
    }
}

/**
 * Özel AI Spark ikonu — Canvas üzerinde çizilir.
 */
@Composable
private fun AiSparkIcon(
    color: Color,
    glowing: Boolean,
    pulseAlpha: Float,
) {
    Canvas(modifier = Modifier.size(18.dp)) {
        val w = size.width
        val h = size.height
        val cx = w / 2f
        val cy = h / 2f

        // Glow efekti
        if (glowing) {
            drawCircle(
                color = color.copy(alpha = pulseAlpha * 0.3f),
                radius = w * 0.6f,
                center = center,
            )
        }

        // Büyük dikey ışın
        drawLine(
            color = color,
            start = androidx.compose.ui.geometry.Offset(cx, cy - h * 0.42f),
            end = androidx.compose.ui.geometry.Offset(cx, cy + h * 0.42f),
            strokeWidth = 2.dp.toPx(),
            cap = StrokeCap.Round,
        )

        // Büyük yatay ışın
        drawLine(
            color = color,
            start = androidx.compose.ui.geometry.Offset(cx - w * 0.42f, cy),
            end = androidx.compose.ui.geometry.Offset(cx + w * 0.42f, cy),
            strokeWidth = 2.dp.toPx(),
            cap = StrokeCap.Round,
        )

        // Çapraz küçük ışınlar
        val smallLen = w * 0.25f
        val diags = listOf(
            Pair(-1f, -1f), Pair(1f, -1f), Pair(-1f, 1f), Pair(1f, 1f)
        )
        diags.forEach { (dx, dy) ->
            drawLine(
                color = color.copy(alpha = 0.7f),
                start = androidx.compose.ui.geometry.Offset(cx, cy),
                end = androidx.compose.ui.geometry.Offset(cx + dx * smallLen, cy + dy * smallLen),
                strokeWidth = 1.5.dp.toPx(),
                cap = StrokeCap.Round,
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// GHOST TEXT ÖNİZLEME ALANI
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun GhostTextPreviewArea(
    state: GhostTextControllerState,
    isAiEnabled: Boolean,
    onAccept: () -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier, contentAlignment = Alignment.CenterStart) {

        // ─── AI Kapalıyken ───────────────────────────────────────────
        AnimatedVisibility(
            visible = !isAiEnabled,
            enter = fadeIn(),
            exit = fadeOut(),
        ) {
            Text(
                text = "AI Ghost Text Kapalı",
                style = SolenzTypography.BodyMedium,
                color = SolenzColors.TextHint,
            )
        }

        // ─── Model Yüklenirken ───────────────────────────────────────
        AnimatedVisibility(
            visible = isAiEnabled && state is GhostTextControllerState.Idle,
            enter = fadeIn(),
            exit = fadeOut(),
        ) {
            ShimmerLoadingText()
        }

        // ─── Tahmin Hazır → Ghost önizleme ──────────────────────────
        AnimatedVisibility(
            visible = isAiEnabled && state is GhostTextControllerState.Showing,
            enter = slideInHorizontally(
                initialOffsetX = { it / 3 },
                animationSpec = spring(
                    dampingRatio = Spring.DampingRatioMediumBouncy,
                    stiffness = Spring.StiffnessMediumLow,
                )
            ) + fadeIn(tween(200)),
            exit = slideOutHorizontally(
                targetOffsetX = { -it / 4 },
                animationSpec = tween(150),
            ) + fadeOut(tween(150)),
        ) {
            val ghost = (state as? GhostTextControllerState.Showing)?.ghost
            if (ghost != null) {
                GhostPreviewChip(
                    ghost = ghost,
                    onAccept = onAccept,
                    onDismiss = onDismiss,
                )
            }
        }

        // ─── Commit animasyonu ───────────────────────────────────────
        var showCommit by remember { mutableStateOf(false) }
        LaunchedEffect(state) {
            if (state is GhostTextControllerState.Committed) {
                showCommit = true
                delay(600)
                showCommit = false
            }
        }
        AnimatedVisibility(
            visible = showCommit,
            enter = scaleIn(spring(Spring.DampingRatioLowBouncy)) + fadeIn(),
            exit = scaleOut() + fadeOut(),
        ) {
            CommitSuccessIndicator(
                text = (state as? GhostTextControllerState.Committed)?.text ?: ""
            )
        }

        // ─── Hazır ama ghost yok ────────────────────────────────────
        AnimatedVisibility(
            visible = isAiEnabled && state is GhostTextControllerState.Ready,
            enter = fadeIn(tween(400)),
            exit = fadeOut(tween(200)),
        ) {
            Text(
                text = "Yazmaya devam edin...",
                style = SolenzTypography.BodyMedium,
                color = SolenzColors.TextHint,
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// GHOST PREVIEW CHİP
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun GhostPreviewChip(
    ghost: ActiveGhost,
    onAccept: () -> Unit,
    onDismiss: () -> Unit,
) {
    // Confidence'a göre renk
    val chipColor = if (ghost.confidence > 0.7f)
        SolenzColors.NeonPrimary
    else
        SolenzColors.TextSecondary

    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        chipColor.copy(alpha = 0.12f),
                        SolenzColors.Surface.copy(alpha = 0.1f),
                    )
                )
            )
            .border(
                width = 1.dp,
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        chipColor.copy(alpha = 0.5f),
                        chipColor.copy(alpha = 0.1f),
                    )
                ),
                shape = RoundedCornerShape(20.dp),
            )
            .clickable(onClick = onAccept)
            .padding(horizontal = 10.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // Tab ikon
        Text(
            text = "⇥",
            fontSize = 11.sp,
            color = chipColor.copy(alpha = 0.8f),
        )

        // Ghost text
        Text(
            text = ghost.completion,
            style = SolenzTypography.BodyMedium.copy(
                color = chipColor.copy(alpha = 0.9f),
                fontSize = 13.sp,
            ),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )

        // Kapat butonu
        Spacer(Modifier.width(2.dp))
        Box(
            modifier = Modifier
                .size(16.dp)
                .clip(CircleShape)
                .background(SolenzColors.Surface.copy(alpha = 0.5f))
                .clickable(onClick = onDismiss),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "×",
                fontSize = 10.sp,
                color = SolenzColors.TextSecondary,
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// METRİKS PANEL (Confidence + Latency)
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun MetricsPanel(
    confidence: Float,
    latencyMs: Long,
) {
    // Confidence bar animasyonu
    val animatedConfidence by animateFloatAsState(
        targetValue = confidence,
        animationSpec = tween(600, easing = FastOutSlowInEasing),
        label = "confidence",
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(SolenzColors.Surface.copy(alpha = 0.3f))
            .padding(horizontal = 8.dp, vertical = 4.dp),
    ) {
        // Confidence bar
        Box(
            modifier = Modifier
                .width(36.dp)
                .height(3.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(SolenzColors.ConfidenceBg),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(animatedConfidence)
                    .clip(RoundedCornerShape(2.dp))
                    .background(
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                SolenzColors.NeonPrimary,
                                SolenzColors.NeonSecondary,
                            )
                        )
                    ),
            )
        }

        Spacer(Modifier.height(2.dp))

        // Latency badge
        Text(
            text = if (latencyMs > 0) "${latencyMs}ms" else "—",
            style = SolenzTypography.LabelSmall,
            color = when {
                latencyMs < 60  -> SolenzColors.Success
                latencyMs < 100 -> SolenzColors.Warning
                else            -> SolenzColors.Error
            },
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// AYARLAR İKON BUTONU
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun SettingsIconButton(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(32.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(SolenzColors.Surface.copy(alpha = 0.3f))
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Canvas(Modifier.size(16.dp)) {
            val cx = size.width / 2f
            val cy = size.height / 2f
            val color = SolenzColors.TextSecondary

            // Dış dişli daire
            drawCircle(
                color = color,
                radius = size.width * 0.38f,
                center = center,
                style = Stroke(width = 1.8.dp.toPx()),
            )
            // İç nokta
            drawCircle(
                color = color,
                radius = size.width * 0.12f,
                center = center,
            )
            // 6 dişli çıkıntı
            for (i in 0 until 6) {
                val angle = Math.toRadians(i * 60.0)
                val innerR = size.width * 0.38f
                val outerR = size.width * 0.52f
                drawLine(
                    color = color,
                    start = androidx.compose.ui.geometry.Offset(
                        cx + (innerR * cos(angle)).toFloat(),
                        cy + (innerR * sin(angle)).toFloat(),
                    ),
                    end = androidx.compose.ui.geometry.Offset(
                        cx + (outerR * cos(angle)).toFloat(),
                        cy + (outerR * sin(angle)).toFloat(),
                    ),
                    strokeWidth = 2.5.dp.toPx(),
                    cap = StrokeCap.Round,
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// COMMIT BAŞARI ANİMASYONU
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun CommitSuccessIndicator(text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(SolenzColors.Success.copy(alpha = 0.15f))
            .border(1.dp, SolenzColors.Success.copy(alpha = 0.4f), RoundedCornerShape(20.dp))
            .padding(horizontal = 10.dp, vertical = 4.dp),
    ) {
        // Checkmark
        Canvas(Modifier.size(14.dp)) {
            val w = size.width
            val h = size.height
            drawLine(
                color = SolenzColors.Success,
                start = androidx.compose.ui.geometry.Offset(w * 0.15f, h * 0.5f),
                end = androidx.compose.ui.geometry.Offset(w * 0.4f, h * 0.75f),
                strokeWidth = 2.dp.toPx(),
                cap = StrokeCap.Round,
            )
            drawLine(
                color = SolenzColors.Success,
                start = androidx.compose.ui.geometry.Offset(w * 0.4f, h * 0.75f),
                end = androidx.compose.ui.geometry.Offset(w * 0.85f, h * 0.25f),
                strokeWidth = 2.dp.toPx(),
                cap = StrokeCap.Round,
            )
        }
        Text(
            text = "\"$text\" eklendi",
            style = SolenzTypography.LabelSmall.copy(
                color = SolenzColors.Success,
                fontSize = 12.sp,
            ),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// LOADING / SHIMMER BİLEŞENLERİ
// ─────────────────────────────────────────────────────────────────────

@Composable
private fun LoadingRing(modifier: Modifier = Modifier) {
    val rotation by rememberInfiniteTransition(label = "loadingRing")
        .animateFloat(
            initialValue = 0f,
            targetValue = 360f,
            animationSpec = infiniteRepeatable(
                animation = tween(800, easing = LinearEasing),
            ),
            label = "rotation",
        )

    Canvas(modifier = modifier.rotate(rotation)) {
        // Dönen neon yay
        drawArc(
            brush = Brush.sweepGradient(
                colors = listOf(
                    Color.Transparent,
                    SolenzColors.NeonPrimary.copy(alpha = 0.3f),
                    SolenzColors.NeonPrimary,
                )
            ),
            startAngle = 0f,
            sweepAngle = 270f,
            useCenter = false,
            style = Stroke(
                width = 2.dp.toPx(),
                cap = StrokeCap.Round,
            ),
        )
    }
}

@Composable
private fun ShimmerLoadingText() {
    val shimmerTranslate by rememberInfiniteTransition(label = "shimmer")
        .animateFloat(
            initialValue = -1f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(1200, easing = LinearEasing),
                repeatMode = RepeatMode.Restart,
            ),
            label = "shimmerTranslate",
        )

    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(4.dp))
            .height(14.dp)
            .width(120.dp)
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        SolenzColors.Surface.copy(alpha = 0.3f),
                        SolenzColors.NeonPrimary.copy(alpha = 0.15f),
                        SolenzColors.Surface.copy(alpha = 0.3f),
                    ),
                    startX = shimmerTranslate * 200f + 100f,
                    endX = shimmerTranslate * 200f + 300f,
                )
            )
    )
}
