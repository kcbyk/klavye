/**
 * =====================================================================
 * SolenzTheme.kt
 * Solenz AI Keyboard — Design System & Theme
 * =====================================================================
 * Tüm UI bileşenlerinin kullandığı merkezi design token sistemi.
 * Koyu mod öncelikli, neon vurgulu, glassmorphism uyumlu.
 */

package dev.solenz.keyboard.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// ─────────────────────────────────────────────────────────────────────
// RENK PALETİ — Solenz Brand Colors
// ─────────────────────────────────────────────────────────────────────
object SolenzColors {

    // Ana arka plan — derin, zengin koyu
    val Background         = Color(0xFF0A0A14)   // Derin lacivert-siyah
    val BackgroundElevated = Color(0xFF12121F)   // Hafif yükseltilmiş yüzey
    val Surface            = Color(0xFF1A1A2E)   // Kart/panel yüzeyi
    val SurfaceGlass       = Color(0x1AFFFFFF)   // Cam efekti — %10 beyaz

    // Neon Vurgular — AI kimliği
    val NeonPrimary    = Color(0xFF6E82FF)   // Elektrik mavisi
    val NeonSecondary  = Color(0xFF9B6EFF)   // Mor-lavendır
    val NeonTertiary   = Color(0xFF6EFFDB)   // Mint yeşil
    val NeonGlow       = Color(0x336E82FF)   // Primary'nin glow hali (%20 opaklık)
    val NeonGlowStrong = Color(0x666E82FF)   // Güçlü glow (%40 opaklık)

    // Ghost Text Renkleri
    val GhostText      = Color(0x99B4B4C3)   // %60 soğuk gri
    val GhostHighConf  = Color(0xCC6E82FF)   // Yüksek güven → neon mavi

    // Metin Hiyerarşisi
    val TextPrimary    = Color(0xFFF0F0FF)   // Neredeyse beyaz, hafif mavi ton
    val TextSecondary  = Color(0xFF8888AA)   // Soluk orta gri
    val TextHint       = Color(0xFF4A4A6A)   // En soluk metin

    // Durum Renkleri
    val Success  = Color(0xFF4EFFA0)   // Neon yeşil
    val Warning  = Color(0xFFFFB347)   // Amber
    val Error    = Color(0xFFFF4E6E)   // Neon kırmızı
    val Info     = Color(0xFF4ECBFF)   // Açık mavi

    // Sınır ve Ayırıcı
    val BorderSubtle = Color(0x1AFFFFFF)   // %10 beyaz border
    val BorderGlow   = Color(0x336E82FF)   // Neon glow border

    // Smartbar özel
    val SmartbarBg       = Color(0xCC0D0D1A)  // %80 koyu arka plan (glass için)
    val SmartbarBorder   = Color(0x266E82FF)  // Çok ince neon border
    val ConfidenceBar    = Color(0xFF6E82FF)  // Güven göstergesi dolgu rengi
    val ConfidenceBg     = Color(0x1A6E82FF)  // Güven göstergesi arka plan
}

// ─────────────────────────────────────────────────────────────────────
// TİPOGRAFİ — Inter font ailesi (Google Fonts)
// ─────────────────────────────────────────────────────────────────────
// NOT: Inter fontunu res/font/ klasörüne ekleyin veya
// Google Fonts dependency kullanın:
// implementation("androidx.compose.ui:ui-text-google-fonts:1.6.1")

object SolenzTypography {
    val DisplayLarge = TextStyle(
        fontWeight = FontWeight.Bold,
        fontSize = 24.sp,
        letterSpacing = (-0.5).sp,
        color = SolenzColors.TextPrimary,
    )
    val TitleMedium = TextStyle(
        fontWeight = FontWeight.SemiBold,
        fontSize = 16.sp,
        letterSpacing = 0.sp,
        color = SolenzColors.TextPrimary,
    )
    val BodyMedium = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        letterSpacing = 0.1.sp,
        color = SolenzColors.TextSecondary,
    )
    val LabelSmall = TextStyle(
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        letterSpacing = 0.5.sp,
        color = SolenzColors.TextHint,
    )
    val GhostText = TextStyle(
        fontWeight = FontWeight.Light,
        fontSize = 15.sp,
        letterSpacing = 0.2.sp,
        color = SolenzColors.GhostText,
    )
    val MonoCode = TextStyle(
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        color = SolenzColors.NeonPrimary,
    )
}

// ─────────────────────────────────────────────────────────────────────
// BOYUT SABİTLERİ
// ─────────────────────────────────────────────────────────────────────
object SolenzDimens {
    // Smartbar boyutları
    val SmartbarHeight  = 48
    val SmartbarPaddingH = 12
    val SmartbarPaddingV = 6

    // Border ve köşe yarıçapları
    val CornerSmall  = 8
    val CornerMedium = 12
    val CornerLarge  = 16
    val CornerFull   = 50   // Pill shape

    // Boşluklar
    val SpaceXS = 4
    val SpaceS  = 8
    val SpaceM  = 12
    val SpaceL  = 16
    val SpaceXL = 24

    // İkon boyutları
    val IconSmall  = 16
    val IconMedium = 20
    val IconLarge  = 24

    // Animasyon süreleri (ms)
    val AnimFast   = 150
    val AnimNormal = 300
    val AnimSlow   = 500
}
