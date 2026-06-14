# ProGuard kuralları — TFLite sınıflarını obfuscation'dan koru
# Bu dosyayı FlorisBoard'un mevcut proguard-rules.pro dosyasına ekleyin.

# ─── TensorFlow Lite ──────────────────────────────────────────────────
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }
-keep class org.tensorflow.lite.support.** { *; }
-dontwarn org.tensorflow.lite.**

# ─── Solenz AI Engine ─────────────────────────────────────────────────
-keep class dev.solenz.keyboard.ai.** { *; }
-keepclassmembers class dev.solenz.keyboard.ai.** { *; }

# ─── Kotlin Coroutines ───────────────────────────────────────────────
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-dontwarn kotlinx.coroutines.**

# ─── JSON (tokenizer yükleme) ────────────────────────────────────────
-keepattributes Signature
-keepattributes *Annotation*
