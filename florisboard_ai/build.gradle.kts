// =====================================================================
// build.gradle.kts (app modülü)
// Solenz AI Keyboard — FlorisBoard Entegrasyonu
// =====================================================================
// FlorisBoard'un mevcut build.gradle.kts dosyasına
// aşağıdaki bağımlılıkları ekleyin.
//
// MEVCUT DOSYAYA EKLENECEK KISIMLAR:
// =====================================================================

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
}

android {
    namespace = "dev.solenz.keyboard"
    compileSdk = 34

    defaultConfig {
        applicationId = "dev.solenz.keyboard"
        minSdk = 26         // TFLite GPU Delegate için minimum API 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf("-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi")
    }

    // TFLite model dosyasını assets'e sıkıştırma — kritik!
    // Sıkıştırılırsa model yüklenemez.
    androidResources {
        noCompress += listOf("tflite", "lite")
    }

    buildFeatures {
        viewBinding = true
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.8"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // ─── Mevcut FlorisBoard bağımlılıkları (değiştirme) ───────────────
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")

    // ─── Kotlin Coroutines ────────────────────────────────────────────
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")

    // ─── TensorFlow Lite — ANA BAĞIMLILIK ────────────────────────────
    // Core TFLite runtime
    implementation("org.tensorflow:tensorflow-lite:2.14.0")

    // GPU Delegate (destekleyen cihazlarda GPU hızlandırma)
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.14.0")

    // NNAPI Delegate (Android Neural Networks API)
    implementation("org.tensorflow:tensorflow-lite-api:2.14.0")

    // TFLite Support Library (tensör yönetimi kolaylıkları)
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // XNNPack (ARM NEON optimizasyonu — CPU hızlandırma)
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")

    // ─── Jetpack Compose ──────────────────────────────────────────────
    val composeBom = platform("androidx.compose:compose-bom:2024.02.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.animation:animation")
    implementation("androidx.activity:activity-compose:1.8.2")

    // ─── Lifecycle ────────────────────────────────────────────────────
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")

    // ─── Debug ────────────────────────────────────────────────────────
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    // ─── Test ─────────────────────────────────────────────────────────
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
