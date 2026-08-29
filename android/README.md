# DeployBench Android app (Phase 2 step_04)

Gradle project added 2026-08-28. `android/` is a self-contained Gradle root
inside the repo — open **this directory** in Android Studio, not the repo root.
`settings.gradle` includes only `:app`, so `android/harness/` (desktop-JVM
verification tools) stays out of the build without being moved.

**BUILD VERIFIED 2026-08-28.** `assembleRelease` succeeds and produces an
18.11 MB APK. This is a desktop build result only — the app has never been
installed or run on a handset.

## Verified toolchain

| Component | Version | Note |
|---|---|---|
| Gradle | 8.7 | via the committed wrapper |
| JDK | Temurin 21.0.12.1 LTS | `C:\tools\jdk21` |
| AGP | 8.5.2 | declared in root `build.gradle` |
| Kotlin plugin | 1.9.24 | declared in root `build.gradle` |
| compileSdk / targetSdk | 34 | `platforms;android-34` |
| Build-Tools | 34.0.0 | |
| ORT Android AAR | **1.22.0 (resolved)** | `releaseRuntimeClasspath` confirms |

Android Studio's bundled JBR is **25**, and the machine's system Java is **26**;
both are newer than Gradle 8.7 accepts. The build uses a separate JDK 21. If you
build from Studio, set Gradle JDK to 21 or the build will not start.

APK contents, checked with `aapt2 dump badging` and by listing the archive:

```
package        org.deploybench.app  versionName 0.1-p3
sdkVersion     24        targetSdkVersion 34
native-code    arm64-v8a  (only)
lib/arm64-v8a/libonnxruntime.so         17.37 MB
lib/arm64-v8a/libonnxruntime4j_jni.so    0.10 MB
classes.dex                              2.02 MB
permissions    MANAGE_EXTERNAL_STORAGE
               READ_EXTERNAL_STORAGE  maxSdkVersion 32
               WRITE_EXTERNAL_STORAGE maxSdkVersion 29
```

## Build

```
cd android
./gradlew assembleRelease
```

APK lands at `app/build/outputs/apk/release/app-release.apk`.
`local.properties` (holding `sdk.dir`) is gitignored — regenerate it per machine,
or export `ANDROID_HOME`.

**Build the `release` variant for anything you report.** The `debug` variant is
for wiring-up and crash triage only: a debuggable APK runs under different ART
behaviour and its p50/p90/p95 are not comparable to a shipped app. `release`
here is `debuggable false`, `minifyEnabled false`, signed with the debug key
purely so `adb install` works without a keystore.

The debug variant carries `applicationIdSuffix ".dbg"`, so it installs as
`org.deploybench.app.dbg` and **cannot** be driven by `run_cell.sh`
(`PKG=org.deploybench.app`). That mismatch is intentional: it makes it
impossible to record a measurement against the debuggable build by accident.

### Windows path warning
The Android/Gradle toolchain is native code and the OneDrive project root
contains `Tài liệu`. The same non-ASCII path already broke the ONNX C++ API once
(`infer_shapes_path` silently wrote nothing; `convert/ptq.py` now stages through
an ASCII temp dir). Build from the ASCII clone at `C:\dev\deploybench`, with
`GRADLE_USER_HOME=C:\gradle`.

**Worse than it looks:** OneDrive holds *two* directories that both render as
`Tài liệu` — one NFC-normalised, one NFD. NTFS does not normalise Unicode, so
they are distinct directories Explorer cannot tell apart. The live OneDrive copy
is the **NFD** one; a partial older copy of this project (including a second
TSSDNet vendor clone with checkpoints) sits under the NFC one at
`CS material\Mobile_Deepfake_Detection_Paper` — note the missing `Research\`
level. Confirm which tree you are in by content, not by the path string.

## Install and grant

```
adb install -r app/build/outputs/apk/release/app-release.apk
adb shell appops set --uid org.deploybench.app MANAGE_EXTERNAL_STORAGE allow
adb shell svc power stayon usb
```

The `appops` grant is **not optional on API 30+**. `BatchActivity` reads
`/sdcard/deploybench/` — written by `adb push`, i.e. by the shell user, so
app-specific storage is not an option without changing the device path in
`run_cell.sh`, `soak.sh` and `BatchActivity.kt` together. Granting from the host
keeps the campaign scriptable with no on-device tapping. Without it the first
run dies in `File(root, "config.json").readText()`.

`svc power stayon usb` matters for measurement, not convenience: if the screen
sleeps mid-cell the governor drops clocks and the latency distribution measures
the screen state. It is the ONLY screen-on mechanism here — the app declares no
WAKE_LOCK and sets no keepScreenOn flag.

## `run_cell.sh` must force-stop before starting

`BatchActivity` is `launchMode="singleTask"` and does all its work in
`onCreate`. Starting it against a still-live instance delivers `onNewIntent`
instead, so the cell silently does not run. Worse, `run_cell.sh`'s poll loop has
no failure path: after 720 x 5 s it falls through, pulls an absent `out/`, and
prints `cell -> ...` as if it had succeeded. Required:

```
adb shell am force-stop org.deploybench.app
adb shell am start -n org.deploybench.app/.BatchActivity --ez batch true
```

and the poll loop needs a non-zero exit when `DONE` never appears.

## ORT version skew — report this, do not hide it

The scorer was certified on 2026-07-27 against the ORT **Java** binding
**1.27.0** on desktop x86 (max |d| 1.15e-05, Pearson 1.000000000000, 0/0/0 flips
vs the Python ONNX scores, all three precisions). The Android AAR resolves to
**1.22.0** — Maven Central's newest for `onnxruntime-android`, which lags the
core release train (core is at 1.29.0). So the device runs different native
kernels, at a different version, on a different ISA from the binding that
certified the scorer.

That does not invalidate the certification — it is exactly what the Stage P3
parity gate tests. But it must be stated in the P3 report.

## Editing this manifest

`AndroidManifest.xml` failed its first build with
`ManifestMerger2$MergeFailureException: Error parsing`. Cause: **XML comments
cannot contain a double hyphen**, and the comments documented `adb` flags that
begin with one. Command lines with flags therefore live in this README, not in
manifest comments. If you add a comment there, check it first.

## Modes (driven by `/sdcard/deploybench/config.json`)
- **batch**: score `in/*.wav` -> `out/scores.csv` (+ `latency.csv`), write
  `out/DONE` (run_cell.sh polls it)
- **soak**: loop inference for `soak_minutes`, log latency drift
- **demo**: mic capture -> live score on screen (paper screenshot artifact) —
  *not implemented; no runtime permission flow exists for it, and the manifest
  declares the microphone `required="false"`*

`config.json` **must** set `input_samples`; `BatchActivity` refuses to run
without it rather than emit plausible wrong scores against a fixed-shape graph.
`scripts/run_cell.sh` does not yet write that key — see the open items below.

There is **no ExecuTorch runner** (OD-4), so this app is ORT-only and the first
full pathway is an ORT-only claim.

## Preprocessing contract
16 kHz mono float32; repeat-tile short clips and **truncate long ones FROM THE
START** (D10 — centre-cropping is banned and was a reproduction-breaking bug,
fixed in `Preprocess.kt` on 2026-07-27). Fixed input length is per model: 64,600
samples for AASIST / AASIST-L / RawNet2 / XLSR, **96,000 for Res-TSSDNet**. No
LFCC path — D30 dropped LFCC-LCNN, so every roster model is raw-waveform.

The resampler is OURS (`Preprocess.kt::resampleTo16k`) and it is **NOT
equivalent to `scipy.signal.resample_poly`**, which is what `score.py` uses to
produce every score this project is calibrated against — measured on a synthetic
probe: 28/40 clips exceed 1e-3, max |delta| 1.19, and the Kotlin side leaks
aliasing at complex ratios (OD-8). The 16 kHz control row is exactly 0.
**Preferred resolution: never resample on device** —
`scripts/make_device_wavs.py` already materialises the manifest as 16 kHz PCM16,
so guarantee `srIn == 16000` and let `resampleTo16k` early-return.

Preprocessing parity vs the Python reference currently **PASSES bit-identically**
(max |delta| = 0.000000e+00 over 1,000 clips, exercising both length branches),
run on the desktop JVM against the exact sources that ship in the APK.

## Open before the first device run
1. `scripts/run_cell.sh` writes no `input_samples` (A2), pushes the `.onnx`
   without its `.onnx.data` sidecar so fp32 lands weightless (A3), is still on
   200/20 rather than D23's >=1024 reps / >=60 s (C1), does not force-stop
   before starting the activity, and reports success when the poll loop times
   out.
2. The resampler resolution needs a D-number.
3. `PREREGISTRATION.md` must be committed and tagged `roster-freeze` **before**
   the first device run, or the D30 restatements become deviations.
4. `exports/` and `device_wavs/` are gitignored and exist only in the OneDrive
   tree; the device campaign needs them present next to the clone.
