# DeployBench Android app (Phase 2 step_04)

Gradle project added 2026-08-28. `android/` is a self-contained Gradle root
inside the repo — open **this directory** in Android Studio, not `repo/`.
`settings.gradle` includes only `:app`, so `android/harness/` (desktop-JVM
verification tools) stays out of the build without being moved.

Sources are compile-target minSdk 24 / target 34, arm64-v8a only.
**Never built or installed yet** — expect fix cycles on the first
`assembleRelease`.

## Build

```
cd repo/android
./gradlew assembleRelease
```

`gradlew` and `gradle-wrapper.jar` are **not in the repo yet** — only
`gradle/wrapper/gradle-wrapper.properties` is. Generate them once, either by
opening `repo/android/` in Android Studio (it offers to create the wrapper on
first sync) or, if a system Gradle is on PATH:

```
gradle wrapper --gradle-version 8.7
```

Then commit `gradlew`, `gradlew.bat` and `gradle/wrapper/gradle-wrapper.jar`
so a fresh clone can build. `.gitignore` deliberately does not exclude that jar.

APK lands at `app/build/outputs/apk/release/app-release.apk`.

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
The Android/Gradle toolchain is native code and this project root contains
`Tài liệu`. The same non-ASCII path already broke the ONNX C++ API once
(`infer_shapes_path` silently wrote nothing; `convert/ptq.py` now stages through
an ASCII temp dir). Build from an ASCII path — clone to e.g. `C:\dev\deploybench`
— and set `GRADLE_USER_HOME` to an ASCII directory.

**Worse than it looks:** OneDrive holds *two* directories that both render as
`Tài liệu` — one NFC-normalised, one NFD. NTFS does not normalise Unicode, so
they are distinct directories that Explorer cannot tell apart. The live project
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
`onCreate`. `am start` against a still-live instance delivers `onNewIntent`
instead, so the cell silently does not run. Worse, `run_cell.sh`'s poll loop has
no failure path: after 720 × 5 s it falls through, pulls an absent `out/`, and
prints `cell -> ...` as if it had succeeded. Required:

```
adb shell am force-stop org.deploybench.app
adb shell am start -n org.deploybench.app/.BatchActivity --ez batch true
```

and the poll loop needs a non-zero exit when `DONE` never appears.

## ORT version skew — report this, do not hide it

The scorer was certified on 2026-07-27 against the ORT **Java** binding
**1.27.0** on desktop x86 (max |d| 1.15e-05, Pearson 1.000000000000, 0/0/0 flips
vs the Python ONNX scores, all three precisions). Maven Central's newest
`onnxruntime-android` is **1.22.0** — the Android AAR lags the core release
train (core is at 1.29.0). So the device runs different native kernels, at a
different version, on a different ISA from the binding that certified the
scorer.

That does not invalidate the certification — it is exactly what the Stage P3
parity gate tests. But the **resolved** AAR version must be recorded in the P3
report, read from the build, not copied from this file.

`minSdk 24` is a guess against the AAR's actual floor; if 1.22.0 requires
higher, manifest merging fails loudly at build time, which is an acceptable way
to find out.

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
   200/20 rather than D23's >=1024 reps / >=60 s (C1), does not `force-stop`
   before `am start`, and reports success when the poll loop times out.
2. The resampler resolution needs a D-number.
3. `PREREGISTRATION.md` must be committed and tagged `roster-freeze` **before**
   the first device run, or the D30 restatements become deviations.
