# DeployBench Android app (skeleton — Phase 2 step_04)

Kotlin sources below compile-target: minSdk 26, target 34. UNTESTED until
built in Android Studio; expect 2-3 fix cycles (gradle deps for ORT-Mobile
`com.microsoft.onnxruntime:onnxruntime-android` and ExecuTorch AAR — pin
versions at Phase 2 kickoff against current docs).

Modes (driven by /sdcard/deploybench/config.json):
- batch: score /sdcard/deploybench/in/*.wav -> out/scores.csv (+ latency.csv),
  write out/DONE marker (run_cell.sh polls it)
- soak:  loop inference for soak_minutes, log latency drift
- demo:  mic capture -> live score on screen (paper screenshot artifact)

Preprocessing contract: 16 kHz mono float32; pad-repeat/center-crop to the
model's fixed input; the resampler is OURS (Resample.kt) so desktop parity
is testable clip-for-clip (phase_2 step_03 feature-parity gate).
