package org.deploybench.app

import android.app.Activity
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import org.json.JSONObject
import java.io.File

/** Batch + soak modes; driven by /sdcard/deploybench/config.json. */
class BatchActivity : Activity() {
    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        Thread { run() }.start()
    }

    private fun run() {
        val root = File("/sdcard/deploybench")
        val cfg = JSONObject(File(root, "config.json").readText())
        val outDir = File(root, "out").apply { mkdirs() }
        val scorer = OrtScorer(  // ExecuTorch runner added in fix-cycle
            File(root, "models/${cfg.getString("model")}_${cfg.getString("precision")}.onnx").path)
        // NO DEFAULT. 64600 (AASIST) vs 96000 (TSSDNet) silently changes what
        // audio the model sees, and a fixed-shape graph would still run. A
        // missing key must fail loudly rather than emit plausible wrong scores.
        require(cfg.has("input_samples")) {
            "config.json must set input_samples (TSSDNet 96000, AASIST 64600)"
        }
        val inputSamples = cfg.getInt("input_samples")
        val wavs = File(root, "in").listFiles { f -> f.extension == "wav" }?.sorted() ?: listOf()
        val scores = StringBuilder("utt_id,score\n")
        val lat = StringBuilder("utt_id,rep,ms\n")
        val warmup = cfg.optInt("warmup", 20)
        val reps = cfg.optInt("reps", 1)
        val soakMin = cfg.optInt("soak_minutes", 0)

        if (wavs.isNotEmpty()) {
            val w0 = Preprocess.fixLength(WavIo.read(wavs[0]), inputSamples)
            repeat(warmup) { scorer.score(w0) }
        }
        for (f in wavs) {
            val w = Preprocess.fixLength(
                Preprocess.resampleTo16k(WavIo.read(f), WavIo.sampleRate(f)), inputSamples)
            var s = 0f
            repeat(reps) { r ->
                val t0 = SystemClock.elapsedRealtimeNanos()
                s = scorer.score(w)
                val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                lat.append("${f.nameWithoutExtension},$r,$ms\n")
            }
            scores.append("${f.nameWithoutExtension},$s\n")
        }
        if (soakMin > 0 && wavs.isNotEmpty()) {           // soak mode: loop until time is up
            val end = System.currentTimeMillis() + soakMin * 60_000L
            var i = 0
            val w = Preprocess.fixLength(
                Preprocess.resampleTo16k(WavIo.read(wavs[0]), WavIo.sampleRate(wavs[0])), inputSamples)
            while (System.currentTimeMillis() < end) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                scorer.score(w)
                lat.append("soak,${i++},${(SystemClock.elapsedRealtimeNanos() - t0) / 1e6}\n")
            }
        }
        File(outDir, "scores.csv").writeText(scores.toString())
        File(outDir, "latency.csv").writeText(lat.toString())
        val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi)
        File(outDir, "memory.txt").writeText("pss_kb=${mi.totalPss}")
        File(outDir, "DONE").writeText("ok")
        finish()
    }
}
