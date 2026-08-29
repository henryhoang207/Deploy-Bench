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

        // RUNTIME MUST BE HONOURED, NOT RECORDED. This build has an ORT runner
        // and nothing else (OD-4: no ExecuTorch runner exists). Without this
        // check a cell configured runtime=executorch would run ORT and emit
        // scores LABELLED executorch -- the device-side twin of review defect
        // B2, and campaign.sh iterates RUNTIMES=(ort executorch), so it would
        // have fired on half of every phone's cells.
        val runtime = cfg.optString("runtime", "ort")
        require(runtime == "ort") {
            "config.json requests runtime='$runtime' but this build only has an " +
            "ORT runner (OD-4). Refusing to emit ORT scores under another label."
        }

        val scorer = OrtScorer(
            File(root, "models/${cfg.getString("model")}_${cfg.getString("precision")}.onnx").path)
        // NO DEFAULT. 64600 (AASIST) vs 96000 (TSSDNet) silently changes what
        // audio the model sees, and a fixed-shape graph would still run. A
        // missing key must fail loudly rather than emit plausible wrong scores.
        require(cfg.has("input_samples")) {
            "config.json must set input_samples (TSSDNet 96000, AASIST 64600)"
        }
        val inputSamples = cfg.getInt("input_samples")
        val wavs = File(root, "in").listFiles { f -> f.extension == "wav" }?.sorted() ?: listOf()
        val soakMin = cfg.optInt("soak_minutes", 0)

        fun prep(f: File): FloatArray = Preprocess.fixLength(
            Preprocess.resampleTo16k(WavIo.read(f), WavIo.sampleRate(f)), inputSamples)

        if (soakMin > 0) {
            // -----------------------------------------------------------
            // SOAK MODE, exclusively. The accuracy and latency legs are NOT
            // run here on purpose: a soak measures thermal drift from the
            // gated start temperature, and scoring ~1,000 clips first would
            // heat the device before the measurement begins -- destroying the
            // baseline the >35 C gate exists to establish. It also used to eat
            // into soak.sh's fixed polling budget.
            // -----------------------------------------------------------
            check(wavs.isNotEmpty()) { "soak mode needs at least one wav in in/" }
            val lat = StringBuilder("utt_id,query,ms\n")
            val end = System.currentTimeMillis() + soakMin * 60_000L
            var i = 0
            val w = prep(wavs[0])
            while (System.currentTimeMillis() < end) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                scorer.score(w)
                lat.append("soak,${i++},${(SystemClock.elapsedRealtimeNanos() - t0) / 1e6}\n")
            }
            File(outDir, "latency.csv").writeText(lat.toString())
            File(outDir, "run_meta.json").writeText(
                """{"mode":"soak","queries":$i,"soak_minutes":$soakMin,""" +
                """"input_samples":$inputSamples,"runtime":"$runtime"}""")
        } else {
            // -----------------------------------------------------------
            // ACCURACY LEG: exactly ONE inference per clip.
            // Scores are per-clip values; repeating a clip cannot change its
            // score and only inflates the run. This feeds the P3 parity gate.
            // -----------------------------------------------------------
            val scores = StringBuilder("utt_id,score\n")
            for (f in wavs) {
                scores.append("${f.nameWithoutExtension},${scorer.score(prep(f))}\n")
            }
            File(outDir, "scores.csv").writeText(scores.toString())

            // -----------------------------------------------------------
            // LATENCY LEG: MLPerf single-stream per D23 -- run until BOTH
            // >= min_queries inferences AND >= min_seconds have elapsed.
            // Supersedes the retired 200/20 scheme; "warmup" is deliberately
            // NOT a parameter because D23 records it is not an MLPerf concept.
            // The runtime is warm regardless: the accuracy leg above has
            // already executed one inference per clip before this loop starts.
            //
            // Queries cycle a bounded ring of PRE-PROCESSED waveforms so that
            // file I/O and resampling stay outside the timed region -- this
            // measures the model, not the loader. The ring is capped because
            // holding every clip would cost ~384 MB at 1,000 x 96,000 float32.
            // -----------------------------------------------------------
            require(cfg.has("min_queries") && cfg.has("min_seconds")) {
                "config.json must set min_queries and min_seconds (D23: >=1024, >=60)"
            }
            val minQueries = cfg.getInt("min_queries")
            val minSeconds = cfg.getInt("min_seconds")
            check(wavs.isNotEmpty()) { "latency leg needs at least one wav in in/" }

            val ringFiles = wavs.take(64)
            val ring = ringFiles.map { prep(it) }
            val lat = StringBuilder("utt_id,query,ms\n")
            var q = 0
            val t0 = SystemClock.elapsedRealtimeNanos()
            var elapsedS: Double
            do {
                val i = q % ring.size
                val s0 = SystemClock.elapsedRealtimeNanos()
                scorer.score(ring[i])
                val ms = (SystemClock.elapsedRealtimeNanos() - s0) / 1e6
                lat.append("${ringFiles[i].nameWithoutExtension},$q,$ms\n")
                q++
                elapsedS = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9
            } while (q < minQueries || elapsedS < minSeconds)
            File(outDir, "latency.csv").writeText(lat.toString())

            // Compliance record: the analysis layer must be able to VERIFY the
            // run met D23 rather than trust that it was configured to.
            File(outDir, "run_meta.json").writeText(
                """{"mode":"batch","queries":$q,"elapsed_s":$elapsedS,""" +
                """"min_queries":$minQueries,"min_seconds":$minSeconds,""" +
                """"ring":${ring.size},"clips":${wavs.size},""" +
                """"input_samples":$inputSamples,"runtime":"$runtime"}""")
        }

        val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi)
        File(outDir, "memory.txt").writeText("pss_kb=${mi.totalPss}")
        File(outDir, "DONE").writeText("ok")
        finish()
    }
}
