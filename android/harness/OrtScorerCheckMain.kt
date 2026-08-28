import org.deploybench.app.OrtScorer
import org.deploybench.app.Preprocess
import org.deploybench.app.WavIo
import java.io.File

/**
 * Executes the REAL OrtScorer.kt against the REAL exported graph, on the desktop JVM.
 *
 * WHY: the rank-1 output fix in OrtScorer.kt was reasoned about, not run. ORT-Java
 * returns FloatArray for a [B] tensor and Array<FloatArray> for [B,2], and the old
 * code cast unconditionally to the latter -- but "I believe the API does X" is
 * exactly the kind of claim this project has been burned by three times. The Java
 * ORT binding is the same native library the Android AAR wraps, so running it on
 * x86 here genuinely exercises the deserialisation path and the score convention.
 *
 * What this does NOT prove: anything about ARM kernels, or on-device numerics.
 * That is Stage P3 and still needs a phone. This only establishes that the app's
 * scoring code runs at all and produces the project's score convention.
 *
 * Compare the emitted csv against the Python ONNX scores for the same clips.
 */
fun main(args: Array<String>) {
    if (args.size < 5) {
        System.err.println("usage: OrtScorerCheckMain <onnx> <manifest> <wavDir> <out.csv> <inputSamples> [limit]")
        kotlin.system.exitProcess(2)
    }
    val onnx = args[0]
    val utts = File(args[1]).readLines().map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") }.map { it.split(Regex("\\s+"))[0] }
    val wavDir = File(args[2])
    val out = File(args[3])
    val inputSamples = args[4].toInt()
    val limit = if (args.size > 5) args[5].toInt() else utts.size

    println("model : $onnx")
    println("clips : ${minOf(limit, utts.size)}")

    val scorer = OrtScorer(onnx)          // the real class, unmodified
    val sb = StringBuilder("utt_id,score\n")
    var n = 0
    for (utt in utts) {
        if (n >= limit) break
        val f = File(wavDir, "$utt.wav")
        if (!f.exists()) continue
        val w = Preprocess.fixLength(
            Preprocess.resampleTo16k(WavIo.read(f), WavIo.sampleRate(f)), inputSamples)
        val s = scorer.score(w)           // <-- the line that used to throw
        sb.append("$utt,$s\n")
        n++
    }
    out.writeText(sb.toString())
    println("scored: $n -> ${out.path}")
}
