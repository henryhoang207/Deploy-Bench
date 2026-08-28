import org.deploybench.app.Preprocess
import org.deploybench.app.WavIo
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Stage P3, preprocessing leg -- run on the DESKTOP JVM, no phone required.
 *
 * experiment_protocol.md Stage P3 asks for a "separate parity check for
 * preprocessing: on-device feature extraction (resampling, padding) vs Python
 * reference". Preprocess.kt and WavIo.kt carry no Android imports, so the
 * EXACT SOURCES THAT SHIP IN THE APK can be exercised here against the same
 * 1,000-clip manifest and diffed sample-for-sample with score.py.
 *
 * SCOPE, stated honestly: this validates the preprocessing contract only. It is
 * NOT the P3 score gate -- that needs the ONNX graph running on real ARM. What
 * it does buy is that when a phone finally appears, a decision flip cannot be
 * blamed on padding or resampling, because those were settled here.
 *
 * Emits the post-fixLength waveform for every clip as raw little-endian float32
 * (matching numpy's '<f4') plus an index csv, so the Python side can compute a
 * true max|delta| rather than comparing summary statistics.
 */
fun main(args: Array<String>) {
    if (args.size < 5) {
        System.err.println(
            "usage: PreprocParityMain <manifest> <wavDir> <outBin> <outIndex> <inputSamples>")
        kotlin.system.exitProcess(2)
    }
    val manifest = File(args[0])
    val wavDir = File(args[1])
    val outBin = File(args[2])
    val outIndex = File(args[3])
    val inputSamples = args[4].toInt()

    val utts = manifest.readLines()
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") }
        .map { it.split(Regex("\\s+"))[0] }

    println("manifest      : ${manifest.path} (${utts.size} clips)")
    println("wav dir       : ${wavDir.path}")
    println("input samples : $inputSamples")

    val idx = StringBuilder("utt_id,offset,n,src_samples,src_sr\n")
    var offset = 0L
    var missing = 0
    val buf = ByteBuffer.allocate(inputSamples * 4).order(ByteOrder.LITTLE_ENDIAN)

    BufferedOutputStream(FileOutputStream(outBin), 1 shl 20).use { out ->
        for ((i, utt) in utts.withIndex()) {
            val f = File(wavDir, "$utt.wav")
            if (!f.exists()) { missing++; continue }
            val sr = WavIo.sampleRate(f)
            val raw = WavIo.read(f)
            // Same order as BatchActivity: resample first, then fix length.
            val wave = Preprocess.fixLength(Preprocess.resampleTo16k(raw, sr), inputSamples)
            check(wave.size == inputSamples) { "$utt: got ${wave.size}, want $inputSamples" }

            buf.clear()
            for (v in wave) buf.putFloat(v)
            out.write(buf.array())

            idx.append("$utt,$offset,$inputSamples,${raw.size},$sr\n")
            offset += inputSamples.toLong() * 4
            if ((i + 1) % 200 == 0) println("  ${i + 1}/${utts.size}")
        }
    }
    outIndex.writeText(idx.toString())
    println("\nwritten : ${outBin.path} (${outBin.length()} bytes)")
    println("index   : ${outIndex.path}")
    println("missing : $missing")
    if (missing > 0) kotlin.system.exitProcess(1)
}
