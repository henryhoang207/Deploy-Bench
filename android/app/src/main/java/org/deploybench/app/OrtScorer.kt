package org.deploybench.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** ORT-Mobile runner. Model assets under assets/models/<name>_<precision>.onnx */
class OrtScorer(modelPath: String, threads: Int = 1) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        session = env.createSession(modelPath, opts)
    }

    /** Returns bona-fide-ness score; higher = bona fide (project convention).
     *
     *  The export contract (convert/to_onnx.py + each wrapper's export_module())
     *  is ONE scalar per clip, shape [B], ALREADY in project score space --
     *  to_onnx.py refuses to export a wrapper that lacks export_module()
     *  precisely so the app never sees raw class logits. ORT-Java therefore
     *  returns a FloatArray here; the previous rank-2 cast to Array<FloatArray>
     *  threw ClassCastException on the first inference.
     *
     *  A rank-2 output means a raw [B,2] module slipped through, and the
     *  bona-fide class index is MODEL-SPECIFIC (TSSDNet 0, AASIST 1 --
     *  wrapper_audit.md rows 2-3). Guessing a sign here is exactly how an EER
     *  silently becomes 100-EER, so refuse rather than assume.
     */
    fun score(wave: FloatArray): Float {
        val shape = longArrayOf(1, wave.size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(wave), shape).use { t ->
            session.run(mapOf("waveform" to t)).use { r ->
                val v = r[0].value
                if (v is FloatArray) return v[0]
                throw IllegalStateException(
                    "export contract violated: expected a rank-1 [B] score output, got " +
                    "${v?.javaClass?.simpleName}. Re-export via export_module().")
            }
        }
    }
}
