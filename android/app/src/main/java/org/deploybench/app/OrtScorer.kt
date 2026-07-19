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

    /** Returns bona-fide-ness score; higher = bona fide (project convention). */
    fun score(wave: FloatArray): Float {
        val shape = longArrayOf(1, wave.size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(wave), shape).use { t ->
            session.run(mapOf("waveform" to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val out = (r[0].value as Array<FloatArray>)[0]
                return if (out.size >= 2) out[1] - out[0] else out[0]
            }
        }
    }
}
