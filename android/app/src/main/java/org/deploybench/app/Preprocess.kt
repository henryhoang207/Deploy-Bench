package org.deploybench.app

/** Preprocessing bit-matched (within 1e-6) to score.py::fix_length + load_wav. */
object Preprocess {
    fun fixLength(wave: FloatArray, n: Int): FloatArray {
        // TRUNCATE FROM THE START, never centre-crop. Authors' eval convention
        // (aasist data_utils.pad; TSSDNet data prep `x[0:dur*fs]`) -- see
        // wrapper_audit.md row 4, where centre-cropping is recorded as a
        // reproduction-breaking bug already fixed once on the Python side.
        // Must stay bit-identical to score.py::fix_length or Stage P3 parity
        // fails for a reason that has nothing to do with the device.
        if (wave.size >= n) return wave.copyOfRange(0, n)
        val out = FloatArray(n)
        var i = 0
        while (i < n) { out[i] = wave[i % wave.size]; i++ }
        return out
    }

    /** Windowed-sinc polyphase resampler to 16 kHz (matches scipy resample_poly
     *  closely enough for the decision-flip parity gate; exact comparison is
     *  part of phase_2 step_03). */
    fun resampleTo16k(wave: FloatArray, srIn: Int): FloatArray {
        if (srIn == 16000) return wave
        val g = gcd(srIn, 16000); val up = 16000 / g; val down = srIn / g
        val taps = 64 * maxOf(up, down)
        val cutoff = 0.5f / maxOf(up, down)
        val h = FloatArray(2 * taps + 1)
        for (k in -taps..taps) {
            val x = k.toFloat()
            val sinc = if (k == 0) 2f * cutoff else
                (Math.sin(2.0 * Math.PI * cutoff * x) / (Math.PI * x)).toFloat()
            val win = (0.54 + 0.46 * Math.cos(Math.PI * k / taps)).toFloat()
            h[k + taps] = sinc * win * up
        }
        val outLen = (wave.size.toLong() * up / down).toInt()
        val out = FloatArray(outLen)
        for (m in 0 until outLen) {
            val t = m.toLong() * down
            val phase = (t % up).toInt(); val base = (t / up).toInt()
            var acc = 0f
            var k = -taps + phase
            while (k <= taps) {
                val idx = base - (k - phase) / up
                if (idx in wave.indices) acc += h[k + taps] * wave[idx]
                k += up
            }
            out[m] = acc
        }
        return out
    }

    private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
}
