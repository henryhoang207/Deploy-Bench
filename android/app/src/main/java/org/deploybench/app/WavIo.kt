package org.deploybench.app

import java.io.File

/** Minimal PCM16 WAV reader (mono/stereo), enough for the benchmark batches. */
object WavIo {
    fun sampleRate(f: File): Int {
        val b = f.readBytes()
        return le32(b, 24)
    }

    fun read(f: File): FloatArray {
        val b = f.readBytes()
        val channels = le16(b, 22)
        val bits = le16(b, 34)
        require(bits == 16) { "expect PCM16, got $bits-bit" }
        var off = 12
        while (!(b[off] == 'd'.code.toByte() && b[off + 1] == 'a'.code.toByte() &&
                 b[off + 2] == 't'.code.toByte() && b[off + 3] == 'a'.code.toByte())) {
            off += 8 + le32(b, off + 4)
        }
        val n = le32(b, off + 4) / 2 / channels
        val out = FloatArray(n)
        var p = off + 8
        for (i in 0 until n) {
            var acc = 0f
            for (c in 0 until channels) {
                val v = ((b[p + 1].toInt() shl 8) or (b[p].toInt() and 0xFF)).toShort()
                acc += v / 32768f; p += 2
            }
            out[i] = acc / channels
        }
        return out
    }

    private fun le16(b: ByteArray, o: Int) = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)
    private fun le32(b: ByteArray, o: Int) = le16(b, o) or (le16(b, o + 2) shl 16)
}
