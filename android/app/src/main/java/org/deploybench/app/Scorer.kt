package org.deploybench.app

/** One runtime's scorer. score() returns the bona-fide-ness score (higher = bona fide),
 *  ONE scalar per clip, already in project score space (the export_module() contract). */
interface Scorer {
    fun score(wave: FloatArray): Float
}
