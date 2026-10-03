package org.deploybench.app

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/** ExecuTorch (XNNPACK) runner, D48 / DEVIATIONS.md DEV-7. Model assets under
 *  models/<name>_<precision>.pte; run_cell.sh pushes only artifacts that passed the
 *  desktop fp32 gate (deploybench.artifacts.et_artifact). Every .pte takes and returns
 *  fp32 (the fp16 export casts inside the graph), so this class is precision-agnostic. */
class EtScorer(modelPath: String, threads: Int = 1) : Scorer {
    private val module: Module = Module.load(modelPath, Module.LOAD_MODE_FILE, threads)

    /** Same contract as OrtScorer: a [1] (or [1,1]) float output, nothing else. A
     *  [B,2] output would mean a raw module slipped through, and guessing the bona-fide
     *  index is how an EER silently becomes 100-EER, so refuse. */
    override fun score(wave: FloatArray): Float {
        val x = Tensor.fromBlob(wave, longArrayOf(1, wave.size.toLong()))
        val out = module.forward(EValue.from(x))
        check(out.size == 1) { "export contract violated: ${out.size} outputs, expected 1" }
        val t = out[0].toTensor()
        check(t.numel() == 1L) {
            "export contract violated: output shape ${t.shape().contentToString()}, " +
            "expected one score. Re-export via export_module()."
        }
        return t.dataAsFloatArray[0]
    }
}
