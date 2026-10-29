# Model cards for the published audit

These cards describe the **eight studied detector configurations** and the
authors' retraining/head artifacts. Intended use is desktop research on
fixed-threshold speech-deepfake evaluation and reproducibility of the
published article. These are the
tested configurations, not a general certification of each model family.
Model-card text is CC-BY-4.0; implementation, weight and dataset licenses have
their own scope in [LICENSING.md](LICENSING.md) and
[DATA_ACCESS.md](DATA_ACCESS.md).

## Shared input and decision contract

Input is a mono waveform at **16,000 Hz**. [score.py](score.py) loads float32
samples, averages channels when needed and resamples other sample rates. For a
fixed input length, evaluation keeps the first `n` samples of a long clip and
repeat-tiles a short clip to `n`; it does not center-crop or zero-pad. Pascu LR
uses the whole utterance, with the documented exception below. CUDA fp32 scoring
disables TF32; loaded detectors are in evaluation mode.

The shared labels are `1 = bona fide`, `0 = spoof`; **higher scores mean more
bona fide**, and a clip is rejected as spoof when `score < tau`. Scores are not
interchangeable calibrated probabilities. Frozen thresholds come from **LA19
development data for seven detectors** and **ASVspoof 5 development data for AS5
AASIST**. Target-oracle thresholds are analysis comparisons, not deployment
thresholds. The authoritative settings are in [models/base.py](models/base.py)
and each linked wrapper; [wrapper contract checks](tests/test_wrapper_contracts.py)
and [round-2 checks](tests/test_round2.py) guard these conventions.

## Studied configurations

All downloaded upstream detector/backbone weights in this table are obtained
separately; this package releases their scores, thresholds and provenance.

| Detector / registry ID | Provenance and license | Evaluation input and exact wrapper score |
|---|---|---|
| **AASIST** / `aasist` | [clovaai/aasist](https://github.com/clovaai/aasist), released `models/weights/AASIST.pth`; LA19-trained; MIT, NAVER | **64,600** samples; raw bona fide logit `out[:, 1]` from the model's second returned object. [Wrapper](models/wrap_aasist.py) |
| **AASIST-L** / `aasist_l` | Same upstream, released `models/weights/AASIST-L.pth`; LA19-trained; MIT, NAVER | **64,600** samples; same score as AASIST. [Wrapper](models/wrap_aasist.py) |
| **RawNet2** / `rawnet2` | [ASVspoof 2021 RawNet2 baseline](https://github.com/asvspoof-challenge/2021/tree/main/DF/Baseline-RawNet2), `pre_trained_DF_RawNet2.pth`; LA19-trained; MIT code, eurecom-asp | **64,600** samples; author's bona fide output `out[:, 1]`. It is not replaced by a difference of class outputs. [Wrapper](models/wrap_rawnet2.py) |
| **Res-TSSDNet** / `tssdnet` | [ghua-ac/end-to-end-synthetic-speech-detection](https://github.com/ghua-ac/end-to-end-synthetic-speech-detection), released Res-TSSDNet checkpoint; LA19-trained; upstream GNU GPL v3 | **96,000** samples; `out[:, 0] - out[:, 1]`, the log-odds corresponding to the author's bona fide softmax probability. Its class order is opposite to AASIST's. [Wrapper](models/wrap_tssdnet.py) |
| **XLS-R+AASIST reference** / `xlsr_aasist` | [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing), released **DF-track** `Best_LA_model_for_DF.pth` plus original XLS-R 300M backbone; detector fine-tuned on LA19; MIT code | **64,600** samples; raw bona fide logit `out[:, 1]`. Unranked round-1 reference, excluded from primary-model hypothesis counts. [Wrapper](models/wrap_xlsr.py) |
| **AS5 AASIST** / `as5_aasist` | [ASVspoof 5 Baseline-AASIST](https://github.com/asvspoof-challenge/asvspoof5/tree/fe23d30/Baseline-AASIST) at `fe23d30`, released `models/weights/AASIST/best.pth`; AS5-trained; MIT, NAVER | **64,600** samples; raw bona fide logit `out[:, 1]`. Threshold source is AS5 dev. Training audio did not receive the non-speech post-processing applied to AS5 dev/eval. [Wrapper](models/wrap_aasist.py) |
| **Pascu LR** / `pascu_lr` | Independent implementation of [Pascu/aletheia's recipe](https://github.com/danoneata/aletheia), whose unlicensed source is not redistributed; frozen Meta [XLS-R 2B](https://huggingface.co/facebook/wav2vec2-xls-r-2b) at `3b6d89d` (Apache-2.0), mean-pooled features, author-fitted LA19 train-only linear head | **Whole utterance**; clips under **400** samples are repeat-tiled to 400 (DEV-9 amendment 5). Score `-(features @ coef + intercept)`, bona fide log-odds; `C = 1e6`, `max_iter = 5000`. [Wrapper](models/wrap_pascu.py) |
| **XLSR-Mamba DF** / `xlsr_mamba` | [swagshaw/XLSR-Mamba](https://github.com/swagshaw/XLSR-Mamba/tree/10ec738) at `10ec738`; [HF DF checkpoint](https://huggingface.co/AustinXiao/XLSR-Mamba-DF) at `8aed3da`; LA19-trained with RawBoost algo 3; MIT code/DF model | **66,800** samples on every study surface; raw bona fide logit `out[:, 1]`. The authors' published ITW comparison used their LA checkpoint and 64,600 samples, so it is not an identical configuration. [Wrapper](models/wrap_xlsr_mamba.py) |

## Author-trained artifacts included in the package

**AASIST-L runs:** `results/round2/runs/` contains untrimmed seeds 1–5 plus the
1234 pipeline check and trimmed seeds 1–5, with `swa.pth`, `last.pth`, epoch logs,
run metadata and SHA-256 records. Registry IDs `aasist_l_r2` and `aasist_l_trim`
require an explicit run checkpoint and retain AASIST-L's 64,600-sample/logit
contract. The trimmed arm applies the registered `vad35` energy VAD to training,
development and scoring audio **before** length adjustment. Primary weights are
the recipe's final SWA of snapshots eligible at dev EER <= 1%; without an
eligible snapshot, final weights equal the last epoch. Last-epoch aliases are
secondary comparisons. No evaluation-set checkpoint selection occurs in these
runs. Recipe, selection and modifications:
[registration](PREREGISTRATION_ROUND2.md), [training driver](train/aasist_r2_train.py).

**Pascu heads:** `results/round2/pascu/head_train.npz` is the train-only head used
in the audit; `head_traindev.npz` implements the authors' train-plus-dev fit solely
for the reproduction gate (`pascu_lr_gate`), outside analysis counts. JSON
sidecars record fit settings and feature/head hashes. These contain fitted
coefficients, not the separately downloaded 2B backbone.
[Fit code](train/pascu_fit.py). The authors' rights in their trained weights and
heads are released under **CC-BY-4.0**, with upstream/corpus attribution retained.

## Evidence boundaries and limitations

Use the article and its limitations section, the released score files and
registrations for performance evidence; this card adds no new measurements.
Released AASIST/AASIST-L checkpoints may have been selected using LA19 evaluation
data. The roster is not a representative sample of detector families; only
AASIST-L is retrained, and the trimmed/untrimmed arms differ in SWA eligibility.
ITW is a public-figure benchmark with 54 metadata speaker labels; DF21 results
use the documented 455,508-trial integrity-verified surface. MLAAD/M-AILABS has a
class-associated sample-rate difference, so its results do not isolate language
effects. Digital codecs, additive noise and impulse-response convolution do not
represent a full physical transmission or adaptive attack campaign.

The paper's evidence is **fp32 desktop scoring**. No phone measurement establishes
on-device numerical parity, latency, energy or thermal performance. Authenticity
decisions for an individual speaker, production service or forensic case need
their own validation. Source-audio reuse and account requirements, including
MLAAD's non-commercial license and gated access, remain as listed in
[DATA_ACCESS.md](DATA_ACCESS.md).
