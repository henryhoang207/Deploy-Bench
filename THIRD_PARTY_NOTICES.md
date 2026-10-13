# Third-party notices

Checked against the original providers on 13 October 2026. See
[DATA_ACCESS.md](DATA_ACCESS.md) for downloads and access restrictions and
[LICENSING.md](LICENSING.md) for the artifact-level license map.

## Included third-party portions and notices

* **AASIST / AASIST-L:** copyright (c) 2021-present NAVER Corp. The training
  recipe in `train/aasist_r2_train.py` is adapted from
  [clovaai/aasist `main.py` at `a04c986`](https://github.com/clovaai/aasist/blob/a04c986/main.py).
  Its header and PREREGISTRATION_ROUND2.md identify the changes (evaluation-set
  checkpoint selection removed, explicit final/last weights, trimmed arm, logging).
  Retained MIT notice: [licenses/AASIST-MIT.txt](licenses/AASIST-MIT.txt).
  The ASVspoof 5 `Baseline-AASIST` implementation also carries the same NAVER MIT
  notice at the study's pinned `fe23d30` revision.
* **Gradle wrapper scripts/components:** copyright the original Gradle authors,
  as retained in `android/gradlew` and `android/gradlew.bat`; Apache License 2.0.
  Full text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
* **ASVspoof 2019, ASVspoof 2021, ASVspoof 5:** copied and normalized protocol
  information is attributed to their challenge/database authors. Contains
  information from [ASVspoof 2019](https://zenodo.org/records/6906306),
  [ASVspoof 2021](https://www.asvspoof.org/index2021.html), and
  [ASVspoof 5](https://zenodo.org/records/14498691), which are made available under
  the [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/).
  The full ODC-By 1.0 legal text from Open Data Commons is retained in
  [licenses/ODC-By-1.0.txt](licenses/ODC-By-1.0.txt). Selected/normalized IDs,
  cluster mappings, generated score columns and audit subsets are changes made
  by this study. Exact provider protocol copies compressed in
  `reproducibility/metadata/` are `ASVspoof2019.LA.cm.dev.trl.txt.gz`,
  `ASVspoof2019.LA.cm.eval.trl.txt.gz`, DF21 `trial_metadata.txt.gz`, and
  `ASVspoof5.dev.track_1.tsv.gz` / `ASVspoof5.eval.track_1.tsv.gz`. AS5 protocol
  originals were acquired from the dataset coauthor's pinned
  [Jung mirror](https://huggingface.co/datasets/jungjee/asvspoof5/tree/ebf8037ec54eaa5a054781ee34246206cef9fc11)
  and verified against the original provider README's MD5 values, as recorded in
  DATA_ACCESS.md and the reproduction source manifest; their original ODC-By
  terms and Zenodo dataset attribution are preserved.
* **In-the-Wild:** dataset authors Nicolas M. Müller, Pavel Czempin, Franziska
  Dieckmann, Adam Froghyar and Konstantin Böttinger; *Does Audio Deepfake
  Detection Generalize?* (Interspeech 2022). Source/provider and the conflicting
  Apache-2.0 / CC-BY-SA-4.0 license declarations are documented in DATA_ACCESS.md.
  This release uses CC-BY-SA-4.0 for linked score/metadata files, retaining the
  original provider's Apache-2.0 notice. Selected/normalized metadata and scores
  are changes made by this study. The exact ITW metadata copy in
  `reproducibility/metadata/meta.csv.gz` is compressed without content changes.
  No original audio is included.
* **MLAAD:** dataset authors Nicolas M. Müller, Piotr Kawa, Wei Herng Choong,
  Edresson Casanova, Eren Gölge, Thorsten Müller, Piotr Syga, Philip Sperl and
  Konstantin Böttinger; *MLAAD: The Multi-Language Audio Anti-Spoofing Dataset*
  (IJCNN 2024). [Provider](https://deepfake-total.com/mlaad),
  [creator's dataset card](https://huggingface.co/datasets/mueller91/MLAAD),
  CC-BY-NC-4.0. Five-language selection, normalized fields, M-AILABS pairing,
  checksums and scores are changes made by this study. Original audio is excluded.
* **M-AILABS:** copyright (c) 2017-2019 by the original creators @ M-AILABS.
  [Original creator's license](https://github.com/i-celeste-aurora/m-ailabs-dataset#license)
  is retained verbatim in [licenses/M-AILABS.txt](licenses/M-AILABS.txt), including
  conditions, disclaimer and non-endorsement term. Provider says its October 2018
  license supersedes older restrictions in download archives. The package uses
  the en/de/fr/it/pl material and does not include Ukrainian source audio.
* **MUSAN and RIRS_NOISES provenance:** stress manifests refer to MUSAN noise
  (David Snyder, Guoguo Chen, Daniel Povey; CC-BY-4.0,
  [OpenSLR 17](https://www.openslr.org/17)) and the AIR mock-up-phone responses in
  RIRS_NOISES (Apache-2.0, [OpenSLR 28](https://www.openslr.org/28)). Cite
  *MUSAN: A Music, Speech, and Noise Corpus* (2015) and *A Study on Data Augmentation
  of Reverberant Speech for Robust Speech Recognition* (2017) when recreating
  those conditions. Their waveform files are obtained separately.

## External implementations used by the study

These repositories/backbones are downloaded separately and are not included by
virtue of their being listed here. Their notices are supplied to make the
reproduction dependency chain explicit.

| Implementation | Original rights holder / verified terms | Notice or original source |
|---|---|---|
| ASVspoof 2021 RawNet2 baseline | Copyright (c) 2021 eurecom-asp, MIT; DF LICENSE is a symlink to LA baseline LICENSE | [licenses/RawNet2-MIT.txt](licenses/RawNet2-MIT.txt); [upstream](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2) |
| Res-TSSDNet | GNU GPL version 3; original model authors Guoying Hua, Andrew B. J. Teoh, Haijun Zhang | [licenses/GPL-3.0.txt](licenses/GPL-3.0.txt); [upstream](https://github.com/ghua-ac/end-to-end-synthetic-speech-detection). The GPL license of upstream model code is not replaced by the harness's MIT license. |
| SSL_Anti-spoofing (XLS-R+AASIST) | Copyright (c) 2022 Hemlata, MIT | [licenses/SSL-Anti-spoofing-MIT.txt](licenses/SSL-Anti-spoofing-MIT.txt); [upstream](https://github.com/TakHemlata/SSL_Anti-spoofing) |
| XLSR-Mamba | Copyright (c) 2025 Austin Xiao, MIT | [licenses/XLSR-Mamba-MIT.txt](licenses/XLSR-Mamba-MIT.txt); [pinned upstream](https://github.com/swagshaw/XLSR-Mamba/tree/10ec738). DF model card also declares MIT. |
| fairseq (pinned `a54021305d6b3c4c5959ac9395135f63202db8f1`) | Copyright (c) Facebook, Inc. and its affiliates, MIT | [licenses/fairseq-MIT.txt](licenses/fairseq-MIT.txt); [original license](https://github.com/facebookresearch/fairseq/blob/a54021305d6b3c4c5959ac9395135f63202db8f1/LICENSE) |
| Meta XLS-R 2B and 300M backbones | Hugging Face Meta model cards declare Apache-2.0; code implementations have their separate licenses | [2B card](https://huggingface.co/facebook/wav2vec2-xls-r-2b), [300M card](https://huggingface.co/facebook/wav2vec2-xls-r-300m) |
| aletheia (Pascu recipe reference) | No repository license verified; **source not redistributed** | [Original recipe](https://github.com/danoneata/aletheia). `models/wrap_pascu.py` and `train/pascu_fit.py` independently implement the described frozen-feature/logistic-regression method. |
| EnCodec | Copyright (c) Meta Platforms, Inc. and affiliates, MIT code | [Original license](https://github.com/facebookresearch/encodec/blob/main/LICENSE); pretrained models and waveform transformations are obtained separately. |

The full Res-TSSDNet GNU GPL license is retained for readers using that external
implementation. Numeric evaluation outputs do not copy its source; GPL section
2 states that output is covered only when the content constitutes a covered
work. This package does not redistribute a compiled Res-TSSDNet implementation
or claim MIT rights in upstream GPL code.

Installed libraries/runtimes (including PyTorch, NumPy, SciPy, scikit-learn,
soundfile, transformers, ONNX Runtime, ExecuTorch, FFmpeg and codec libraries) are
not redistributed as part of this source-and-results package. Their own notices
and any binary-distribution obligations apply to a separately assembled runtime
or application. Retain all provider notices in any local `vendor/` download.
