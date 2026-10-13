# Data, model access, and release contents

Primary-source access and license information checked on 13 October 2026.
The public reproducibility package contains scores, frozen thresholds, evaluation
manifests, cluster mappings, registration snapshots, deviations, analysis code,
and the authors' round-2 trained weights/linear heads. Source audio, transformed
audio, downloaded third-party detector weights, XLS-R backbones, and vendored
detector repositories are obtained separately from their original providers.

Recomputing tables from released scores and metadata does not require downloading
audio, obtaining a provider account, or running neural-network inference.
Re-running scoring or training requires the separate resources below. The package
can be downloaded without registration; download access does not remove its
corpus-specific reuse conditions. See [LICENSING.md](LICENSING.md) for the exact
scope of the release's licenses and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for attribution notices.

## Corpora and stress sources

| Resource used in the paper | Original provider and version | License and access conditions |
|---|---|---|
| ASVspoof 2019 logical access (LA19), train/dev/eval | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336); [organisers' Zenodo deposit](https://zenodo.org/records/6906306), DOI 10.5281/zenodo.6906306 | Open Data Commons Attribution License (ODC-By 1.0), supplied as `LICENSE_text.txt` in the deposit. Public download; no application or login shown. The separate ASVspoof 2019 **real PA** agreement is not the license for the LA corpus used here. |
| ASVspoof 2021 deepfake (DF21) | [Zenodo, version 1.0](https://zenodo.org/records/4835108), DOI 10.5281/zenodo.4835108; [official keys and metadata](https://www.asvspoof.org/index2021.html) | The organisers explicitly offer the databases under the Open Data Commons Attribution License. Audio and full keys/metadata have public download links; no application or login shown. The paper's verified 455,508-trial surface is an author-selected subset of the 533,928 evaluation trials; the selected IDs and decoder audit are supplied. |
| ASVspoof 5, dev/eval (track 1) | [Zenodo, v0.1](https://zenodo.org/records/14498691), DOI 10.5281/zenodo.14498691; [actual license file](https://zenodo.org/records/14498691/files/LICENSE.txt?download=1) | ODC-By 1.0. The current post-challenge release is public; the old challenge-registration requirement is not required for these Zenodo downloads. Read its `README.txt` and `LICENSE.txt`. A request to the organisers is needed if creating spoofed data using the ASVspoof 5 protocol; that activity is not part of reproducing this paper's evaluation. |
| In-the-Wild (ITW) | [Fraunhofer AISEC original dataset page](https://deepfake-total.com/in_the_wild); [first author's Hugging Face release](https://huggingface.co/datasets/mueller91/In-The-Wild) | The original provider page declares **Apache-2.0** for the dataset/documentation; the same author's Hugging Face card currently declares **CC-BY-SA-4.0**. Both statements are preserved as a provider discrepancy, rather than asserting a single resolved upstream license. The release conservatively distributes ITW-linked scores and copied/normalized metadata under CC-BY-SA-4.0, and retains the Apache notice. The dataset ZIP is publicly linked; no gated-access application is shown. |
| MLAAD, five-language evaluation subset (MLAAD-5) | [Fraunhofer AISEC dataset page](https://deepfake-total.com/mlaad); [creator's Hugging Face repository](https://huggingface.co/datasets/mueller91/MLAAD). This study pins revision `30c3dec763fa4803f11c2df1557d3b0828395366` in `configs/mlaad5/snapshot_5lang.json`; “MLAAD-5” here means the **five languages en/de/fr/it/pl**, not a claim that the corpus is version 5. | **CC-BY-NC-4.0**; provider states non-commercial academic research use and no commercial use. The current Hugging Face repository is gated: login, consent to share contact information, and acceptance of provider conditions are required for source files. Public release of this study's derived non-audio scores/manifests does not provide access to the gated source audio or transfer an account's access authorization. |
| M-AILABS bona fide speech paired with MLAAD | [original creator's repository and download links](https://github.com/i-celeste-aurora/m-ailabs-dataset). English (US/UK), German, French, Italian and Polish; archive hashes in `configs/mlaad5/` | Custom permissive M-AILABS data license: redistribution, modification, and commercial use permitted while retaining the copyright, conditions, disclaimer, and non-endorsement condition. Do not label it a standard BSD license. Exact notice is retained in `licenses/M-AILABS.txt`. Public provider download links; no approval process stated. MLAAD-linked mixed protocol files also carry MLAAD's CC-BY-NC-4.0 terms. |
| MUSAN noise used in SNR conditions | [OpenSLR 17](https://www.openslr.org/17) | **CC-BY-4.0**, with Snyder, Chen and Povey attribution. Public archive downloads. Only noise-file identifiers/offsets and generated analysis outputs are included here. |
| AIR mock-up-phone impulse responses in RIRS_NOISES | [OpenSLR 28](https://www.openslr.org/28) | OpenSLR distributes RIRS_NOISES under **Apache-2.0**. The paper uses its 23 `air_type1_air_phone_*` responses, originating in the Aachen Impulse Response database; selected names and hashes are recorded in stress manifests. Public archive downloads; source impulse-response audio is not redistributed here. |

ODC-By permits extraction/reuse and derivative databases with notices (section
4.2), and requires a database attribution notice on publicly used outputs
(section 4.3). It addresses database rights; it does not grant every independent
right in each audio recording. This package releases non-audio evaluation
artifacts and preserves the original database terms for copied identifiers,
labels, speaker fields and protocols.

The package also includes exact ASVspoof 5 track-1 dev/eval protocol contents,
compressed as `reproducibility/metadata/ASVspoof5.dev.track_1.tsv.gz` and
`ASVspoof5.eval.track_1.tsv.gz`. Acquisition used
[coauthor Jung's pinned mirror](https://huggingface.co/datasets/jungjee/asvspoof5/resolve/ebf8037ec54eaa5a054781ee34246206cef9fc11/ASVspoof5_protocols.tar?download=true)
after the original Zenodo protocol download timed out. The extracted originals
match the provider README's MD5 values: dev
`e949dee1091a49b04943c548b5cf9860`, eval
`50c0591d45a282cb787a5a4614394f67`. These remain attributed to the original
Zenodo v0.1 dataset under ODC-By 1.0. Acquisition URLs and original/packaged
hashes are retained in `reproducibility/source_manifest.json`.

MLAAD/M-AILABS changes in the released protocol: five languages selected; all
284,000 spoof clips from the pinned snapshot retained; a seeded, count-matched
draw of 210,853 bona fide M-AILABS clips selected; paths/IDs and labels normalized
to the harness schema. English is imbalanced because its bona fide pool is
smaller. No transcripts or source waveform samples are needed for the released
cluster mappings. The license scope of a mixed file is documented in
[LICENSING.md](LICENSING.md).

## Detector code and checkpoints

| Paper model | Original implementation / checkpoint | Verified license and access |
|---|---|---|
| AASIST and AASIST-L | [clovaai/aasist](https://github.com/clovaai/aasist), including the released AASIST and AASIST-L weights; the round-2 recipe uses revision `a04c986` | Repository **MIT**, NAVER Corp.; public source/weight files. The authors' adapted training driver preserves its notice in this package. |
| RawNet2, DF21 baseline | [ASVspoof 2021 DF baseline](https://github.com/asvspoof-challenge/2021/tree/main/DF/Baseline-RawNet2); [official released checkpoint archive](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip) | Baseline code **MIT**, copyright 2021 eurecom-asp. DF's `LICENSE` points to the LA baseline's license. The official checkpoint is publicly downloadable; a separate weight-specific redistribution grant was not verified, so this package links the provider rather than redistributing it. |
| Res-TSSDNet | [ghua-ac/end-to-end-synthetic-speech-detection](https://github.com/ghua-ac/end-to-end-synthetic-speech-detection), model and `pretrained/` | Repository **GNU GPL version 3**. Obtain upstream source and weights directly; they are not relicensed MIT by this package. Runtime use of that implementation retains its GPL conditions. |
| XLS-R+AASIST reference | [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing); authors' DF-track model download linked from its README | Code **MIT**, copyright 2022 Hemlata. Released detector weights are supplied through an author-linked Google Drive folder; access depends on that link's current sharing policy, and a separate weight-specific license was not verified. The package does not mirror those weights. |
| AS5 AASIST | [ASVspoof 5 Baseline-AASIST at `fe23d30`](https://github.com/asvspoof-challenge/asvspoof5/tree/fe23d30/Baseline-AASIST); `models/weights/AASIST/best.pth` | This subdirectory carries NAVER's **MIT** license, even though the repository root has no single global license. The released baseline checkpoint is in the public upstream repository. Download it separately; it is not an author-trained weight from this study. |
| Pascu LR | Recipe described in [danoneata/aletheia](https://github.com/danoneata/aletheia); [XLS-R 2B backbone](https://huggingface.co/facebook/wav2vec2-xls-r-2b), pinned `3b6d89d` | **No license found** for aletheia; its source is not vendored or redistributed. This study independently implements the documented feature/linear-classifier recipe. The Meta backbone card declares **Apache-2.0** and is publicly accessible. The package contains only the authors' fitted linear heads, with train-only and train-plus-dev gate fits separately identified. |
| XLSR-Mamba DF | [swagshaw/XLSR-Mamba at `10ec738`](https://github.com/swagshaw/XLSR-Mamba/tree/10ec738); [AustinXiao/XLSR-Mamba-DF](https://huggingface.co/AustinXiao/XLSR-Mamba-DF), pinned `8aed3da` | Code and DF model card explicitly **MIT**, Austin Xiao; public sources and model files, no approval gate shown. Download the backbone and DF detector separately. |
| XLS-R 300M shared backbone | [Meta/fairseq original XLS-R release](https://github.com/facebookresearch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1/examples/wav2vec/xlsr); [Meta model card](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | fairseq code **MIT**; Hugging Face Meta model card **Apache-2.0**. Use the exact original checkpoint expected by the registered wrapper, not an arbitrary converted replacement. |
| EnCodec stress transform | [facebookresearch/encodec](https://github.com/facebookresearch/encodec) | Code **MIT**; models download automatically on first use from the upstream provider. This package does not contain EnCodec model files or encoded/decoded audio. |

The package includes eleven independently trained AASIST-L runs under
`results/round2/runs/` (untrimmed seeds 1–5 and 1234; trimmed seeds 1–5), with
`swa.pth`, `last.pth`, epoch logs, selection records and hashes. The authors release
their rights in these learned weights under **CC-BY-4.0**. The same applies to
`results/round2/pascu/head_train.npz` and `head_traindev.npz`. AASIST's MIT code
notice and ASVspoof 2019 attribution remain applicable. The Pascu heads contain
fitted linear coefficients, not the Meta XLS-R backbone.

Published numeric scores alone do not establish permission to redistribute a
third party's checkpoint. For upstream implementations and weights, use the
linked provider licenses and any supplied file-specific notices. This release
does not impose provider accounts or approval on its score-based reproduction.
