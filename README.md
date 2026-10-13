# DeployBench

Reproducibility package for **Thresholded Once, Deployed Elsewhere: A Two-Round
Pre-Registered Audit of Threshold Transfer in Speech Deepfake Detection**
(Speech Communication manuscript).

The version associated with the manuscript is
[speechcomm-v1.0.0](https://github.com/henryhoang207/Deploy-Bench/releases/tag/speechcomm-v1.0.0).
Download the release archive or clone this repository and check out that tag.
The package can be downloaded without an account, registration, or approval.
Use the fixed release when reproducing or citing the article; the main branch
may subsequently change.

## Reproduce the paper without downloading audio or upstream checkpoints

Use Python 3.12 or newer in a fresh virtual environment:

```sh
git clone https://github.com/henryhoang207/Deploy-Bench.git
cd Deploy-Bench
git checkout speechcomm-v1.0.0
python -m pip install -r requirements-reproduce.txt
python scripts/reproduce_paper.py --verify
python paper/speechcomm/audit_claims.py
python -m pytest tests/ -q
```

The reproduction command regenerates the paper's tables and figures into a
temporary directory and checks the generated tables against the published
source. The claim audit checks the numerical statements against the included
results and historical registrations. Neither command needs access to OneDrive,
the authors' server, source audio, or upstream checkpoints. Remaining author
information in the manuscript is reported by the claim audit; it is separate
from the numerical checks.
Tests that require optional neural inference runtimes are skipped when those
runtimes are absent; the score-based reproduction does not require them.

To regenerate the checked-in table and figure files deliberately:

```sh
python scripts/reproduce_paper.py --write
```

To build the article and supplement, install a LaTeX distribution with
`pdflatex`, `bibtex`, and `elsarticle`, then run:

```sh
python scripts/build_paper.py
```

## Contents and provenance

| Location | Contents |
|---|---|
| `paper/speechcomm/` | Article, supplement, generated tables, figures, bibliography, and claim audit |
| `reproducibility/round1/` | Bundled round-one analysis sources and score-derived results previously held outside the code repository |
| `reproducibility/metadata/` | Supporting metadata for reproducing the analyses; provenance and original-provider terms accompany the files |
| `results/phase_1_reproduction/` | Released-detector reproduction scores and frozen development thresholds |
| `results/phase_2_conversion_app/` | Round-one target scores and related artifacts; conversion results also support a separate deployment study |
| `results/phase_3_stress/` | Round-one channel results and manifests |
| `results/round2/` | Round-two scores, frozen thresholds, registered and exploratory results, logs, and author-trained checkpoints |
| `PREREGISTRATION.md`, `PREREGISTRATION_ROUND2.md`, `DEVIATIONS.md` | Registration documents and the subsequent deviation record |
| `analysis/`, `scripts/`, `deploybench/`, `models/`, `train/` | Analysis, reproduction, scoring, model adapters, and training code |
| `configs/`, `tests/` | Experiment configurations, protocols, and verification tests |
| `convert/`, `android/` | Conversion tooling and Android harness for the separate deployment study |

The historical registrations remain at **roster-freeze** (`97e9753`) and
**round2-freeze** (`73f2721`). Their original Git history is preserved. Publishing
this package does not supply an independent pre-experiment timestamp for those
author-held registrations.

The article reports desktop score analyses. The repository also contains
deployment tooling and conversion artifacts for a separate study; these are not
evidence of completed measurements on a phone.

## Licenses and access

Author-written code is licensed under [MIT](LICENSE). Original documentation
and author-trained checkpoints use CC BY 4.0. Third-party components retain their
own licenses and attribution notices. Scores and metadata derived from source
datasets follow the corpus-specific terms in [LICENSING.md](LICENSING.md) and
[DATA_ACCESS.md](DATA_ACCESS.md), including the In-the-Wild share-alike and MLAAD
non-commercial conditions. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/` for the retained notices and license texts.

Source audio and upstream checkpoints are obtained from their original
providers. Full inference or training reproduction therefore requires the
datasets, model backbones, vendor code, and the providers' permissions described
in `DATA_ACCESS.md`; the analysis reproduction above uses the included scores
and supporting metadata. MLAAD audio requires a Hugging Face account and
acceptance of the provider's access conditions.

## Score contract

Score tables use the fields
`utt_id,label,score,model,dataset,precision,runtime,condition,seed`.
Label `1` denotes bona fide speech; larger scores indicate more bona-fide-like
speech; a clip is classified as spoof when `score < threshold`. Every detector
uses its frozen source-development thresholds. Model-specific conventions and
input policies are guarded by the wrapper contract tests.

The round-one roster is AASIST, AASIST-L, RawNet2, Res-TSSDNet, and XLS-R+AASIST
(the unranked server reference). Round two adds ASVspoof 5 AASIST, Pascu LR,
XLSR-Mamba, and the registered AASIST-L retraining runs. Additional model adapters
are retained for the harness but are not part of the article's studied roster.

[MODEL_CARDS.md](MODEL_CARDS.md) records the exact input and score policies,
upstream provenance, and intended use of the eight studied detectors.
