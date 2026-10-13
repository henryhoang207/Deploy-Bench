# Licensing map

The package contains original author contributions and third-party information.
The root [LICENSE](LICENSE) grants **MIT** rights in author-written software. It
does not relicense source corpora, copied metadata, upstream implementations,
Gradle wrapper scripts, or pretrained checkpoints. More specific entries below
take precedence over a directory's default. License texts are in `licenses/`;
provider attribution and notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Material / paths | Release terms |
|---|---|
| Author-written Python, shell, PowerShell, Kotlin, configuration and analysis/reproduction code | **MIT**, root `LICENSE`. Third-party portions retain their notices; in particular, `train/aasist_r2_train.py` adapts clovaai's MIT training driver. |
| Original author documentation, registration documents/snapshots, deviation log, manuscript prose, aggregate analyses, tables and figures, frozen thresholds and numeric summaries | **CC-BY-4.0** for author-held rights, `licenses/CC-BY-4.0.txt`. Embedded copied metadata and externally authored material retain their own terms. |
| Authors' AASIST-L learned weights in `results/round2/runs/*/{swa,last}.pth` and fitted Pascu heads in `results/round2/pascu/*.npz` | **CC-BY-4.0** for author-held rights. The architecture/training software remains under its upstream MIT terms; original training corpus attribution remains required. No downloaded Meta backbone is included in the Pascu heads. |
| Author-generated per-utterance score values on ASVspoof 2019/2021/5, including stress variants of LA19 | **CC-BY-4.0** for author-held rights, with the ASVspoof ODC attribution notices below. Copied trial IDs, labels, speaker identifiers, attack/condition metadata and protocol fields retain **ODC-By 1.0**. |
| ASVspoof-derived manifests, protocol fields and cluster mappings: relevant files in `configs/`, `reproducibility/metadata/`, `reproducibility/round1/meta/`, and `results/round2/stress/` | Copied/derived database information under **ODC-By 1.0** (`licenses/ODC-By-1.0.txt`); author-created additions under CC-BY-4.0. Preserve database notices when redistributing. |
| ITW-linked scores, trial/cluster metadata and manifests: files whose name or recorded dataset is `itw` / In-the-Wild, including round-1 `scores_*_itw*`, round-2 `itw.csv*`, trim-score files, and ITW metadata under `reproducibility/` | **CC-BY-SA-4.0** for the released ITW-linked dataset files. Attribute Müller et al. and this study, indicate changes, and retain share-alike terms when sharing adapted files. The provider's conflicting Apache-2.0 statement is preserved in DATA_ACCESS.md and THIRD_PARTY_NOTICES.md; no source audio is redistributed. |
| MLAAD/M-AILABS mixed per-trial files: `configs/mlaad5/protocol.csv.gz`, `mlaad_SHA256SUMS_5lang.txt.gz`, relevant normalized metadata under `reproducibility/metadata/`, `results/phase_2_conversion_app/mlaad5/**/scores_*`, and `results/round2/scores/*/s*/mlaad5.csv.gz` | The authors elect to distribute these linked evaluation datasets under **CC-BY-NC-4.0** for their contributions, preserving MLAAD's non-commercial terms for its identifiers/metadata. M-AILABS fields retain the original custom permissive notice. Public download is permitted for this non-commercial reproducibility release; source MLAAD audio must still be obtained from its provider's gated service. |
| Any other file containing per-trial scores or metadata from multiple corpora | Each corpus's rows/fields retain the corresponding terms above. Joining datasets does not remove ODC attribution, ITW share-alike, or MLAAD non-commercial conditions. |
| M-AILABS-only copied data descriptions, archive IDs, paths and checksums | Original custom permissive license in `licenses/M-AILABS.txt`. Mixed MLAAD/M-AILABS protocol files additionally follow the mixed-file rule above. |
| `android/gradlew`, `android/gradlew.bat` and Gradle wrapper components | **Apache-2.0**; retain their original Gradle copyright and notices. Full text in `licenses/Apache-2.0.txt`. |
| License files, bibliography records, third-party copyright notices, and any explicitly licensed third-party file | Original terms prevail. Including the full license text does not make the associated upstream implementation part of this release. |

The ITW and MLAAD score-file licenses are deliberate choices for this publication
package to keep linked scores and metadata under clear, corpus-compatible terms;
they are not a claim that every independent numeric result automatically inherits
an audio corpus's copyright license. CC-BY-4.0 aggregate summaries do not grant
rights to commercially exploit MLAAD source material or remove upstream database
conditions. Code may be used under MIT while particular input/score datasets
remain subject to their own conditions.

## Required source attribution

Contains information from [ASVspoof 2019](https://zenodo.org/records/6906306),
[ASVspoof 2021](https://www.asvspoof.org/index2021.html), and
[ASVspoof 5](https://zenodo.org/records/14498691), which are made available under
the [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/).

Contains information from In-the-Wild by Nicolas M. Müller, Pavel Czempin,
Franziska Dieckmann, Adam Froghyar and Konstantin Böttinger, *Does Audio Deepfake
Detection Generalize?* (Interspeech 2022),
[original provider](https://deepfake-total.com/in_the_wild),
[creator's CC-BY-SA-4.0 dataset card](https://huggingface.co/datasets/mueller91/In-The-Wild).
Score columns and normalized cluster mappings were added by the authors of this
study; source audio was not copied into the package.

Contains information from MLAAD by Nicolas M. Müller, Piotr Kawa, Wei Herng
Choong, Edresson Casanova, Eren Gölge, Thorsten Müller, Piotr Syga, Philip Sperl
and Konstantin Böttinger, *MLAAD: The Multi-Language Audio Anti-Spoofing Dataset*
(IJCNN 2024), [provider](https://deepfake-total.com/mlaad), available under
[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Five-language selection, normalization, seeded M-AILABS matching, hashes and score
columns were added by this study; source audio was not copied into the package.

Contains information from the M-AILABS Speech Dataset by I. Celeste Aurora
(Solak) and Dima Naumov, [provider](https://github.com/i-celeste-aurora/m-ailabs-dataset).
Its original copyright, conditions and disclaimer are retained in
[licenses/M-AILABS.txt](licenses/M-AILABS.txt).

When referring to the scientific results, cite this study and the applicable
original corpus/model publication. Legal attribution requirements apply
independently of the scientific citation request.
