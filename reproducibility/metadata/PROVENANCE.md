# Packaged dataset metadata

The four round-1 gzip files preserve the exact original metadata used by the DEV-8
revision analyses. The source copies were obtained from the public releases on
4 October 2026 and packaged on 13 October 2026. The uncompressed byte counts and
SHA-256 values are recorded in `../source_manifest.json` and verified by
`scripts/reproduce_paper.py`. Packaging changed only the lossless encoding.

| Packaged file | Original provider release | Analysis use |
|---|---|---|
| `meta.csv.gz` | In-the-Wild (`release_in_the_wild/meta.csv`) | Public speaker labels |
| `ASVspoof2019.LA.cm.dev.trl.txt.gz` | ASVspoof 2019 LA protocols | Development speakers and labels |
| `ASVspoof2019.LA.cm.eval.trl.txt.gz` | ASVspoof 2019 LA protocols | Evaluation speakers, attacks and labels |
| `trial_metadata.txt.gz` | ASVspoof 2021 DF full keys (`keys/DF/CM/`) | Speakers, conditions, sources and evaluation subsets |
| `ASVspoof5.dev.track_1.tsv.gz` | ASVspoof 5 v0.1 track-1 development protocol | Source-development speakers and labels |
| `ASVspoof5.eval.track_1.tsv.gz` | ASVspoof 5 v0.1 track-1 evaluation protocol | Evaluation speakers, codec conditions and labels |

The ASVspoof 5 protocols were acquired on 13 October 2026 from coauthor Jee-weon
Jung's [pinned public mirror](https://huggingface.co/datasets/jungjee/asvspoof5/tree/ebf8037ec54eaa5a054781ee34246206cef9fc11)
of the [original Zenodo v0.1 release](https://zenodo.org/records/14498691).
The mirror archive contains the exact original development and evaluation
files: their MD5s match the organiser's release README
(`e949dee1091a49b04943c548b5cf9860` and
`50c0591d45a282cb787a5a4614394f67`, respectively). The manifest records the
actual pinned acquisition URL, original provider URL and independent SHA-256
hashes. Only these two track-1 protocols were repackaged; no speech was fetched.

MLAAD-5 metadata is read from the existing author-normalized frozen protocol,
`configs/mlaad5/protocol.csv.gz`, whose decompressed SHA-256 is
`14bd7b487600c0446248b5599a55da0b05d9c15823f8c8cf3c90e4d294a7a84f`.
Its mixed MLAAD/M-AILABS permissions are documented in the repository notices.

Provider licensing and attribution notices in the repository apply to these
files separately from the author's code. Audio is obtained from the original
providers and is not part of this package.
