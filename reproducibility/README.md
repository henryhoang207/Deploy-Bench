# Manuscript reproduction inputs

`python scripts/reproduce_paper.py --verify` recomputes the round-1 estimator
checks from the released utterance scores, regenerates all manuscript tables and
renders the six figures, in a temporary directory or in the folder given with
`--out`. It requires only the analysis dependencies in
`requirements-reproduce.txt`, without speech audio, pretrained weights, GPU execution,
the original OneDrive folder, or an external server. PDF bytes are not compared:
timestamps and plotting-library versions can change their encoding.

The manuscript sources are not part of this package. When the authors' copy is
present, `--verify` also compares the regenerated table text with it, and
`python scripts/reproduce_paper.py --write` replaces its generated tables and
figures.

The `round1/` directory contains the author-maintained DEV-8 revision analyses,
numeric outputs, silence-trimmed scores and silence statistics used by this
manuscript. `round1/paper/` preserves the earlier text and generated tables used
as provenance by the authors' claim audit. It is a historical input, not the
submission manuscript. The source snapshot was copied on 13 October 2026 from the local
`DFV_Conference/revision` tree. It does not create an earlier public timestamp.

`source_manifest.json` records each original relative path, original SHA-256 and
packaged SHA-256. Gzip files decompress to exactly the original bytes. Changes
to copied analysis code are limited to portable repository discovery,
gzip-aware input reading, selectable output directories, and a read-only check
of the DF21 source summary. No score or numeric output was changed. The two
documented round-1 table-header edits distinguish the initial clip analysis
from the exploratory speaker analysis; the reproduction entrypoint applies
them to generated text.

The bundled `metadata/` files retain the respective provider licenses; consult
the repository's third-party notices. They contain protocols and public dataset
metadata, not audio. Exact ASVspoof 5 development/evaluation protocols are
bundled here as gzip files. The existing frozen MLAAD-5 normalized protocol in
`configs/mlaad5/protocol.csv.gz` supplies its language and speaker/system
metadata. The round-2 analysis library selects these files automatically; the
ASVspoof 5 condition analysis also uses the bundled protocol. Original provider
MD5s, original SHA-256s and acquisition provenance are recorded in the manifest.

Re-running the round-2 registered and exploratory statistics from released
scores is possible with `python analysis/round2/r2_registered.py` and
`python analysis/round2/r2_explore.py --b 1000`. This is a substantially longer
calculation than table/figure regeneration; it uses the fixed resampling
settings in the registrations. It writes analysis result files, so run it in a
separate checkout when comparing with the released numeric outputs. Do not
interpret a reduced-bootstrap smoke run as the reported result. Original
metadata paths can still be selected through `R2_META` or `DATA_ROOT`.

The registrations and deviation log are in the repository root. Their historical
version-control states are preserved at `roster-freeze` and `round2-freeze`.
Exact registration and deviation-log snapshots are also bundled under
`registrations/`, with the original commit IDs, blob IDs and file hashes;
`--verify` checks these bytes in a release archive or a clone.
