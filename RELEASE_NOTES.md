# Reproducibility release

This release provides the versioned reproducibility package for the manuscript
"Thresholded Once, Deployed Elsewhere: A Two-Round Audit of
Threshold Transfer in Speech Deepfake Detection."

It includes the utterance-level score files, frozen thresholds, evaluation
manifests, frozen analysis-plan snapshots, deviation log, analysis and
table/figure generation code, supporting metadata, and author-trained round-two
checkpoints. The round-one supporting analysis files are bundled so the paper
can be reproduced from a fresh checkout without the authors' OneDrive tree.

Run the commands in README.md to regenerate the tables and figures and run the
tests. Source audio and upstream
checkpoints are obtained separately from their original providers; these are
needed for full scoring/training, rather than analysis of the included scores.

MIT covers author-written code. Documentation and author-trained checkpoints
use CC BY 4.0, with corpus-specific conditions on derived scores and metadata.
DATA_ACCESS.md, LICENSING.md, THIRD_PARTY_NOTICES.md, and licenses/ identify the
applicable licenses, required notices, and dataset access conditions.

The release preserves roster-freeze (97e9753) and round2-freeze (73f2721).
Its publication date is not an independent pre-experiment timestamp.
