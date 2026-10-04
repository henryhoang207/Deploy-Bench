# MLAAD-5 frozen protocol (D47 / DEVIATIONS.md DEV-6), committed 2026-10-04 BEFORE any scoring

Built on sol1 by `scripts/build_mlaad_protocol.py` (HEAD `e73c675`) with
`--mlaad-revision 30c3dec763fa4803f11c2df1557d3b0828395366`, seed 20261003.

| file | what |
|---|---|
| `protocol.csv.gz` | utt_id,path,label,lang,system: 494,853 trials; gunzipped sha256 = `protocol_sha256` in the summary (`14bd7b48...`) |
| `protocol_summary.json` | counts per language, bona fide pools, imbalance flags, M-AILABS archive sha256s |
| `snapshot_5lang.json` | MLAAD snapshot revision + spoof counts |
| `mlaad_SHA256SUMS_5lang.txt.gz` | sha256 of every one of the 284,000 MLAAD wavs |
| `mailabs_SHA256SUMS` | sha256 of the 6 M-AILABS archives (mirror ics.tau-ceti.space) |

| lang | spoof (systems) | bona fide drawn / pool |
|---|---|---|
| en | 143,000 (143) | 69,853 / 69,853, **imbalanced** (all of en_US + en_UK) |
| de | 44,000 (44) | 44,000 / 118,528 |
| it | 34,000 (34) | 34,000 / 73,405 |
| fr | 42,000 (42) | 42,000 / 90,348 |
| pl | 21,000 (21) | 21,000 / 26,322 |

The loader (`deploybench/data.py`, dataset `mlaad`) reads `$DATA_ROOT/mlaad5/protocol.csv`, which is this file
gunzipped. `scripts/sol1/mlaad_scoring.sh` refuses to score unless that file's sha256 equals the frozen one.
**Confound, stated with every MLAAD number:** the bona fide audio is native 16 kHz, while the spoof audio comes
from TTS systems at other rates and is resampled by the harness.
