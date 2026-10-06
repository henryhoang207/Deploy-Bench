# Dataset metadata used by the revision analyses

Four metadata files from the public dataset releases. The revision scripts read them from this folder, or from the folder in the
environment variable `DFV_META`. They are **not redistributed here**: obtain them from the original releases and check them against
the SHA-256 values below. They were copied read-only from the server's copies of the releases on 2026-10-04 and only used for labels,
speaker identities and counts, never for a score or a threshold.

| File | Bytes | SHA-256 | Source | Used for |
|---|---:|---|---|---|
| `meta.csv` | 1,021,512 | `2e404150eff3cf171b766da81013a93248cbb219682fff420bbadb6b86a18795` | In-the-Wild release (Müller et al., Interspeech 2022), `release_in_the_wild/meta.csv` | ITW speaker labels (54 distinct labels) |
| `ASVspoof2019.LA.cm.dev.trl.txt` | 822,400 | `2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7` | ASVspoof 2019 LA (Edinburgh DataShare), `LA/ASVspoof2019_LA_cm_protocols/` | LA19 dev speakers |
| `ASVspoof2019.LA.cm.eval.trl.txt` | 2,358,176 | `84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036` | ASVspoof 2019 LA (Edinburgh DataShare), `LA/ASVspoof2019_LA_cm_protocols/` | LA19 eval speakers (67), attacks |
| `trial_metadata.txt` | 62,086,042 | `70e0e7d5964562cb0166e79143f73beaeaf77ee18e799fa4156355a57006d800` | ASVspoof 2021 DF keys (`DF-keys-full`, `keys/DF/CM/`) | DF21 speakers (93), conditions, sources, subsets |
