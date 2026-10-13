# Original registration snapshots

Each subdirectory contains the exact registration and deviation-log files that
existed at the corresponding original Git tag. `manifest.json` records the full
commit identity, author-controlled committer time, Git blob identities, original
byte counts and SHA-256 hashes. These snapshots were extracted from the local
Git history on 13 October 2026; they preserve that history without presenting
the new public archive as an independent timestamp from before the analyses.

`roster-freeze` is commit `97e975390f1de332c8c432027da9470c7a8ac4a0`.
`round2-freeze` is commit `73f2721ee0b075cae6bb9b2841eb804758fa4e0a`.
The repository-root registrations and deviation log also include subsequent
amendments, whose timing and visibility disclosures must be read alongside the
frozen snapshots. Both `scripts/reproduce_paper.py --verify` and the manuscript
claim audit check the snapshot file hashes. In a Git checkout, the claim audit
also requires the original tags to resolve to the recorded full commits.
