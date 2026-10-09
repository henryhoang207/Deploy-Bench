# Speech Communication writing draft

Paper 1, D59. Worktree: `C:\dev\deploybench-paper`, branch `paper-speechcomm`.
The T-BIOM sources remain preserved in `paper/tbiom/`. This port uses `elsarticle`
in review mode and author-year references (`elsarticle-harv`).

The draft contains the introduction, expanded related work, protocol card and
measures, both study designs, audited round-1 findings, limitations, and draft
highlights and declarations. The main text includes the registered extension's
methods; the supplement retains the detailed round-1 analyses and long tables.
Author names, affiliations and email remain placeholders at the user's request.

Round-2 findings, the final abstract, cross-round discussion and conclusion remain
explicitly pending until the experiment session delivers reviewed W4 outputs.
The preliminary channel-results paragraph inherited from the T-BIOM draft has
been removed from this result-independent writing version.

## Rebuild and audit

From the worktree root:

```powershell
python paper/speechcomm/build.py
python paper/speechcomm/audit_claims.py
```

`build.py` builds the manuscript and supplement and checks reference resolution
and highlight lengths. The figure PDFs under `figures/` are the existing derived
round-1 figures, copied from the audited revision; these assets are tracked so
the LaTeX build is self-contained. Build outputs `main.pdf` and `supplement.pdf`
remain ignored. `generate_mlaad_table.py` regenerates only the round-1 fp32 table
from the existing committed MLAAD report, without scoring audio.

The claim checker rebuilds the original round-1 claims from the revision's result
files and checks input files, source annotations, registered method values and
the descriptive MLAAD table. It needs the read-only `DFV_Conference/revision`
analysis/results archive in the project docs tree; see its command-line help
for source-location configuration. It does not read round-2 score/result files.
The pending manuscript passes the draft audit; `--final` intentionally refuses
pending text and unintegrated round-2 numerical auditing. It is not a submission
approval while findings are still missing.

Source annotations precede numerical paragraphs, captions and input tables.
For unchanged generated round-1 tables, the audit compares the files with their
audited originals, allowing checkout line-ending differences only. Bibliography
comments identify primary-source checks; remaining author declarations require
author verification.

## Recorded corrections

The existing MLAAD sample-rate census gives 207,053 of 210,853 bona fide clips at
16 kHz, or 98.2% to one decimal. The registration and earlier prose say 98.1%.
The descriptive manuscript and caption use the census-derived value; the frozen
registration has not been changed. This correction changes no score or analysis.

The port also corrects stale round-1 supplement wording about the registration
chronology, the timing of the H5 manifest, and what test-time silence removal
can establish. The original numerical findings are retained.

## Author-dependent material

- Author names, affiliations, email and CRediT roles.
- Funding and competing-interest declarations.
- A stable public code/results archive with verified content and access terms.
- A complete AI-tool inventory, oversight statement and final responsibility declaration.
- Final highlights and cover letter after round-2 integration.

Journal/publisher policy anchors: the [Speech Communication guide for authors](https://www.sciencedirect.com/journal/speech-communication/publish/guide-for-authors)
recorded in step 4.7, and [Elsevier's current AI policy](https://www.elsevier.com/en-gb/about/policies-and-standards/generative-ai-policies-for-journals)
checked during this writing session. No journal submission or external publication
has been performed.
