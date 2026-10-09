# Writing-session validation — 2026-10-09

Scope: `paper/speechcomm/` in `C:\dev\deploybench-paper`, `paper-speechcomm`.
No experiment was run on sol1, and no file in `scripts/sol1/`, `results/round2/`
or `DEVIATIONS.md` was modified. The T-BIOM manuscript remains preserved.

- Draft claim audit: **211 checks, zero failures**, with 16 pending items.
  The original round-1 reproduction gate also passes before the manuscript audit.
- Final-mode guard: deliberately refuses the incomplete draft, including its
  placeholders and unintegrated round-2 results; compiled-PDF and freshness
  checks pass. It cannot report this writing draft as submission-ready.
- In-memory negative checks: the auditor refuses an absent source, an invented
  numerical claim even with an existing source path, and an unsupported method
  value. These checks changed no manuscript or result file.
- Repository tests, run in the paper worktree: **169 passed, 5 skipped**.
- Local LaTeX build: main **30 review pages**, supplement **13 pages**;
  no overfull boxes, unresolved references/citations or BibTeX warnings.
- Highlights: five entries, 72/74/75/69/72 characters.
- Derived figure PDFs: byte-identical to the archived round-1 revision figures.
- Literature and bibliography reviewed against primary sources; registered
  methods reviewed independently against the addendum and logged amendments.

The mechanical source check verifies annotations, source existence and numerical
consistency; it does not substitute for reading a cited paper or interpreting a
statistical result. Primary-source checks are recorded in `refs.bib`.

Remaining work depends on the reviewed W4 outputs or the authors: round-2
findings and numerical-audit integration, final abstract/discussion/conclusion,
author block, CRediT/funding/interests, public archive, final AI-use statement,
final highlights and cover letter, and submission review. Author-identifying
fields remain placeholders as requested.
