# CLAUDE.md

JAX large-scale 3D linear modal analysis (Kv = λMv). The plan lives in `docs/ROADMAP.md`. Work proceeds one PR at a time, in roadmap order.

## Code

- Write short, simple, non-verbose code. Choose the easiest solution that works; avoid abstraction until it is needed twice.
- Validate every new feature against a reference solution: SciPy (assembled matrices, `eigsh`, `spsolve`), dense NumPy, or analytic results.
- Every feature comes with tests. After each commit, run the tests relevant to that change.

## Docs

- Keep `README.md` and the reference docs in `docs/` current (elements, materials, operators, preconditioners, eigensolvers, ...). Write the math in LaTeX (`$...$`, `$$...$$`).
- Commit docs in the same PR as the code they describe.

## Commits

Format: `<type>: <title>`, then a blank line and a short summary. Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

## Branches (local only for now)

- No remote yet. `main` holds the docs baseline.
- Each roadmap PR gets its own branch, named `pr-NN-<slug>` (e.g. `pr-00-scaffold`, `pr-01-mesh`). Each branch is created from the previous PR's branch, so they stack (`main` ← `pr-00` ← `pr-01` ← ...).
- "Pushing a PR" means finishing its branch locally after the checklist below. If an earlier branch changes, rebase the later branches onto it.

## Before pushing a PR

1. Run the full test suite; it must pass.
2. Run several review rounds, each with 3–5 subagents reviewing the code (correctness, simplicity, tests and reference comparisons, docs, performance). Address the feedback, then start another round. Repeat until there are no substantive findings.
3. Make sure the docs (`README.md`, `docs/`, `docs/ROADMAP.md`) reflect the change.
4. Make sure memory is up to date.

## Planning

- When making a plan, run several review rounds of 5 subagents each to check it. Revise and repeat until all reviewers are satisfied.
- Stop the review loop and ask the user when the remaining issues need the user's input (decisions or information only they have) instead of iterating further.
