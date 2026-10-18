# CLAUDE.md

JAX large-scale 3D linear modal analysis (Kv = λMv). The plan lives in `docs/ROADMAP.md`. Work proceeds PR by PR, in dependency order.

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

- No remote yet; everything is local.
- Each roadmap PR gets its own branch, named `pr-NN-<slug>` (e.g. `pr-00-scaffold`, `pr-01-mesh`), created from `main`. A PR split into parts uses `pr-NNa-<slug>`, `pr-NNb-<slug>`, ...
- "Pushing a PR" means: the checklist below passes, then merge the branch into `main` with `git merge --no-ff`.
- Stack a branch on another PR branch only while that earlier PR is still unfinished. When the base merges, run `git rebase --onto main <base> <branch>` (add `--update-refs` when several branches are stacked).
- Work in dependency order (see the table in `docs/ROADMAP.md`).

## Before pushing a PR

1. Run the full test suite with `scripts/test.sh` (`pytest -m "not gpu"` in parallel stages, slow tests included); it must pass. Mark new tests that assert wall time `serial` and ones that assert a process memory peak `peak`. GPU-only acceptance criteria go into the "Deferred GPU checks" list in `docs/ROADMAP.md`.
2. Run review rounds, each with 5 fresh subagents reviewing the code (correctness, simplicity, tests and reference comparisons, docs, performance); wait for all 5. Address the feedback after each round. Stop when one round has all 5 satisfied, or when two rounds in a row find only small bugs or minor issues; otherwise run another round.
3. Make sure the docs (`README.md`, `docs/`, `docs/ROADMAP.md`) reflect the change.
4. Make sure memory is up to date.

## Planning

- Before implementing any PR, write a plan for it (e.g. `docs/plans/pr-NN-<slug>.md`, committed on the PR branch) and review it as below; implement only once all reviewers are satisfied.
- When making a plan, run review rounds of 5 fresh subagents each to check it. Revise after each round. Stop when one round has all 5 satisfied, or when two rounds in a row ask only for minor changes; otherwise run another round.
- Stop the review loop and ask the user when the remaining issues need the user's input (decisions or information only they have) instead of iterating further.
