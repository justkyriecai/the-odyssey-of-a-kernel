# Vendored

This skill is a copy of [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill),
taken at commit `74a12918e9f64d78036f14da5f8765e435b949a4`, and is maintained
here rather than tracked as a submodule.

Changed from upstream:

- the directory is named `odyssey-ncu-report`, and `SKILL.md`'s `name:` matches it;
- `reference/10-no-counter-fallback.md` is ours -- what to do when hardware
  counters are denied, which is the normal case on a rented pod. `SKILL.md` and
  `reference/09-common-issues.md` are edited to route to it.

Everything else is upstream's.

To take a newer upstream revision, diff against that repository, apply what is
wanted, and record the new commit here. Do not re-add it as a submodule -- the
prefix is the point.
