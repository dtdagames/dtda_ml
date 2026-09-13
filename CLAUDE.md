# dtda_ml, house rules

## Proportions

Two ceilings, measured, not felt:

- **Comments: at most 10%** of the non-empty lines of every `.gd` file.
- **Tests: at most 25%** of the non-empty lines of the modules a suite covers.
  The ceiling is per test file, computed on the modules that file exercises.

Measure with:

```
godot --headless --script res://tools/check_proportions.gd
```

It prints a line per file and exits 1 as soon as a ceiling is crossed.
It is not part of the test suite and declares no `PLAN`.

How the figures are built, so nobody argues with them:

- "lines" means non-empty lines, on both sides of both ratios. Blank lines can
  neither inflate a budget nor dilute a percentage.
- a line counts as commented when it carries a `#` **outside a string**. A full
  line comment and a comment trailing code both count as one. `Color("#14161c")`
  and a maze row made of `#` are code; a counter blind to that credits the demos
  with comments they never had, and a repository that looks clean because the
  counter answers zero everywhere is the failure mode to fear.
- which modules each suite pays for is declared in the script, in `COVERAGE`.
  `tests/run_tests.gd` is the runner, not a test file: the comment ceiling
  applies to it, the test ceiling does not. `tools/check_proportions.gd` is the
  ruler and excludes itself from both.

## What the rule costs

It is a **volume ceiling, not a quality target**. Nothing gets better by
deleting lines. A test removed is a guarantee removed, and the percentage will
drop just the same whether what left was dead weight or the only thing holding a
regression out.

So the choice of what goes is made **by mutation, not by reading**: break the
code the test claims to protect, one guard at a time, and keep what fails. What
nothing notices when you delete it is what leaves first. Comments follow the
same test: a comment that repeats the line below it goes, a comment that records
why a guard exists stays, because that one is the only copy of a reason.

## Two writing rules already in force

- Every suite declares `const PLAN = N`, the number of assertions it runs, and
  the runner fails on any gap. Add a test, move the number.
- Write `check_equal("...", call(), false)`, never `check("...", not call())`.
  A GDScript runtime error answers `null`, so `not` turns a crash into a pass.

Also: `int / int` is an integer division in GDScript. A test written with float
literals proves nothing about that, so assert it with integer arguments.

## The suite

```
godot --headless --path . --script res://tests/run_tests.gd
```
