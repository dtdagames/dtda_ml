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
  applies to it, the test ceiling does not. The same holds for
  `tools/check_proportions.gd`: it is a ruler, not a suite, so the test ceiling
  passes it by, but it is listed and measured like every other file under the
  comment ceiling. A ruler that exempts itself from what it measures weakens the
  rule it carries.

### One named exception

`tests/test_qlearning.gd` stands at 76 non-empty lines against a budget of 59,
and stays there. The ruler grants it by name, in the `EXCEPTIONS` table.

The reason is arithmetic, not taste. Q-Learning is the most stateful model in
the repository, and this suite's budget is computed on one single 237-line
module: 59 lines to hold the epsilon decay, the `to_dict` snapshot, the refusal
loop and the `reset`, four blocks each the only witness of the guards beneath
it. `test_models.gd` covers four modules and draws 123 lines for the same
quarter. The rule penalises a one-module suite mechanically, and getting under
59 here means deleting a whole block, that is, a guarantee.

The doctrine, because it is what serves next time:

- an exception is **asked for when the unique witnesses exceed the budget**, not
  when the file is merely awkward to shrink. Which witnesses are unique is
  settled by mutation, one guard at a time, never by reading.
- it is **justified by the measure**: the figures go in the request, and the
  number granted is the file's count as measured, not a round number above it.
- it **caps**. 76 means 76: at 77 lines the suite is over again and the script
  exits 1. Every other suite over its budget still exits 1 too. The ceiling is
  not negotiable against comfort; it is negotiable against a guard that would
  otherwise lose its only witness.

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
