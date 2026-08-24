# Optimization program

This directory records performance experiments against the same raw-WGSL
DiCoSe graph. It keeps quality and timing evidence together so a fast-looking
shader does not silently replace a correct one, and a correct-looking shader
does not silently make the browser slower.

## Rules

1. Start from the current accepted fixture contract and record an `OPT-NNNN`
   entry before retaining a performance-sensitive change.
2. Validate the narrowest affected primitive in an isolated Chrome profile.
3. Run the supplied WAV in an isolated Chrome profile and require the f16
   deterministic acceptance envelope before interpreting timing.
4. Retain only improvements measured at the full inference boundary.
5. Keep rejected experiments in the ledger; do not leave dormant public
   switches merely to preserve them.

The browser harness is intentionally unattended: each run launches Chrome with
a disposable profile and uses CDP to collect the automatic report. Raw result
artifacts belong in the ignored `benchmark/results/` tree when needed; the
small, reproducible facts belong in this ledger.

The current production-shape operation count and hardware floor are recorded
in [`ARITHMETIC_BUDGET.md`](ARITHMETIC_BUDGET.md). Revisit that budget before
allocating work to a low-impact kernel family.
