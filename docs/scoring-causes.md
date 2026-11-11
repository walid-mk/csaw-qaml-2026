# QNN result causes

Highest priority first; the first applicable cause is `main_cause`, followed by
all other applicable causes in this same order:

1. `submission_invalid_record`
2. `submission_missing_record`
3. `evaluator_timeout`
4. `evaluator_failure`
5. `evaluator_invalid_record`
6. `evaluator_missing_record`
7. `baseline_invalid_record`
8. `baseline_missing_record`
9. `undefined_robust_normalization`
10. `undefined_clean_normalization`
11. `undefined_or_nonfinite_calculation`
12. `no_measured_robust_improvement`
13. `zero_measured_clean_accuracy`

A proven submission fault gives zero without retry, even when another independently
established fault is present. Derived outputs intentionally not computed after
that rejection do not create spurious evaluator-missing causes.

An organizer-side computed metric or baseline that is missing/corrupt triggers
the bounded retry policy. Legitimate undefined normalization or calculation gives
zero with its mathematical cause. Earlier resolved evaluator faults remain in
attempt history; they do not become decisive causes of a later successful score.
