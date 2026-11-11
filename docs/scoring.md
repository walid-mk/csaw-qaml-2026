# Scoring

Organizers compute all metrics using the fixed evaluation pipeline. Participant
claims do not award points. `R` is robust balanced accuracy and `C` is clean
balanced accuracy, represented as fractions from 0 to 1. Feedback includes paired changes and 95% confidence intervals. Road Signs remains public/exposed.

## QNN defense: educational score v3r2

```text
clip(x) = min(1, max(0, x))
R0 = 0.5958333333333333
C0 = 0.9003793261336365
points = 100 × clip((R − R0) / (1 − R0)) × clip(C / C0)
```

`R0` and `C0` are frozen original-model measurements, common to every team. There
is no clean-accuracy eligibility floor, tier threshold or confidence-bound veto
for this score. The maximum is 100. Full clean retention is
capped at 1. Exact scoring precision is retained; rounding is only for display.

For example, `R = R0` and `C = C0` gives **0 points**, while `R = 1`
and `C = C0` gives **100 points**. Use the notebook to calculate further
examples. Local calculations help you understand the formula; your
graded result uses measurements from the full evaluation suite.

Submission-caused invalid or missing records score zero automatically. Organizer
evaluator failures/timeouts receive at most **three total identical-input attempts**.
If all fail, the result is **not scored: evaluator failure**, with null points.
Every zero/not-scored result has a fixed-priority main cause and all other applicable
causes. The full [cause policy](scoring-causes.md) is unchanged.

## QCNN defense

For QCNN, the following clean-utility criteria apply. Clean balanced accuracy must be at least
`0.7915030824420426`; the upper 95% bound on paired clean loss must be at most
`0.06671775562793336`. Let `L` be the lower 95% bound on paired robust gain:

```text
points = 100 × clip(L / 0.12332663690476187)   if clean utility passes
points = 0                                otherwise
```

The robust-gain criterion additionally requires `L > 0`. Existing tier fractions
are one-third, two-thirds, and one. Use the QCNN formula for QCNN submissions.

## Attack variants

For each family/access/objective, `L` is the lower 95% bound on measured attack
success on the fixed common originally-correct set. The score is
`100 × weight × clip(L / full_credit_anchor)`. Targeted weight is 1.25
(maximum 125); untargeted weight is 1 (maximum 100). The exact anchors appear
in [the frozen score settings](../config/scoring.json). Missing/invalid records
stay in the denominator as failures. Different access modes use separate targets.

## Ranking and presentation

Keep QNN and QCNN defense leaderboards separate, and identify every attack variant.
Only the best valid score per team/variant/benchmark counts; retain the earliest
equal result. A not-scored attempt does not erase a prior best. Regional boards
remain separate. No new combined defense score is introduced.

The broader judging rubric remains 45% validated technical points, 25% analysis
and mitigation, 15% creativity, and 15% reproducibility/presentation. Exact
event-wide aggregation across this and the other authors' challenges is pending
publication in the event schedule. Compare results within the same variant.
