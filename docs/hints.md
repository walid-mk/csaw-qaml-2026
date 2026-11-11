# Small implementation hints

Use these optional hints after reading the [Student Guide](../STUDENT_GUIDE.md).
They address common implementation difficulties without prescribing a solution
or changing the rules.

<details>
<summary>Start with a small, correct example</summary>

Load one provided model run and one input before scaling up. Use `kit.sdk` to
load approved assets, and inspect shape and dtype at each boundary. Quantum
simulation can be slow, so verify correctness before increasing batch sizes.
Keep the released model and preprocessing unchanged.

</details>

<details>
<summary>Keep a continuous gradient path</summary>

For local white-box calculations, keep differentiable values as PyTorch tensors.
Converting to NumPy, calling `.detach()`, or using `torch.no_grad()` inside the
calculation removes information needed for gradients. Check that the quantity
you differentiate is a scalar and that the intended input has `requires_grad=True`.
Inspect gradient shape and finite values before relying on it.

</details>

<details>
<summary>Measure changes against the original image</summary>

The relevant maximum change is `max(abs(candidate - original))`, across all pixels
and channels. Comparing only with your previous candidate can hide accumulated
changes. Check the original-relative 8/255 bound, finite values, `[0,1]` range,
shape and IDs again after saving and reloading your numeric submission.

</details>

<details>
<summary>Treat black-box queries as a limited resource</summary>

Every image in a batch counts, and a timed-out request may already have been
charged. The supplied client therefore does not automatically retry. Keep a local
record of requests and returned usage, and follow the issued variant/run binding.
See [black-box access](black-box-access.md) before making requests.

</details>

<details>
<summary>Make a defense recipe easy to reproduce</summary>

Use either the compact recipe or a [custom defense protocol](defense-protocols.md). Organizers reproduce your method from
the approved original checkpoints with fresh optimizer state; uploaded weights
do not replace that process. Change only permitted fields, explain your reasoning,
and do not use evaluation inputs as training data. The examples demonstrate the
format, not a guaranteed high-scoring recipe.

</details>

<details>
<summary>Read the score and measurements together</summary>

QNN educational points use measured robust improvement and clean retention.
Check both robust and clean accuracy. QCNN has its own published eligibility
criteria and a separate leaderboard. An organizer-side
`not_scored` outcome is different from a valid zero. See [scoring](scoring.md)
and [recorded causes](scoring-causes.md).

</details>

<details>
<summary>Check the artifact you actually submit</summary>

Check all five run files, repository-relative paths and the full commit SHA.
Reload NPZ files with `allow_pickle=False`; attack archives contain only `ids`
and `images`. Include reproduction commands and a concise write-up. A successful
local run does not prove that the files captured from your submitted commit are
complete.

</details>
