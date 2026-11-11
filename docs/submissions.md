# Submission workflow

One submission identifies one family and scenario. Submit additional variants as
separate directories and separate private issues. Submissions open on **11 November 2026**. The global cutoff is
**Saturday, 14 November, 08:00 GST**, equivalent to **Friday, 13 November, 23:00 ET**.
Submit a new issue for each version before that time. Evaluation is queued;
opening an issue does not produce an immediate score.

## Create and register your private repository

1. Open the [public challenge starter](https://github.com/walid-mk/csaw-qaml-2026).
2. Select **Use this template → Create a new repository**, name it
   `csaw-qaml-2026-group-NN` and select **Private**. Do not make a public fork.
3. In **Settings → Collaborators → Add people**, invite your teammates and
   organizer **`walid-mk`**. Privately send organizers your group number, region,
   GitHub usernames and repository URL. Confirm organizer access is accepted.
4. Copy `templates/TEAM.json` to `TEAM.json` and enter your assigned details.
   Your registered group, not self-edited metadata, determines ownership.
5. If Issues are disabled in your new repository, enable them under **Settings →
   General → Features**. Keep your repository private throughout the competition.

No organizer-created repository or pull request to the starter is needed.
A private repository protects solution files, submission issues and feedback.
See [GitHub's template instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).

## Prepare your directory

```text
submissions/submission-001/
  submission.json
  writeup.md
  attack/                    # Attack scenario only
    run-0.npz
    run-1.npz
    run-2.npz
    run-3.npz
    run-4.npz
    source/                  # Code and reproduction instructions
  defense/                   # Defense scenario only
    recipe.json
    protocol.py              # Custom protocols only
```

Copy [submission.json](../templates/submission.json) and
[writeup.md](../templates/writeup.md), then replace the example metadata.
Your repository's `TEAM.json` contains your assigned team ID and region.
Paths in `submission.json` are relative to the repository root. For an attack,
set `scenario` to `attack`, `access` to `whitebox` or `blackbox`, `objective` to
`targeted` or `untargeted`, `recipe` to `null`, and `attack_files` to the five
NPZ paths in run order. For a defense, use `scenario: defense`, `access: whitebox`,
`objective: null`, `attack_files: []`, and the recipe path. The defense's evaluation
includes adaptive direct attacks and classical/quantum transfers.

## Attack artifacts

Each NPZ must contain exactly two arrays: Unicode-string `ids` and `float64`
`images`, in supplied ID order. No pickle/object arrays. Each compressed file and
its total expanded arrays must fit within 40,000,000 bytes. Missing or invalid
records count as failures in the original denominator.

After receiving inputs, this **format-only** example writes unchanged images:

```bash
python examples/make_attack_submission.py kit/inputs/qnn-whitebox.npz submissions/submission-001/attack/run-0.npz
```

Produce all five runs for a real submission. The example is not an attack solution.
Include the attack source and commands in your write-up; do not include weights,
credentials, or another team's data. Do not use Git LFS pointers in place of the
actual NPZ files accepted by this submission workflow.

## Defense recipe or custom protocol

Adversarial training is optional. Submit a compact recipe, or implement your own
training and/or preprocessing using the [custom defense protocol](defense-protocols.md).
The organizer reproduces five runs and evaluates the complete defended model
with adaptive attacks. The notebook demonstrates both interfaces. Include source
and reproduction instructions for a custom method; do not upload opaque weights.

## Submit an immutable version

1. Commit and push your completed directory to **your private team repository**.
2. Obtain the full commit SHA with `git rev-parse HEAD`.
3. Open a new issue using **Submit QAML solution** in that same repository.
   Include the SHA and repository-relative `submission.json` path.
4. Organizers copy the issue metadata and exact commit into private intake storage.
   GitHub's server timestamp and the organizer's retained intake record establish
   receipt, not the commit author date. An edited issue does not replace an accepted
   snapshot; open a new submission issue for a revision.
5. Receive validation status and evaluation feedback privately. Follow your regional leaderboard for published standings.

Do not submit through a pull request to the shared hub or starter repository.
Use the announced regional submission window. No pull request to the canonical
challenge repository is needed. Issues in a shared repository are visible to its
other collaborators; your group repository keeps submissions private.
