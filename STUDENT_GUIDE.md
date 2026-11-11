# QAML Student Guide

Welcome! Start with the runnable [QAML Challenge Notebook](QAML_CHALLENGE.ipynb),
or follow this text guide. Materials are released on **11 November 2026**.
You can run the local examples as soon as you have your group repository.

Challenge day is **Friday, 13 November 2026**, full day in your region: GST for
MENA and ET for US–Canada. Check the
[event schedule](https://github.com/walid-mk/csaw-quantum-challenges-2026/blob/main/docs/schedule.md)
for regional times. Submissions open on **11 November** and close on
**Friday, 13 November: 6:00 PM GST for MENA; 6:00 PM ET for US–Canada (online)**.

Start solving on **11 November** when the materials are released. Optional
challenge-day hints will be provided on 13 November.

## 1. Get your team's private repository

Use **Use this template → Create a new repository → Private** on
[the public starter](https://github.com/walid-mk/csaw-qaml-2026). Do not fork it:
a public fork cannot become a private submission repository. Name your repository
`csaw-qaml-2026-group-NN`, replacing NN with your group number.

In **Settings → Collaborators → Add people**, invite your teammates and organizer
**`walid-mk`**. Privately send the organizers your group number, region and repository
URL. Verify that the organizer invitation is accepted before submitting. Copy
`templates/TEAM.json` to `TEAM.json` and enter your assigned ID and region.
Clone your team's repository (replace YOUR-USERNAME and NN):

```bash
git clone https://github.com/YOUR-USERNAME/csaw-qaml-2026-group-NN.git
cd csaw-qaml-2026-group-NN
```

Your code, solution issues and feedback belong in this private repository.
Never post them or tokens to the public starter or event hub.

## 2. Install and verify the kit

Follow [environment setup](docs/setup.md), including Jupyter installation.
Run `kit.sdk.verify_assets()`; the kit status should be `released`.
Keep the kit unchanged and put your work under `submissions/`.

The kit contains:

| Item | Purpose |
|---|---|
| `kit/models/qnn/` and `kit/models/qcnn/` | Original clean models for defense and hardened white-box models for attack; five runs each |
| `kit/inputs/` | Approved attack inputs for each family/access mode, with assigned target classes |
| `kit/training/road-signs-training.npz` | 889 approved training crops, IDs and training labels |
| `kit/preprocessing.npz` | Fixed preprocessing fitted using training data |
| `kit/sdk.py` | Model loading, input loading, integrity checks and NPZ writing |

Black-box weights and true evaluation labels are not included. Road Signs is a
public/exposed educational benchmark. Read the kit's attribution and usage notices.

## 3. Choose your scenario and model

You may work on **QNN** or **QCNN**. Both have eight qubits, RY angle encoding,
classical classification heads and noiseless simulation.

- **Attack:** modify supplied images so a hardened model misclassifies them.
  Choose white-box or black-box access, and targeted or untargeted success.
- **Defense:** submit a reproducible defense protocol to improve an original clean
  model. The organizer trains it and evaluates robustness and clean utility.

Keep different families/scenarios/attack objectives in separate submissions.
The class order is `speedlimit`, `crosswalk`, `trafficlight`, `stop`.

## 4A. Work on an attack

For white-box exploration, load the approved model and inputs:

```python
import torch
from kit.sdk import load_model, load_inputs

model = load_model("qnn", run=0, kind="hardened")
batch = load_inputs("qnn", "whitebox")
images = torch.tensor(batch["images"], dtype=torch.float64, requires_grad=True)
logits = model(images[:1])
print(logits.shape)  # one image, four class scores
```

Your method must preserve image IDs and order, shape `(N,3,32,32)`, finite
`float64` pixels in `[0,1]`, and **maximum per-channel change 8/255** from each
original. A targeted attack seeks the supplied incorrect target class; an
untargeted attack seeks an incorrect class. Do not change preprocessing or encoding.

For black-box work, use `client.prediction.predict` with your privately issued
URL/token as described in [black-box access](docs/black-box-access.md). There
are 273 queries per image for each issued variant/run grant. Repeats and retries
cost queries. The client does not automatically retry a request that may have
already been charged. Your group receives its URL and token privately; see the access guide.

Produce `run-0.npz` through `run-4.npz` for your chosen variant. Every archive
contains exactly Unicode `ids` and float64 `images`. This command illustrates
the file format by copying unchanged images; **it is not an attack solution**:

```bash
mkdir -p submissions/attack-001/attack
python examples/make_attack_submission.py kit/inputs/qnn-whitebox.npz submissions/attack-001/attack/run-0.npz
```

Use `kit.sdk.write_submission` to save the outputs of your own attack for all
five model runs. Include your source, dependencies, reproduction commands and
query usage in your private submission directory.

## 4B. Work on a defense

Adversarial training is optional. You may use custom training, regularization or
an input preprocessing defense while retaining the assigned quantum model.
See [defense protocols](docs/defense-protocols.md). The notebook demonstrates both
an adversarial-training step and a preprocessing layer. Organizers reproduce your
method for five original model runs and evaluate the complete defended pipeline.

## 5. Complete metadata and your write-up

Copy `templates/submission.json` and `templates/writeup.md` into your submission
directory. Replace the team ID, region (`mena` or `us-canada`), submission ID,
family and repository-relative file paths.

For a defense, use `scenario: defense`, `access: whitebox`, `objective: null`,
the recipe path, and `attack_files: []`. For an attack, use `scenario: attack`,
your actual access/objective, `recipe: null`, and the five NPZ paths in run order.
Set `challenge` to `qaml`. The metadata describes your artifact; it does not
replace your authenticated repository identity.

Explain your method, reproduction steps, results, limitations, mitigation ideas,
dependencies and material AI assistance. Distinguish local observations from
organizer-verified measurements. Never include a token or a claimed score as evidence.

## 6. Submit the exact version

Submit before the published cutoff:

```bash
git add submissions/defense-001
git commit -m "Submit QNN defense recipe and analysis"
git push
git rev-parse HEAD
```

Open a new **Submit QAML solution** issue in **your private team repository**.
Provide the full 40-character commit SHA, team/region and metadata path.
The organizer captures that exact commit and retains the intake timestamp.
A branch name or commit author date is not submission evidence. Submit a new
issue for a revision; editing an accepted issue does not replace its snapshot.

GitHub does not train your model or run your attack automatically. An organizer
worker validates and evaluates the captured artifact. Do not use Git LFS pointers
instead of the required numeric file bytes.

## 7. Understand your feedback

Attack feedback shows validated success, uncertainty, invalid/missing record counts
and variant points. Defense feedback shows robust balanced accuracy `R`, clean
balanced accuracy `C`, uncertainty, applicable eligibility checks and points.

QNN uses its approved educational score:

```text
100 × clip((R − 0.5958333333333333) / (1 − 0.5958333333333333))
    × clip(C / 0.9003793261336365)
```

`clip` limits each factor to `[0,1]`. There is no QNN clean floor eligibility or
tier threshold. QCNN uses its published clean-utility and robust-gain criteria. Read [scoring](docs/scoring.md) for exact rules.

QNN submission-caused invalid/missing records score zero with causes. Organizer
evaluator failures receive at most three identical-input attempts, then become
**not scored: evaluator failure**, with null points. A not-scored attempt does
not erase your earlier best. QNN and QCNN defense standings stay separate.

Ask questions privately through your team repository or the announced support
channel. The feedback for your submitted commit is the authoritative result.
