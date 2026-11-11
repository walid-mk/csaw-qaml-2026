# CSAW 2026 · QAML

**Quantum Adversarial Example Challenge — QNN and QCNN**

Welcome! Explore how small image changes affect quantum classifiers and develop
a defense that improves robustness while retaining clean-image performance.
You will use the provided Road Signs images and eight-qubit, noiseless models.

**Materials and submissions open: 11 November 2026.** Challenge day: Friday, **13 November**,
in GST for MENA and ET for US–Canada. See the [event schedule](https://github.com/walid-mk/csaw-quantum-challenges-2026/blob/main/docs/schedule.md)
for your regional times. **Submission cutoff: Friday, 13 November, 6:00 PM GST for MENA and
6:00 PM ET for US–Canada (online).** You can start the local notebook immediately;
black-box access begins when your group receives its service URL and token.

## Start here

1. Create your own **private** repository using **Use this template**, invite teammates
   and organizer **`walid-mk`**, then clone it. [Step-by-step instructions](docs/submissions.md).
2. Follow [setup](docs/setup.md), then open **[QAML Challenge Notebook](QAML_CHALLENGE.ipynb)**.
3. Read [rules](docs/rules.md), [scoring](docs/scoring.md), and [submission instructions](docs/submissions.md).
4. Submit a new issue in **your group's repository**, with the exact commit SHA.

Prefer a text walkthrough? Read the [Student Guide](STUDENT_GUIDE.md).
Start solving from the **11 November release**; do not wait for 13 November.
Optional hints will be provided on challenge day.

## Your scenarios

| Scenario | QNN | QCNN |
|---|---|---|
| Attack the hardened model | White-box / black-box, targeted / untargeted | White-box / black-box, targeted / untargeted |
| Defend the original clean model | Reproducible defense protocol; improvement and clean retention score | Reproducible defense protocol; clean-utility and robust-gain score |

Use `qnn` or `qcnn` in submissions. Each attack variant has five model runs.
White-box access includes encoding, circuit, measurements, head and weights.
For black-box access, your group shares one private token; queries are metered
separately for each model, objective and run. See [access instructions](docs/black-box-access.md).

## Repository map

| Location | Use |
|---|---|
| `QAML_CHALLENGE.ipynb` | Runnable walkthrough, model access, example updates, recipes and score calculations |
| `kit/` | Models, training data, attack inputs, fixed preprocessing and model source |
| `docs/` | Rules, setup, scoring, access and submissions |
| `client/` | Prediction client, route selection and practice score calculators |
| `examples/`, `templates/` | Attack format, defense recipes, metadata and write-up |
| `submissions/` | Your completed submission directories |

Keep solutions and feedback private until both regions finish. Never put tokens
in commits, notebook output or issues. Ask for help through your private group
repository or the event's support channel.
