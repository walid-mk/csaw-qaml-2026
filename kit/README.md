# QAML model and data kit

Your kit contains the original and hardened
**white-box** QNN/QCNN models, fixed preprocessing, four attack-input variants,
and the approved training records. Start with the [notebook](../QAML_CHALLENGE.ipynb).

There are five original and five hardened checkpoints per family. Hidden
black-box weights, true evaluation labels, private training seeds, reference
solutions, attack witnesses, score databases and reserved images are excluded.

```python
from kit.sdk import verify_assets, load_model, load_inputs
verify_assets()
model = load_model("qnn", run=0, kind="hardened")
inputs = load_inputs("qnn", "whitebox")
```

`training/road-signs-training.npz` contains `ids`, `images`, and training `labels`.
`inputs/*.npz` contains `ids`, `images`, and assigned incorrect `targets`, with no
true evaluation labels. Arrays use the frozen RGB 32×32 preprocessing and pixel
range [0,1]. The Road Signs benchmark is **public/exposed**, not a private holdout.

The implementation sources under `src/` are the exact reviewed model sources.
Use QNN/QCNN through the SDK to load the correct models. No training or inference requires the
organizer's private repository or a Kaggle account.

See [NOTICE](NOTICE.md), [data attribution](DATA_ATTRIBUTION.md), and the
[student guide](../STUDENT_GUIDE.md). `manifest.json` binds the delivered bytes.
