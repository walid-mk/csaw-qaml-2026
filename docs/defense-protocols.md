# Submit a defense protocol

Adversarial training is **one example**, not a requirement. You may explore other
training losses, regularization, input preprocessing, or a method you develop.
Keep the assigned eight-qubit model family, encoding, measurements and classical
head. An added image transformation runs before the supplied fixed preprocessing;
do not refit or replace that fixed preprocessing. Quantum noise remains excluded.

## Custom protocol

Copy `examples/custom-defense/` to your submission's `defense/` directory. Include
`recipe.json` and `protocol.py` together. The recipe declares schema `2`, your
`family`, `method: custom_protocol`, `source: protocol.py`, a JSON `config`, and
`backward: autograd` or `bpda_identity`. Config values describe your method; they
are not limited to the adversarial-training parameter grid.

Implement these functions:

```python
def fit(model, images, labels, *, seed, config):
    # Optional training: update the supplied ORIGINAL model in place.
    # Use these approved training data only; create fresh optimizer state.
    # Return JSON state needed by your preprocessing (at most 16 KiB).
    return {}

def preprocess(images, *, state, config):
    # Optional input defense; identity is valid for training-only methods.
    # Return finite float64 tensors, same (N,3,32,32) shape, in [0,1].
    return images
```

Each of five runs starts from its assigned original checkpoint. The organizer
reproduces `fit` with the recorded run seed, then uses your transformed input and
trained parameters through the same quantum architecture. You submit source and
reproduction instructions, not opaque trained weights. Use the pinned software
in the kit; disclose additional dependencies and estimated time/memory before
submission so organizers can arrange reproducible execution.

The supplied adapter supports deterministic transforms, including differentiable
ones and non-differentiable transforms evaluated with an identity BPDA surrogate.
BPDA uses the real transform in the forward pass and an approximate backward
pass to generate attacks. It does not change predictions or award robustness
for absent gradients. Describe other required adaptive techniques, such as a
custom surrogate or expectation over transformation for randomized defenses,
in your write-up and contact organizers early. Those methods are welcome, but
their evaluation adapter must be validated before a score can be issued; the
starter does not automatically evaluate every possible defense.

Keep `protocol.py` at most 256 KiB and the recipe at most 16 KiB. Do not access
networks, other teams, credentials, evaluation labels or image-ID lookup tables.
Training uses only `kit/training/`. The attack pixel-change cap applies to
submitted attack images, not to a defense's internal image transformation.
Clean accuracy and adaptive robustness both contribute to your defense score.

## Compact adversarial-training recipe

The six-field recipes in `examples/defense-recipes/` remain supported. Their
allowed choices are radius numerator 4 or 8 (denominator 255), adversarial
fraction .25/.5/.75, attack steps 1/3/5, and learning rate .005/.02. This compact
interface is convenient for that training method; it does not restrict custom
protocols to adversarial training.

Use `scenario: defense`, `access: whitebox`, `objective: null`, `attack_files: []`
and the path to your recipe in `submission.json`. Include source, config,
dependencies, seed handling, runtime estimate and evaluation limitations in your
write-up. See [submission instructions](submissions.md).
