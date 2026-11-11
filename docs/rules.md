# Challenge rules

- Use only the assigned organizer models, approved images and issued prediction
  endpoint. This is an offline educational benchmark, with no interaction with
  vehicles, live infrastructure, other teams, or third-party systems.
- Preserve the assigned model family, eight qubits, fixed preprocessing, RY angle
  encoding, active quantum component, and classical head. Quantum noise is excluded.
- The four class indices are `0: speedlimit`, `1: crosswalk`, `2: trafficlight`
  (the physical light), and `3: stop`.
- Attack images are RGB, channels-first, shape `(N, 3, 32, 32)`, finite `float64`
  in `[0,1]`. The maximum change in any pixel channel from the supplied original is
  **8/255 by default**. Preserve the supplied IDs and order.
- Targeted attacks aim for the assigned incorrect class; untargeted attacks aim
  for any incorrect class. Use the supplied input set for that family/access variant.
- Black-box grants bind team, region/edition, variant, target run, and benchmark.
  Each accepted image prediction is charged, including repeats, batches and retries.
  The default allowance is **273 per image per variant/run**. Never try to bypass authentication or limits.
- Defense submissions may use custom training and/or input preprocessing, or a
  compact adversarial-training recipe. See [the protocol interface](defense-protocols.md).
  Retain the quantum architecture and use the approved training pool and five runs.
- Do not train on evaluation records or build label lookup tables. Reused Road
  Signs images are public/exposed; no private-holdout claim is made.
- Keep code, artifacts, feedback and credentials private until both regions close.
  Disclose data sources, dependencies, and material AI assistance in your write-up.

The current limits are published in [config/policy.json](../config/policy.json).
Organizers may publish a revised policy, including disabling a limit. A revision
applies consistently to affected teams and retains usage; results identify their
policy version. Follow the announced effective policy and do not mix comparisons
across different limits. Students cannot change the server policy.
