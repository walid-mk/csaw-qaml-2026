# Challenge assets

Use the model and data files in [kit/](../kit/README.md): 20 public checkpoints,
four attack-input variants, fixed preprocessing and 889 labeled training crops.
The kit's SHA-256 manifest lets you check that you have the correct files.

Run `kit.sdk.verify_assets()` before use. You receive the full white-box model
implementation and weights. For hidden black-box targets, use the prediction
service and your group's credentials. Evaluation labels are not supplied.

Road Signs is a public/exposed educational benchmark. Follow the kit's
[usage notice](../kit/NOTICE.md) and [data attribution](../kit/DATA_ATTRIBUTION.md).
