# Black-box access: one token for your group

**Current service URL:** `https://conservative-podcast-bones-trailers.trycloudflare.com/qaml`

Use this as `QAML_PREDICT_URL`. Prediction requests require an enabled group token.
If this temporary address changes, the replacement will be published here.

Every group can participate. Your group receives privately:

- `QAML_PREDICT_URL`: the HTTPS service base URL, ending in `/qaml`.
- `QAML_QUERY_TOKEN`: one random group token, shared only with your teammates.

Your token works for QNN and QCNN, targeted and untargeted attacks, and runs 0–4.
Each combination has its **own default 273-query allowance per image**. Teammates share
these allowances; changing computers or reconnecting does not reset them.
The URL and token will be delivered together when prediction access is available.

## Use the notebook

Open [QAML_CHALLENGE.ipynb](../QAML_CHALLENGE.ipynb). Select the family, objective
and run, set `RUN_BLACKBOX_REQUEST = True`, then enter the issued URL and token
when prompted. Token entry is hidden and the notebook does not print it.
One run of that example consumes **one query** for the selected image and grant.

## Use your own Python program

```python
import getpass
import os
from client.access import prediction_url
from client.prediction import predict
from kit.sdk import load_inputs

base = os.environ.get('QAML_PREDICT_URL') or input('Issued service URL: ').strip()
token = os.environ.get('QAML_QUERY_TOKEN') or getpass.getpass('Group token: ')
batch = load_inputs('qnn', 'blackbox')
url = prediction_url(base, 'qnn', 'untargeted', run=0)
result = predict(url, token, batch['ids'][:1].tolist(), batch['images'][:1].tolist())
print(result['probabilities'])
print('Used:', result['queries'], 'Aggregate allowance:', result['budget'])
del token
```

You can instead set the two environment variables privately. For Bash, avoid
putting the token into shell history:

```bash
read -r -p 'Issued service URL: ' QAML_PREDICT_URL
read -r -s -p 'Group token: ' QAML_QUERY_TOKEN
export QAML_PREDICT_URL QAML_QUERY_TOKEN
```

Never use a GitHub password, GitHub access token or Discord token here.

## Outputs and limits

The request contains exactly `ids` and `images`. You receive four probabilities,
rounded to six decimals, in the order **speedlimit, crosswalk, trafficlight,
stop**, plus cumulative usage and the aggregate allowance for that grant.
You do not receive gradients or hidden weights. The probabilities describe the
selected hidden model; local white-box outputs are not substitutes.

Keep original IDs and supply finite RGB `float64` images of shape `(N,3,32,32)`
in `[0,1]`, with maximum original-relative pixel-channel change **8/255**.
Start with one image; a batch may contain up to 128 distinct IDs. Every image
costs one query, including repeated requests. Usage is reserved before prediction,
so a timeout may consume queries. There are no automatic retries.

| Response | Your next step |
|---|---|
| `400` | Check IDs, shape, pixel range and the original-relative cap; correct the request |
| `403` | Check the issued token and group access; contact support privately if needed |
| `429` | The image or aggregate allowance is exhausted; stop using that grant |
| Timeout / `500` / `503` | Keep the time, route and IDs; contact support without sharing the token |

The service does not send a near-limit warning. Track returned usage and your
per-image calls. A different route has a different allowance; it does not
give more queries against an exhausted variant. Do not seek additional tokens
to bypass a limit. Report a leaked token privately so it can be replaced while
preserving your group's usage.

## Service address and current policy

Access uses the organizer's online computer through a free Cloudflare tunnel.
You do not need an ngrok account or a paid service. Use the current issued
`QAML_PREDICT_URL`; the tunnel address may change after a restart. If organizers
announce a replacement URL, update that variable. Your group token and usage
remain the same. A temporary outage does not reset an allowance.

Read `/qaml/policy` at the issued host to inspect the active version without
spending a model query. Prediction responses also include `policy` and
`policy_sha256`. The default limits are 8/255 and 273. A disabled pixel cap still
requires valid image shape, IDs and pixels in [0,1]. With an explicitly disabled
query limit, `queries` still increases and `budget` is `null` (unlimited), rather
than zero. Ask organizers privately about a limit adjustment; do not bypass it.
