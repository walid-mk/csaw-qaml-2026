"""One metered prediction request; no implicit retries or credential logging."""

import json
import math
import urllib.error
import urllib.parse
import urllib.request


class PredictionError(RuntimeError):
    """A request failed; a timed-out prediction may already have been charged."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect must never carry a team credential to another endpoint.
        return None


def predict(url, token, ids, images, *, timeout=120):
    """Return the gateway response for JSON-compatible ids and RGB CHW images."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("An organizer HTTP(S) prediction URL is required")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Credentials and query parameters do not belong in the URL")
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Non-loopback prediction endpoints must use HTTPS")
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise ValueError("A nonempty private Bearer token is required")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 128:
        raise ValueError("Supply 1–128 original record IDs")
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be nonempty and distinct within this batch")
    if not isinstance(images, list) or len(images) != len(ids):
        raise ValueError("Images must match the ID list")
    payload = json.dumps({"ids": ids, "images": images}, allow_nan=False).encode()
    request = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + token,
        "User-Agent": "CSAW-QAML-Student/1.0",
    })
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(1_000_001)
            if len(data) > 1_000_000:
                raise PredictionError("Oversized evaluator response; do not automatically retry")
            result = json.loads(data)
    except urllib.error.HTTPError as exc:
        # Do not echo response bodies or URLs that could contain private details.
        raise PredictionError(f"HTTP {exc.code}; inspect usage before retrying") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise PredictionError("Connection failure; the request may have been charged") from None
    except (ValueError, UnicodeError):
        raise PredictionError("Invalid evaluator response; do not automatically retry") from None
    probabilities = result.get("probabilities") if isinstance(result, dict) else None
    if not isinstance(probabilities, list) or len(probabilities) != len(ids):
        raise PredictionError("Invalid probability batch")
    for row in probabilities:
        if not isinstance(row, list) or len(row) != 4 or any(
            type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v)
            for v in row
        ) or abs(sum(row) - 1) > 0.000003:
            raise PredictionError("Invalid four-class probability row")
    return result
