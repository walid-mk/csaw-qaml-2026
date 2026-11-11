"""Select a route for a group token without putting credentials in URLs."""
from urllib.parse import urlsplit, urlunsplit


def prediction_url(base_url, family, objective, run):
    if family not in ('qnn', 'qcnn') or objective not in ('targeted', 'untargeted'):
        raise ValueError('Select qnn/qcnn and targeted/untargeted')
    if type(run) is not int or run not in range(5):
        raise ValueError('Select run 0–4')
    parsed = urlsplit(base_url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('Use the issued HTTPS QAML_PREDICT_URL')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('The URL must not contain credentials, query strings or fragments')
    if parsed.path.rstrip('/') != '/qaml':
        raise ValueError('Use the service base URL ending in /qaml')
    return urlunsplit((parsed.scheme, parsed.netloc,
                      f'/qaml/{family}/{objective}/run-{run}/predict', '', ''))
