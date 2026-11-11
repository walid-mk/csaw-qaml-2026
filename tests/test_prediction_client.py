import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from client.prediction import PredictionError, _NoRedirect, predict


class PredictionClientTests(unittest.TestCase):
    def response(self, value):
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps(value).encode())
        return opener

    def test_exact_fields_and_single_request(self):
        opener = self.response({"probabilities": [[0.1, 0.2, 0.3, 0.4]], "queries": 1})
        with patch("urllib.request.build_opener", return_value=opener):
            result = predict("https://example.invalid/predict", "test-token", ["image"], [[]])
        self.assertEqual(result["queries"], 1)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(json.loads(request.data), {"ids": ["image"], "images": [[]]})
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")

    def test_failures_do_not_retry_or_echo_secret(self):
        for error in (TimeoutError(), URLError("test-token"),
                      HTTPError("https://example.invalid", 429, "test-token", {}, None)):
            with self.subTest(error=type(error).__name__):
                opener = Mock()
                opener.open.side_effect = error
                with patch("urllib.request.build_opener", return_value=opener):
                    with self.assertRaises(PredictionError) as raised:
                        predict("https://example.invalid/predict", "test-token", ["image"], [[]])
                opener.open.assert_called_once()
                self.assertNotIn("test-token", str(raised.exception))

    def test_bad_probabilities_rejected(self):
        for rows in ([], [[0, 1]], [[True, 0, 0, 0]], [[float("nan"), 0, 0, 1]],
                     [[10**400, 0, 0, 0]], [[0, 0, 0, 0]]):
            with self.subTest(rows=rows):
                opener = self.response({"probabilities": rows})
                with patch("urllib.request.build_opener", return_value=opener):
                    with self.assertRaises(PredictionError):
                        predict("https://example.invalid/predict", "test-token", ["image"], [[]])

    def test_unsafe_url_and_nonfinite_input_rejected_before_network(self):
        with patch("urllib.request.build_opener") as factory:
            for url in ("file:///tmp/test", "http://example.invalid/predict",
                        "https://user:password@example.invalid/predict", "https://example.invalid/?token=x"):
                with self.assertRaises(ValueError):
                    predict(url, "test-token", ["image"], [[]])
            with self.assertRaises(ValueError):
                predict("https://example.invalid/predict", "test-token", ["image"], [[float("nan")]])
            factory.assert_not_called()

    def test_redirect_is_refused(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid"))


if __name__ == "__main__":
    unittest.main()
