import unittest
from client.access import prediction_url
from client.scoring import (SETTINGS, qnn_defense_points, qcnn_defense_points,
                            attack_points, feedback_summary)


class StudentHelperTests(unittest.TestCase):
    def test_qnn_examples(self):
        b = SETTINGS['qnn_defense']['baseline']
        self.assertEqual(qnn_defense_points(float(b['R0']), float(b['C0'])), 0)
        self.assertEqual(qnn_defense_points(1, 1), 100)
        self.assertEqual(qnn_defense_points(1, float(b['C0']) / 2), 50)
        self.assertAlmostEqual(qnn_defense_points(.65, .90), 13.396415621734134)
        self.assertEqual(qnn_defense_points(.5, .9), 0)
        self.assertEqual(qnn_defense_points(.9, 0), 0)

    def test_qcnn_gates_and_boundaries(self):
        c = SETTINGS['qcnn_defense']
        self.assertEqual(qcnn_defense_points(c['clean_floor'], c['allowed_clean_loss'],
                                            c['full_credit_anchor']), 100)
        self.assertEqual(qcnn_defense_points(.79, 0, .2), 0)
        self.assertEqual(qcnn_defense_points(.9, .07, .2), 0)
        self.assertEqual(qcnn_defense_points(.9, 0, -.1), 0)

    def test_attack_variants(self):
        for variant, settings in SETTINGS['attack_anchors'].items():
            self.assertEqual(attack_points(*variant.split('/'), 0), 0)
            self.assertEqual(attack_points(*variant.split('/'), settings['full_credit']),
                             100 * settings['weight'])

    def test_missing_nonfinite_and_percentage_are_not_scores(self):
        for bad in (None, float('nan'), float('inf'), True, '0.9', 90, -.1):
            with self.assertRaises(ValueError):
                qnn_defense_points(bad, .9)
        self.assertIsNone(feedback_summary({'status': 'not_scored', 'points': None})['points'])

    def test_routes(self):
        for family in ('qnn', 'qcnn'):
            for objective in ('targeted', 'untargeted'):
                for run in range(5):
                    self.assertEqual(prediction_url('https://example.org/qaml', family, objective, run),
                                     f'https://example.org/qaml/{family}/{objective}/run-{run}/predict')
        for base in ('http://example.org/qaml', 'https://user:pass@example.org/qaml',
                     'https://example.org/qaml?token=x', 'https://example.org/wrong'):
            with self.assertRaises(ValueError): prediction_url(base, 'qnn', 'targeted', 0)


if __name__ == '__main__': unittest.main()
