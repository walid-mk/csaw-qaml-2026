"""Practice calculations from valid metrics; official points arrive in feedback."""
import json
import math
from pathlib import Path

SETTINGS = json.loads((Path(__file__).resolve().parents[1] / 'config/scoring.json').read_text())


def clip(value):
    return min(1.0, max(0.0, value))


def metric(value, *, signed=False):
    low = -1 if signed else 0
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= 1:
        raise ValueError('Supply a finite measured fraction, not a percentage or missing record')
    return value


def qnn_defense_points(R, C):
    """Preview for complete, valid metrics only; no evaluator status inference."""
    R, C = metric(R), metric(C)
    baseline = SETTINGS['qnn_defense']['baseline']
    r0, c0 = float(baseline['R0']), float(baseline['C0'])
    return 100 * clip((R - r0) / (1 - r0)) * clip(C / c0)


def qcnn_defense_points(C, clean_loss_upper, robust_gain_lower):
    C = metric(C)
    clean_loss_upper = metric(clean_loss_upper, signed=True)
    robust_gain_lower = metric(robust_gain_lower, signed=True)
    cfg = SETTINGS['qcnn_defense']
    if C < cfg['clean_floor'] or clean_loss_upper > cfg['allowed_clean_loss']:
        return 0.0
    return 100 * clip(robust_gain_lower / cfg['full_credit_anchor'])


def attack_points(family, access, objective, success_lower):
    success_lower = metric(success_lower)
    cfg = SETTINGS['attack_anchors'][f'{family}/{access}/{objective}']
    return 100 * cfg['weight'] * clip(success_lower / cfg['full_credit'])


def feedback_summary(record):
    """Display feedback, retaining null points for results that were not scored."""
    return {key: record.get(key) for key in (
        'submission_id', 'family', 'variant', 'status', 'R', 'C', 'points',
        'main_cause', 'other_causes')}
