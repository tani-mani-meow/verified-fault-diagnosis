"""Deterministic checks of structured claims against measured feature values."""
import json
from pathlib import Path

from knowledge_base import KNOWLEDGE_BASE
from verification_contract import claim_direction, sample_direction, validate_claim, decide_gate, finite_number


def check_mechanism_words(explanation, predicted_class, thresholds):
    """Conservative lexical heuristic, NOT verification of causal prose."""
    supported = []
    for feature in ('kurtosis', 'crest_factor'):
        global_stats = thresholds.get('global', {}).get(feature, {})
        p50, p75 = global_stats.get('p50'), global_stats.get('p75')
        class_p50 = thresholds.get('classes', {}).get(predicted_class, {}).get(feature, {}).get('p50')
        if not all(finite_number(v) for v in (p50, p75, class_p50)) or p50 > p75:
            return ['missing or invalid mechanism evidence']
        supported.append(class_p50 >= p50 + 0.4 * (p75 - p50))
    if any(supported):
        return []
    return [f"Unsupported mechanism word: {term}" for term in
            ('impulsive', 'impact', 'spike', 'knock', 'slam', 'hammer', 'transient', 'burst')
            if term in explanation.lower()]


def load_thresholds(artifact_dir=None):
    root = Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parents[1] / 'outputs'
    return json.loads((root / 'feature_thresholds.json').read_text(encoding='utf-8'))


def load_calibration(artifact_dir=None):
    root = Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parents[1] / 'outputs'
    return json.loads((root / 'kb_calibration.json').read_text(encoding='utf-8'))


def _evaluate(recording, predicted_class, claim, thresholds, calibration):
    result = dict(feature=None, claimed_dir=None, sample_dir=None, class_dir=None,
                  sample_true=False, class_consistent=False, atypical=False,
                  action_critical=False)
    try:
        validate_claim(claim)
        feature = claim['feature']
        direction = claim_direction(claim['direction'])
        result.update(feature=feature, claimed_dir=claim['direction'],
                      action_critical=claim.get('critical', False) or claim.get('action_critical', False))
        if feature not in KNOWLEDGE_BASE.get(predicted_class, {}).get('grounding_features', {}):
            raise ValueError('feature not grounded for this class')
        if feature not in recording:
            raise ValueError(f"feature '{feature}' not found in recording features")
        sample_dir = sample_direction(recording[feature], thresholds.get('global', {}).get(feature))
        class_dir = calibration.get(predicted_class, {}).get(feature, {}).get('data_direction')
        if class_dir not in ('high', 'low', 'neutral'):
            raise ValueError('missing or invalid calibration direction')
        result.update(sample_dir=sample_dir, class_dir=class_dir,
                      sample_true=bool(direction == sample_dir),
                      class_consistent=bool(direction == class_dir),
                      atypical=bool(direction == sample_dir and direction != class_dir))
    except (ValueError, TypeError, AttributeError) as exc:
        result.update(reason=str(exc), validation_error=True)
    return result


def check_sample_faithfulness(recording_features, predicted_class, claims, thresholds=None, calibration=None):
    if thresholds is None:
        thresholds = load_thresholds()
    if calibration is None:
        calibration = load_calibration()
    if not isinstance(claims, list):
        raise ValueError('claims must be a list')
    per_claim = [_evaluate(recording_features, predicted_class, c, thresholds, calibration) for c in claims]
    return {'per_claim': per_claim,
            'faithfulness': sum(c['sample_true'] for c in per_claim) / len(per_claim) if per_claim else 0.0,
            'class_consistency': sum(c['class_consistent'] for c in per_claim) / len(per_claim) if per_claim else 0.0}


def verify_claim(recording_features, predicted_class, claim, thresholds, calibration):
    """Compatibility API: requires both sample truth and class consistency."""
    result = _evaluate(recording_features, predicted_class, claim, thresholds, calibration)
    reason = result.get('reason', '')
    if not reason and not result['class_consistent']:
        reason = ('feature not directionally characteristic for this class' if result['class_dir'] == 'neutral'
                  else 'direction mismatch between claim and class calibration')
    if not reason and not result['sample_true']:
        reason = 'recording value does not support the claimed direction'
    feature = result['feature']
    return dict(claim=claim, passed=bool(result['sample_true'] and result['class_consistent']),
                actual_value=recording_features.get(feature) if feature else None,
                reference={'sample_dir': result['sample_dir'], 'class_dir': result['class_dir']}, reason=reason)


def verify_explanation(recording_features, predicted_class, claims, action_critical_idx=None, thresholds=None, calibration=None):
    if thresholds is None:
        thresholds = load_thresholds()
    if calibration is None:
        calibration = load_calibration()
    if not isinstance(claims, list):
        raise ValueError('claims must be a list')
    per_claim = []
    for index, claim in enumerate(claims):
        result = verify_claim(recording_features, predicted_class, claim, thresholds, calibration)
        result['critical'] = bool(isinstance(claim, dict) and
                                  (claim.get('critical') is True or claim.get('action_critical') is True)) or (
                                      action_critical_idx is not None and index in action_critical_idx)
        per_claim.append(result)
    decision, _ = decide_gate([dict(sample_true=c['passed'], action_critical=c['critical']) for c in per_claim])
    return dict(per_claim=per_claim,
                faithfulness_score=sum(c['passed'] for c in per_claim) / len(per_claim) if per_claim else 0.0,
                escalate=decision == 'ESCALATE')
