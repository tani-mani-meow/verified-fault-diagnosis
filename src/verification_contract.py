"""Fail-closed validation and versioned release decisions.

These checks establish truth of structured feature directions only. They do not
verify free-text causality, diagnostic correctness, or LLM-supplied criticality.
"""
import math
import json
import re
from numbers import Real

STRICT_POLICY = 'strict-v1'
LEGACY_POLICY = 'legacy-0.8'


def parse_response(text):
    if not isinstance(text, str):
        raise ValueError('LLM response must be text')
    text = text.strip()
    wrapped = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)```', text)
    if wrapped:
        text = wrapped.group(1).strip()

    def reject_constant(value):
        raise ValueError(f'Non-JSON numeric constant: {value}')

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def sample_direction(value, thresholds):
    if not isinstance(thresholds, dict):
        raise ValueError('missing feature thresholds')
    low, high = thresholds.get('p25'), thresholds.get('p75')
    if not all(finite_number(v) for v in (value, low, high)):
        raise ValueError('missing or non-finite value/threshold')
    if low >= high:
        raise ValueError('degenerate or reversed percentile thresholds')
    return 'high' if value >= high else 'low' if value <= low else 'neutral'


def claim_direction(value):
    # Explicit aliases for the older public verifier; no substring guessing.
    aliases = {'high': 'high', 'elevated': 'high', 'low': 'low',
               'normal': 'neutral', 'neutral': 'neutral'}
    if not isinstance(value, str) or value.lower() not in aliases:
        raise ValueError('invalid claim direction')
    return aliases[value.lower()]


def validate_claim(claim):
    if not isinstance(claim, dict):
        raise ValueError('claim must be an object')
    if not isinstance(claim.get('feature'), str) or not claim['feature'].strip():
        raise ValueError('claim feature must be a nonempty string')
    claim_direction(claim.get('direction'))
    for key in ('critical', 'action_critical'):
        if key in claim and type(claim[key]) is not bool:
            raise ValueError(f'{key} must be boolean')


def validate_response(response):
    if not isinstance(response, dict):
        raise ValueError('response must be an object')
    if set(response) != {'claims', 'explanation'}:
        raise ValueError('response requires exactly claims and explanation')
    if not isinstance(response.get('explanation'), str) or not response['explanation'].strip():
        raise ValueError('explanation must be a nonempty string')
    if not isinstance(response.get('claims'), list):
        raise ValueError('claims must be a list')
    seen = set()
    for claim in response['claims']:
        validate_claim(claim)
        if set(claim) != {'feature', 'direction', 'action_critical'}:
            raise ValueError('claim has missing or unsupported fields')
        if claim['direction'] not in ('high', 'low') or 'action_critical' not in claim:
            raise ValueError('generated claims require high/low and boolean action_critical')
        if claim['feature'] in seen:
            raise ValueError('duplicate feature claims')
        seen.add(claim['feature'])


def decide_gate(per_claim, mechanism_warnings=(), policy=STRICT_POLICY):
    """Legacy is for labeled historical comparisons; production uses strict-v1."""
    if policy not in (STRICT_POLICY, LEGACY_POLICY):
        raise ValueError(f'Unknown gate policy: {policy}')
    if not per_claim:
        return 'ESCALATE', 'no_verifiable_claims'
    if policy == STRICT_POLICY and any(c.get('validation_error') for c in per_claim):
        return 'ESCALATE', 'invalid_verification_data'
    if any(c.get('action_critical') and not c['sample_true'] for c in per_claim):
        return 'ESCALATE', 'false_action_critical'
    faithfulness = sum(c['sample_true'] for c in per_claim) / len(per_claim)
    if faithfulness < (1.0 if policy == STRICT_POLICY else 0.8):
        return 'ESCALATE', 'low_faithfulness'
    if any(c.get('atypical') and c.get('action_critical') for c in per_claim):
        return 'ESCALATE', 'atypical_action_critical'
    if policy == STRICT_POLICY and mechanism_warnings:
        return 'ESCALATE', 'mechanism_warning'
    return 'RELEASE', 'passed_verification'
