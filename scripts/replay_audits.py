"""Reverify stored claims locally, keeping historical audit files unchanged."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reproducibility import sha256, sample_id, split_indices
from verification_contract import STRICT_POLICY, LEGACY_POLICY, decide_gate, validate_response
from verifier import load_calibration, load_thresholds, check_sample_faithfulness, check_mechanism_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'outputs'
    frame = pd.read_csv(source / 'features.csv')
    _, test = split_indices(frame)
    thresholds, calibration = load_thresholds(), load_calibration()
    results, sets, per_condition = {}, [], {}
    inputs = {name:sha256(source/name) for name in ['features.csv','feature_thresholds.json','kb_calibration.json']}
    for condition, filename in [('sample_grounded','faithfulness_audit.json'),
                                ('values_only','faithfulness_audit_valuesonly.json'),
                                ('class_only','faithfulness_audit_classonly.json')]:
        path = source / filename
        inputs[filename] = sha256(path)
        cases = json.loads(path.read_text())
        ids = [c['index'] for c in cases]
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(test)):
            raise ValueError('Duplicate audit rows or rows outside held-out test split')
        sets.append(set(ids))
        reviewed = []
        for case in sorted(cases, key=lambda c:c['index']):
            row = frame.iloc[case['index']].to_dict()
            if row['class'] != case['true_label']:
                raise ValueError('Stored audit label differs from feature matrix')
            # Original response envelopes were not retained: reconstruction is explicitly limited.
            claims = [dict(feature=c.get('feature'), direction=c.get('claimed_dir'),
                           action_critical=c.get('action_critical')) for c in case['per_claim']]
            response = dict(claims=claims, explanation=case['explanation'])
            schema_error = None
            try:
                validate_response(response)
            except ValueError as exc:
                schema_error = str(exc)
            verified = check_sample_faithfulness(row, case['predicted'], claims, thresholds, calibration)
            warnings = check_mechanism_words(case['explanation'], case['predicted'], thresholds)
            strict, reason = decide_gate(verified['per_claim'], warnings)
            if schema_error:
                strict, reason = 'ESCALATE', 'invalid_llm_schema'
            legacy, _ = decide_gate(case['per_claim'], policy=LEGACY_POLICY)
            reviewed.append(dict(index=case['index'], sample_id=sample_id(row['class'],row['file_path']),
                predicted=case['predicted'], historical_gate=case['gate_decision'],
                historical_faithfulness=case['faithfulness'], legacy_gate_replayed=legacy,
                strict_gate=strict, strict_reason=reason, schema_error=schema_error,
                faithfulness=verified['faithfulness'], per_claim=verified['per_claim'], mechanism_warnings=warnings))
        per_condition[condition] = reviewed
        results[condition] = dict(n=len(reviewed),
            historical_mean_faithfulness=float(np.mean([c['historical_faithfulness'] for c in reviewed])),
            reverified_mean_faithfulness=float(np.mean([c['faithfulness'] for c in reviewed])),
            historical_releases=sum(c['historical_gate']=='RELEASE' for c in reviewed),
            strict_releases=sum(c['strict_gate']=='RELEASE' for c in reviewed),
            legacy_gate_disagreements=sum(c['legacy_gate_replayed']!=c['historical_gate'] for c in reviewed),
            changed_indices=[c['index'] for c in reviewed if c['strict_gate']!=c['historical_gate']])
        (args.output_dir / f'{condition}.json').write_text(json.dumps(reviewed, indent=2, allow_nan=False)+'\n')
    if not sets[0] == sets[1] == sets[2]:
        raise ValueError('Conditions use different sample identities')
    # Resample paired recordings, never individual claims, for exploratory intervals.
    rng = np.random.default_rng(42)
    n = len(sets[0])
    indices = rng.integers(0, n, size=(10000,n))
    comparisons = {}
    for a,b in [('sample_grounded','values_only'), ('values_only','class_only')]:
        difference = np.array([c['faithfulness'] for c in per_condition[a]]) - np.array([c['faithfulness'] for c in per_condition[b]])
        comparisons[f'{a}_minus_{b}'] = dict(mean=float(difference.mean()),
            percentile_bootstrap_95_ci=np.quantile(difference[indices].mean(axis=1), [0.025,0.975]).tolist())
    summary = dict(policy=STRICT_POLICY, conditions=results, paired_differences=comparisons,
        input_sha256=inputs, source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__), ROOT/'src/verifier.py', ROOT/'src/verification_contract.py', ROOT/'src/knowledge_base.py']},
        scope='Offline replay of reconstructed stored claims using historical predictions, not fresh LLM experiments.',
        limitations=['Raw LLM response envelopes and original prompts were not saved; schema replay cannot recover omitted fields.',
                     'Claims do not verify prose coverage, causal truth, or classifier correctness.',
                     'Bootstrap intervals assume independent recordings; recording-session independence is unestablished.',
                     'Mechanism warnings are a lexical heuristic, not a causal validity test.'])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
