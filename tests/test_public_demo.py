"""Keep the public showcases tied to the saved evidence and current gate."""
import json
from pathlib import Path
import re

import pandas as pd
import pytest

from verifier import check_sample_faithfulness, check_mechanism_words, load_thresholds, load_calibration
from verification_contract import decide_gate


ROOT = Path(__file__).resolve().parents[1]


def content(name):
    source = (ROOT / 'demo/src/data/content.js').read_text(encoding='utf-8')
    return json.JSONDecoder().raw_decode(source.split(f'export const {name} = ', 1)[1])[0]


def test_demo_gradient_matches_saved_evidence():
    for bar, filename in zip(content('GRADIENT_BARS'), [
        'faithfulness_audit_classonly.json', 'faithfulness_audit_valuesonly.json',
        'faithfulness_audit.json',
    ]):
        records = json.loads((ROOT / 'outputs' / filename).read_text())
        assert bar['height'] == pytest.approx(sum(c['faithfulness'] for c in records) / len(records))


def test_demo_heatmap_matches_saved_evidence():
    heatmap = content('HEATMAP')
    saved = json.loads((ROOT / 'outputs/feature_ablation.json').read_text())
    for row, values in zip(heatmap['rows'], heatmap['grid']):
        for col, value in zip(heatmap['cols'], values):
            assert value == saved[f'{row}_{col}']['mean_f1']


def test_recorded_demo_decisions_match_current_gate():
    html = (ROOT / 'presentation.html').read_text(encoding='utf-8')
    cases = json.JSONDecoder().raw_decode(html.split('const CASES = ', 1)[1])[0]
    audit = json.loads((ROOT / 'outputs/faithfulness_audit.json').read_text())
    frame = pd.read_csv(ROOT / 'outputs/features.csv')
    thresholds, calibration = load_thresholds(), load_calibration()
    for card in content('DEMO_CARDS'):
        index = int(re.search(r'#(\d+)', card['title'])[1])
        saved = next(c for c in audit if c['index'] == index)
        claims = [dict(feature=c['feature'], direction=c['claimed_dir'],
                       action_critical=c['action_critical']) for c in saved['per_claim']]
        result = check_sample_faithfulness(frame.iloc[index].to_dict(), saved['predicted'],
                                          claims, thresholds, calibration)
        warnings = check_mechanism_words(saved['explanation'], saved['predicted'], thresholds)
        gate, reason = decide_gate(result['per_claim'], warnings)
        assert card['verdict'] == f'{gate} / {reason}'
        assert card['llm'] == saved['explanation']
        assert cases[card['id']]['gate'] == gate
        assert cases[card['id']]['expl'] == saved['explanation']


def test_feature_matrix_has_portable_unique_identities():
    frame = pd.read_csv(ROOT / 'outputs/features.csv')
    assert frame['file_path'].is_unique
    for label, path in zip(frame['class'], frame['file_path']):
        assert path.startswith(label + '/')
        assert len(path.split('/')) == 2
        assert '\\' not in path and ':' not in path
