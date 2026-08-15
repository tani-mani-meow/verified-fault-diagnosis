import pytest

import pipeline
from verification_contract import (decide_gate, sample_direction, validate_response,
                                   STRICT_POLICY, LEGACY_POLICY, parse_response)
from verifier import check_sample_faithfulness
from test_verifier import MOCK_THRESHOLDS, MOCK_CALIBRATION, MOCK_LIV_RECORDING
from reproducibility import split_indices, sample_id


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True, '1', None])
def test_nonfinite_or_nonnumeric_evidence_rejected(value):
    with pytest.raises(ValueError):
        sample_direction(value, {'p25': 0, 'p75': 1})


@pytest.mark.parametrize('bounds', [{}, {'p25': 0}, {'p25': 1, 'p75': 1},
                                  {'p25': 2, 'p75': 1}, {'p25': 0, 'p75': float('nan')}])
def test_missing_or_degenerate_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        sample_direction(0.5, bounds)


def test_boundary_rules():
    bounds = {'p25': 0, 'p75': 1}
    assert [sample_direction(x, bounds) for x in [0, 0.5, 1]] == ['low', 'neutral', 'high']


@pytest.mark.parametrize('claim', [None, [], {'feature': [], 'direction': 'low'},
    {'feature': 'skewness', 'direction': 'not high'},
    {'feature': 'skewness', 'direction': 'context_dependent'},
    {'feature': 'skewness', 'direction': None},
    {'feature': 'skewness', 'direction': 'low', 'action_critical': 'false'}])
def test_bad_claims_fail_closed(claim):
    result = check_sample_faithfulness(MOCK_LIV_RECORDING, 'LIV', [claim], MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert result['faithfulness'] == 0
    assert decide_gate(result['per_claim'])[0] == 'ESCALATE'


def test_missing_calibration_cannot_pass():
    result = check_sample_faithfulness(MOCK_LIV_RECORDING, 'LIV',
        [{'feature': 'skewness', 'direction': 'low'}], MOCK_THRESHOLDS, {})
    assert result['faithfulness'] == 0
    assert result['per_claim'][0]['validation_error']


def test_strict_gate_rejects_one_false_noncritical_claim():
    claims = [dict(sample_true=True, action_critical=False)] * 4 + [dict(sample_true=False, action_critical=False)]
    assert decide_gate(claims, policy=STRICT_POLICY)[0] == 'ESCALATE'
    assert decide_gate(claims, policy=LEGACY_POLICY)[0] == 'RELEASE'


def test_mechanism_warning_blocks_release():
    claims = [dict(sample_true=True, action_critical=False)]
    assert decide_gate(claims, ['unsupported word']) == ('ESCALATE', 'mechanism_warning')


@pytest.mark.parametrize('response', [[], None, 5, {}, {'claims': {}, 'explanation': 'text'},
    {'claims': [], 'explanation': None}, {'claims': [], 'explanation': '  '}])
def test_response_schema(response):
    with pytest.raises(ValueError):
        validate_response(response)


def test_duplicate_claims_rejected():
    c = dict(feature='skewness', direction='low', action_critical=False)
    with pytest.raises(ValueError, match='duplicate'):
        validate_response(dict(claims=[c, c], explanation='text'))


@pytest.mark.parametrize('response', ['[]', 'null', '{"claims": null, "explanation": "text"}',
                                    'not JSON', "{'claims': [], 'explanation': 'python literal'}"])
def test_pipeline_handles_bad_llm_outputs(monkeypatch, model_artifacts, response):
    import pandas as pd
    frame = pd.read_csv(pipeline.Path(__file__).resolve().parents[1] / 'outputs' / 'features.csv')
    _, test = split_indices(frame)
    row = frame.iloc[test].query("`class` == 'LIV'").iloc[0].to_dict()
    monkeypatch.setenv('NVIDIA_API_KEY', 'offline-test')
    monkeypatch.setattr(pipeline.llm_client, 'generate', lambda *a, **k: response)
    report = pipeline.run_pipeline(row, artifact_dir=model_artifacts)
    assert report['gate_decision'] == 'ESCALATE'
    assert report['explanation'] == ''
    assert report['reason'] in ('unparseable_llm_output', 'invalid_llm_schema')


def test_portable_identity():
    assert sample_id('LIV', r'C:\data\LIV\recording.wav') == 'LIV/recording.wav'


@pytest.mark.parametrize('text', [None, '{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}',
    '```json\n{}\n``` trailing text'])
def test_strict_json_parser(text):
    with pytest.raises(ValueError):
        parse_response(text)


def test_numeric_claim_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match='unsupported fields'):
        validate_response(dict(explanation='text', claims=[dict(feature='skewness', direction='low',
                                                               action_critical=False, value=100)]))


def test_pipeline_withholds_candidate_on_warning(monkeypatch, model_artifacts):
    import pandas as pd
    frame = pd.read_csv(pipeline.Path(__file__).resolve().parents[1] / 'outputs' / 'features.csv')
    _, test = split_indices(frame)
    row = frame.iloc[test].query("`class` == 'LIV'").iloc[0].to_dict()
    monkeypatch.setattr(pipeline, 'check_mechanism_words', lambda *a: ['unsupported mechanism'])
    report = pipeline.run_pipeline(row, mock='correct', artifact_dir=model_artifacts)
    assert report['faithfulness'] == 1
    assert report['reason'] == 'mechanism_warning'
    assert report['explanation'] == ''
    assert report['candidate_explanation']


def test_version_mismatched_model_is_rejected(monkeypatch, model_artifacts):
    from sklearn.exceptions import InconsistentVersionWarning
    import warnings
    def incompatible(*args):
        warnings.warn(InconsistentVersionWarning(estimator_name='StandardScaler',
                      current_sklearn_version='1.7.2', original_sklearn_version='1.9.0'))
    monkeypatch.setattr(pipeline.joblib, 'load', incompatible)
    with pytest.raises(ValueError, match='Incompatible historical model'):
        pipeline.run_pipeline({}, mock=True, artifact_dir=model_artifacts)
