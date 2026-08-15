"""Local integration artifacts trained in the running environment, without APIs."""
import json
from pathlib import Path
import shutil

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from reproducibility import split_indices


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Tests must not make HTTP requests')
    monkeypatch.setattr('requests.sessions.Session.request', forbidden)


@pytest.fixture(scope='session')
def model_artifacts(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / 'outputs'
    target = tmp_path_factory.mktemp('compatible_model')
    frame = pd.read_csv(source / 'features.csv')
    names = json.loads((source / 'feature_names.json').read_text())
    train, _ = split_indices(frame)
    classes = sorted(frame['class'].unique())
    x = frame[names].iloc[train]
    y = frame['class'].map({c:i for i,c in enumerate(classes)}).iloc[train]
    scaler = StandardScaler().fit(x)
    model = XGBClassifier(objective='multi:softprob', num_class=len(classes),
                          random_state=42, eval_metric='mlogloss', n_jobs=1)
    model.fit(scaler.transform(x), y)
    joblib.dump(model, target / 'xgb_model.joblib')
    joblib.dump(scaler, target / 'scaler.joblib')
    for name in ('feature_names.json', 'feature_thresholds.json', 'kb_calibration.json', 'feature_glossary.json'):
        shutil.copy2(source / name, target / name)
    return target
