"""Rebuild the classifier locally without overwriting historical artifacts."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reproducibility import sha256, sample_id, split_indices


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-audio', choices=['sample', 'all', 'none'], default='sample',
                        help='Compare extracted features with the CSV (sample = one recording per class).')
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    # Immutable run directories prevent accidental replacement or mixed runs.
    dest.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'outputs'
    captured = {}
    for base in [ROOT / 'src', ROOT / 'scripts', ROOT / 'tests', source]:
        for path in sorted(base.rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                    and dest not in path.parents
                    and 'reproducibility' not in path.relative_to(ROOT).parts):
                captured[path.relative_to(ROOT).as_posix()] = sha256(path)
                if base != source:
                    snapshot = dest / 'source' / path.relative_to(ROOT)
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, snapshot)
    for name in ['README.md', 'paper_draft.md', 'paper/main.tex', 'requirements.txt', 'pytest.ini']:
        path = ROOT / name
        if path.exists():
            captured[name] = sha256(path)
            snapshot = dest / 'source' / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot)
    packages = dict(sorted((d.metadata['Name'], d.version) for d in distributions() if d.metadata['Name']))
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(), 'packages': packages,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'input_sha256': captured,
        'scope': 'Local retraining from existing feature matrix; historical LLM outputs are not regenerated.',
    }
    write_json(dest / 'manifest.json', manifest)
    (dest / 'environment.freeze.txt').write_text(''.join(f'{k}=={v}\n' for k,v in packages.items()), encoding='utf-8')
    frame = pd.read_csv(source / 'features.csv')
    names = json.loads((source / 'feature_names.json').read_text())
    if names != [c for c in frame if c not in ('class', 'file_path')]:
        raise ValueError('Feature matrix and model feature order differ')
    if not np.isfinite(frame[names].to_numpy()).all():
        raise ValueError('Non-finite feature values')
    ids = [sample_id(c, p) for c,p in zip(frame['class'], frame['file_path'])]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample identities')
    audio_hashes = {}
    for identity in ids:
        path = ROOT / 'data' / 'AirCompressorDataset' / identity
        if not path.exists():
            raise FileNotFoundError(f'Local recording missing: {identity}')
        audio_hashes[identity] = sha256(path)
    write_json(dest / 'recording_sha256.json', audio_hashes)
    checked = []
    if args.verify_audio != 'none':
        import librosa
        from features import extract_features
        selection = frame if args.verify_audio == 'all' else frame.groupby('class', sort=True).head(1)
        print(f'Checking extracted features for {len(selection)} local recordings...', flush=True)
        for i,row in selection.iterrows():
            signal, sr = librosa.load(str(ROOT / 'data' / 'AirCompressorDataset' / ids[i]), sr=None)
            values = extract_features(signal, sr)
            if list(values) != names or not np.allclose(list(values.values()), row[names].to_numpy(dtype=float), rtol=1e-6, atol=1e-8):
                raise ValueError(f'Extracted features differ from CSV: {ids[i]}')
            checked.append(ids[i])
    write_json(dest / 'feature_check.json', {'mode': args.verify_audio, 'checked_ids': checked,
                                            'rtol': 1e-6, 'atol': 1e-8})
    train, test = split_indices(frame)
    write_json(dest / 'split.json', {
        'features_sha256': sha256(source / 'features.csv'), 'seed': 42,
        'train_indices': train.tolist(), 'test_indices': test.tolist(),
        'train_ids': [ids[i] for i in train], 'test_ids': [ids[i] for i in test],
        'limitation': 'Stratification by fault label does not establish independent recording sessions.',
    })
    classes = sorted(frame['class'].unique())
    x = frame[names]
    y = frame['class'].map({c:i for i,c in enumerate(classes)}).to_numpy()
    params = dict(objective='multi:softprob', num_class=len(classes), random_state=42,
                  eval_metric='mlogloss', n_jobs=1)
    scaler = StandardScaler().fit(x.iloc[train])
    model = XGBClassifier(**params).fit(scaler.transform(x.iloc[train]), y[train])
    pred = model.predict(scaler.transform(x.iloc[test]))
    joblib.dump(scaler, dest / 'scaler.joblib')
    joblib.dump(model, dest / 'xgb_model.joblib')
    for name in ['feature_names.json', 'feature_thresholds.json', 'kb_calibration.json', 'feature_glossary.json']:
        shutil.copy2(source / name, dest / name)
    # Preserve the measured feature definition; repair only its human-readable label.
    glossary = json.loads((dest / 'feature_glossary.json').read_text())
    for key in ('label', 'description'):
        glossary['spectral_rolloff_85'][key] = 'frequency below which 85% of FFT magnitude lies (Hz)'
    write_json(dest / 'feature_glossary.json', glossary)
    # Check that the stored thresholds really derive from this exact training split.
    thresholds = json.loads((source / 'feature_thresholds.json').read_text())
    errors = []
    for feature in names:
        for group, values in [('global', x.iloc[train][feature])] + [
            (c, frame.iloc[train].loc[frame.iloc[train]['class'] == c, feature]) for c in classes
        ]:
            recorded = thresholds['global'][feature] if group == 'global' else thresholds['classes'][group][feature]
            for pct in [10,25,50,75,90]:
                if not np.isclose(recorded[f'p{pct}'], np.percentile(values, pct), rtol=1e-10, atol=1e-12):
                    errors.append(f'{group}/{feature}/p{pct}')
    if errors:
        raise ValueError(f'Training threshold mismatch: {errors[:10]}')
    calibration = json.loads((source / 'kb_calibration.json').read_text())
    for label, features in calibration.items():
        for feature, stored in features.items():
            stats = thresholds['global'][feature]
            median = thresholds['classes'][label][feature]['p50']
            p40 = stats['p25'] + 0.6 * (stats['p50'] - stats['p25'])
            p60 = stats['p50'] + 0.4 * (stats['p75'] - stats['p50'])
            expected = 'high' if median >= p60 else 'low' if median <= p40 else 'neutral'
            if expected != stored['data_direction']:
                raise ValueError(f'Calibration direction mismatch: {label}/{feature}')
    print('Training thresholds verified; running five-fold CV...', flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(x, y))
    write_json(dest / 'cv_splits.json', [{'train': a.tolist(), 'test': b.tolist()} for a,b in folds])
    scores = cross_val_score(Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier(**params))]),
                             x, y, cv=folds, scoring='accuracy', n_jobs=1)
    results = dict(rows=len(frame), feature_count=len(names), class_names=classes,
                   train_count=len(train), test_count=len(test), accuracy=float(accuracy_score(y[test], pred)),
                   macro_f1=float(f1_score(y[test], pred, average='macro')),
                   cv_scores=scores.tolist(), cv_mean=float(scores.mean()), cv_std=float(scores.std()),
                   thresholds_match_training_split=True,
                   calibration_directions_match_thresholds=True,
                   audio_features_checked=len(checked),
                   unique_recording_hashes=len(set(audio_hashes.values())),
                   cv_scope='Separate full-dataset CV estimate; not tuning against the held-out test set.')
    write_json(dest / 'metrics.json', results)
    write_json(dest / 'predictions.json', [dict(index=int(i), sample_id=ids[i], true=classes[y[i]], predicted=classes[p]) for i,p in zip(test,pred)])
    write_json(dest / 'model_metadata.json', {'class_names': classes, 'parameters': params,
                                             'booster_config': json.loads(model.get_booster().save_config()),
                                             'packages': packages, 'features_sha256': sha256(source / 'features.csv')})
    write_json(dest / 'output_sha256.json', {p.relative_to(dest).as_posix():sha256(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
