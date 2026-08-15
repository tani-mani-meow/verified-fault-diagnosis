"""Shared, deterministic split and artifact provenance helpers (no network)."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_indices(frame):
    """Match the original stratified split; persist returned row order in runs."""
    return train_test_split(np.arange(len(frame)), test_size=0.2,
                            random_state=42, stratify=frame['class'])


def sample_id(label, file_path):
    # Historical CSVs can contain Windows paths even on non-Windows hosts.
    return f"{label}/{str(file_path).replace(chr(92), '/').split('/')[-1]}"
