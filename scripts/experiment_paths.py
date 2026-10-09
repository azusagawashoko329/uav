"""Resolve experiment output directories under the project registry."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID_DATASETS = {'da','uit','da_uit'}
VALID_FAMILIES = {'appearance','behavior','fusion','aggregation','boundary','gating','audit'}


def require_experiment(exp: str, *, dataset: str | None = None, family: str | None = None) -> Path:
    path = Path(exp)
    parts = path.parts
    if path.is_absolute() or len(parts) < 3 or '..' in parts or '.' in parts:
        raise ValueError(f'experiment must be a registry-relative path: {exp!r}')
    if parts[0] not in VALID_DATASETS or parts[1] not in VALID_FAMILIES:
        raise ValueError(f'unknown experiment dataset/family: {exp!r}')
    if dataset is not None and parts[0] != dataset:
        raise ValueError(f'experiment dataset must be {dataset}: {exp!r}')
    if family is not None and parts[1] != family:
        raise ValueError(f'experiment family must be {family}: {exp!r}')
    target = ROOT / 'experiments' / path
    if not (target / 'record.json').is_file():
        raise FileNotFoundError(f'create and commit experiment record first: {target / "record.json"}')
    return target
