"""Portable RGB action-outcome release; no Torch or Genesis dependency."""

import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.data.storage import file_sha256, json_bytes, relative_path
from cloth_fling.frozen_laydown import CONFIG_SHA256
from cloth_fling.learning import transformed_rgb

SPLITS = ('train', 'validation', 'test')
V1_COUNTS = {'train': (308, 272), 'validation': (23, 22), 'test': (52, 52)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def project_episode(root, episode_name):
    """Keep every recorded outcome; only settled valid executed actions get targets."""
    root = Path(root)
    verification = verify_dataset(root)
    if not verification['valid']:
        raise ValueError(f'dataset integrity failure: {verification["errors"]}')
    projected = []
    for row in read_transitions(root).to_pylist():
        support = read_support(root, row['evidence_path'])
        selection = support.get('evidence', {}).get('selection', {})
        encoding = selection.get('encoding')
        settled = support.get('diagnostics', {}).get('settled') is True
        feasible = bool(encoding and selection['valid_action_mask'][
            tuple(encoding['map_index_tvu'])])
        eligible = bool(row['execution_valid'] and row['reward'] is not None
                        and settled and encoding and feasible)
        record = {k: row[k] for k in (
            'run_id', 'episode_id', 'step_id', 'transition_id', 'split', 'reset_group_id',
            'reset_id', 'reset_sha256', 'cloth_id', 'seed', 'policy', 'execution_valid',
            'terminated', 'truncated', 'reason', 'end_reason', 'coverage_before',
            'coverage_after', 'coverage_gain', 'reward', 'area_before_m2', 'area_after_m2',
            'flat_area_m2', 'action_u0', 'action_v0', 'action_u1', 'action_v1',
            'before_observation_id', 'after_observation_id', 'simulated_seconds', 'wall_seconds',
        )}
        for prefix in ('before', 'after'):
            ref = row[f'{prefix}_rgb_path']
            record[f'{prefix}_rgb'] = {
                'bytes': relative_path(root, ref).read_bytes(), 'path': None,
            } if ref else None
            record[f'{prefix}_rgb_sha256'] = row[f'{prefix}_rgb_sha256']
        record.update(
            episode_key=episode_name, settled=settled, eligible=eligible,
            failed=not row['execution_valid'] or not settled,
            supervised_target=float(row['reward']) if eligible else None,
            executed_feasible=feasible,
            action_encoding_json=json.dumps(encoding, sort_keys=True),
            map_index_tvu=encoding['map_index_tvu'] if encoding else None,
            to_original=encoding['to_original'] if encoding else None,
        )
        projected.append(record)
    return projected


def write_split(root, split, rows):
    if any(row['split'] != split for row in rows):
        raise ValueError('split mismatch')
    path = Path(root) / 'data' / f'{split}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    # Use a stable image struct even if every path is null.
    for name in ('before_rgb', 'after_rgb'):
        i = table.schema.get_field_index(name)
        table = table.set_column(i, name, table[name].cast(pa.struct([
            ('bytes', pa.binary()), ('path', pa.string()),
        ])))
    pq.write_table(table, path, compression='zstd')


def check_split_isolation(rows):
    groups = {}
    for split, examples in rows.items():
        for row in examples:
            if row['split'] != split:
                raise ValueError('split mismatch')
            for key in ('reset_group_id', 'reset_id', 'episode_key'):
                identity = (key, row[key])
                if groups.setdefault(identity, split) != split:
                    raise ValueError('split/episode leakage')


def release_manifest(root, **metadata):
    root = Path(root)
    files = {str(p.relative_to(root)): {'sha256': file_sha256(p),
             'size_bytes': p.stat().st_size} for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'release-manifest.json'}
    manifest = {'schema': 1, 'frozen_config_sha256': CONFIG_SHA256,
                'files': files, **metadata}
    write_json(root / 'release-manifest.json', manifest)
    return manifest


def verify_release(root):
    root = Path(root)
    manifest = json.loads((root / 'release-manifest.json').read_text())
    if manifest['schema'] != 1 or manifest['frozen_config_sha256'] != CONFIG_SHA256:
        raise ValueError('incompatible release configuration')
    for ref, expected in manifest['files'].items():
        path = relative_path(root, ref)
        if path.stat().st_size != expected['size_bytes'] or file_sha256(path) != expected['sha256']:
            raise ValueError(f'release artifact corruption: {ref}')
    return manifest


def read_split(root, split='validation'):
    if split not in SPLITS:
        raise ValueError('unknown split')
    manifest = verify_release(root)
    ref = f'data/{split}.parquet'
    if ref not in manifest['files']:
        raise ValueError('split missing from release manifest')
    rows = pq.ParquetFile(relative_path(Path(root), ref)).read().to_pylist()
    if any(row['split'] != split for row in rows):
        raise ValueError('split mismatch')
    return rows


def decode_rgb(value):
    with Image.open(io.BytesIO(value['bytes'])) as image:
        if image.mode != 'RGB' or image.format != 'PNG':
            raise ValueError('expected RGB PNG')
        return np.asarray(image).copy()


def eligible_arrays(root, split='train'):
    rows = read_split(root, split)
    images, pixels, targets = [], [], []
    for row in rows:
        if not row['eligible']:
            if row['supervised_target'] is not None:
                raise ValueError('ineligible action has a supervised label')
            continue
        encoding = json.loads(row['action_encoding_json'])
        t, v, u = row['map_index_tvu']
        if (not row['execution_valid'] or not row['settled'] or not row['executed_feasible']
                or not (0 <= t < 96 and 0 <= v < 64 and 0 <= u < 64)
                or encoding['codec'] != 'reference_spatial_v1'
                or encoding['map_index_tvu'] != row['map_index_tvu']
                or encoding['to_original'] != row['to_original']
                or row['supervised_target'] != row['coverage_gain']):
            raise ValueError('invalid supervised example')
        images.append(transformed_rgb(decode_rgb(row['before_rgb']), row['to_original']))
        pixels.append((v, u))
        targets.append(row['supervised_target'])
    if not images:
        raise ValueError('no eligible supervised examples')
    sources = [{'relative_file': f'data/{split}.parquet',
                'sha256': file_sha256(Path(root) / 'data' / f'{split}.parquet')}]
    return np.stack(images), np.asarray(pixels), np.asarray(targets, np.float32), sources


def fixture_bundles(root, split='validation'):
    from cloth_fling.reference_tune import read_hashed_bundle

    manifest = verify_release(root)
    root = Path(root)
    records = json.loads((root / 'fixtures-manifest.json').read_text())
    if records['config_sha256'] != CONFIG_SHA256:
        raise ValueError('fixture configuration changed')
    seen, results = {}, []
    for case in records['cases']:
        previous = seen.setdefault(case['state_hash'], case['split'])
        if previous != case['split']:
            raise ValueError('fixture split leakage')
        ref = case['fixture']
        path = relative_path(root, ref['path'])
        if (ref['path'] not in manifest['files'] or file_sha256(path) != ref['sha256']
                or not case['accepted'] or not case['cleanup']['passed']):
            raise ValueError('fixture identity/acceptance mismatch')
        bundle = json.loads(path.read_text())
        arrays = bundle['arrays']
        array_path = path.parent / arrays['path']
        if (str(array_path.relative_to(root)) not in manifest['files']
                or file_sha256(array_path) != arrays['sha256']):
            raise ValueError('fixture array corruption')
        state = read_hashed_bundle(path.parent, path.name)
        if state['state_hash'] != case['state_hash'] or not np.asarray(state['free']).all():
            raise ValueError('fixture state mismatch')
        if case['split'] == split:
            results.append((case, state))
    if not results:
        raise ValueError('no fixtures in selected split')
    return results


def export_v1(source, output):
    """Reproduce dataset v1 from the completed original pilot only."""
    source, output = Path(source).resolve(), Path(output).resolve()
    protocol = json.loads((source / 'protocol.json').read_text())
    ledger = json.loads((source / 'ledger.json').read_text())
    if (protocol['version'] != 'image_learning_pilot_v1' or ledger['status'] != 'complete'
            or any(s['status'] != 'complete' for s in ledger['stages'])
            or len(ledger['actions']) != 383):
        raise ValueError('only the completed original image-learning pilot is exportable')
    output.mkdir(parents=True, exist_ok=False)
    from cloth_fling.learning_pilot import fixtures

    records, _ = fixtures(source)
    sources, rows = {}, {s: [] for s in SPLITS}
    for episode in sorted((source / 'episodes').iterdir()):
        result = json.loads((episode / 'result.json').read_text())
        dataset = episode / 'dataset'
        if file_sha256(dataset / 'manifest.json') != result['dataset_manifest_sha256']:
            raise ValueError('episode manifest changed')
        projection = project_episode(dataset, episode.name)
        rows[result['split']].extend(projection)
        sources[str((dataset / 'manifest.json').relative_to(source))] = file_sha256(
            dataset / 'manifest.json')
    counts = {s: (len(rows[s]), sum(r['eligible'] for r in rows[s])) for s in SPLITS}
    if counts != V1_COUNTS:
        raise ValueError(f'v1 count mismatch: {counts}')
    check_split_isolation(rows)
    for split, examples in rows.items():
        write_split(output, split, examples)
    shutil.copyfile(source / 'fixtures-manifest.json', output / 'fixtures-manifest.json')
    for case in records:
        ref = case['fixture']['path']
        path = relative_path(source, ref)
        target = relative_path(output, ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        array = json.loads(path.read_text())['arrays']['path']
        shutil.copyfile(path.parent / array, target.parent / array)
    for name in ('protocol.json', 'ledger.json', 'implementation.json', 'runtime.json',
                 'frozen-laydown.json', 'fixtures-manifest.json'):
        sources[name] = file_sha256(source / name)
    assets = Path(__file__).parent / 'release_assets'
    for src, dest in (('dataset-card.md', 'README.md'), ('CC-BY-4.0.txt', 'LICENSE')):
        shutil.copyfile(assets / src, output / dest)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    manifest = release_manifest(
        output, dataset='cloth-fling', version='v1', license='CC-BY-4.0',
        code_revision=revision,
        executed_code_revision=json.loads((source / 'runtime.json').read_text())['code_revision'],
        source_artifacts=sources, counts={s: {'recorded': a, 'eligible': b}
                                         for s, (a, b) in counts.items()},
        negative_eligible_examples={s: sum(r['eligible'] and r['coverage_gain'] < 0
                                         for r in examples) for s, examples in rows.items()},
        episodes=len(list((source / 'episodes').iterdir())),
    )
    fixture_bundles(output)
    for split in SPLITS:
        eligible_arrays(output, split)
    return manifest
