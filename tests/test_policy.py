import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_fling import policy, release_data
from cloth_fling.cli import build_parser
from cloth_fling.data.storage import file_sha256


def example(split='train', target=-.25):
    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 123, np.uint8)).save(buffer, format='PNG')
    encoding = {'codec': 'reference_spatial_v1', 'map_index_tvu': [0, 32, 32],
                'to_original': np.eye(3).tolist()}
    return {'split': split, 'eligible': target is not None,
            'execution_valid': True, 'settled': target is not None,
            'executed_feasible': True, 'coverage_gain': -.25,
            'supervised_target': target, 'action_encoding_json': json.dumps(encoding),
            'map_index_tvu': [0, 32, 32], 'to_original': np.eye(3).tolist(),
            'before_rgb': {'bytes': buffer.getvalue(), 'path': None},
            'after_rgb': {'bytes': buffer.getvalue(), 'path': None}}


def release(tmp_path, rows):
    for split in release_data.SPLITS:
        examples = [r for r in rows if r['split'] == split]
        if examples:
            release_data.write_split(tmp_path, split, examples)
    release_data.release_manifest(tmp_path, version='test')
    return tmp_path


def test_saved_projection_filters_unsettled_and_keeps_negative_targets(tmp_path):
    root = release(tmp_path, [example(), example(target=None)])
    images, pixels, targets, _ = release_data.eligible_arrays(root)
    assert len(images) == 1 and (images == 123).all()
    assert pixels.tolist() == [[32, 32]] and targets.tolist() == [-.25]
    assert release_data.read_split(root, 'train')[1]['supervised_target'] is None
    (root / 'data/train.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corruption'):
        release_data.read_split(root, 'train')


def test_split_and_label_validation(tmp_path):
    with pytest.raises(ValueError, match='split mismatch'):
        release_data.write_split(tmp_path, 'test', [example()])
    row = example(target=None)
    row['supervised_target'] = 0
    root = release(tmp_path, [row])
    with pytest.raises(ValueError, match='ineligible'):
        release_data.eligible_arrays(root)


def test_validation_default_and_explicit_test_and_paths():
    parser = build_parser()
    base = ['policy', 'evaluate', '--checkpoint', 'model.pt', '--output', 'out']
    assert parser.parse_args(base).split == 'validation'
    assert parser.parse_args(base + ['--split', 'test']).split == 'test'
    assert parser.parse_args(['policy', 'collect', '--fixtures', 'f', '--output', 'o']).max_actions == 3
    with pytest.raises(SystemExit):
        parser.parse_args(['policy', 'train', '--output', 'o'])
    with pytest.raises(SystemExit):
        parser.parse_args(base + ['--max-actions', '4'])


def test_custom_predictor_rejects_bad_shape_and_nonfinite(monkeypatch):
    module = SimpleNamespace(predict=lambda rgb: np.zeros((96, 64, 64)))
    monkeypatch.setattr(policy.importlib, 'import_module', lambda _: module)
    predict = policy.custom_predictor('mine:predict')
    rgb = np.zeros((96, 64, 64, 3), np.uint8)
    assert predict(rgb).shape == (96, 64, 64)
    module.predict = lambda rgb: np.zeros(3)
    with pytest.raises(ValueError, match='finite values'):
        policy.custom_predictor('mine:predict')(rgb)
    module.predict = lambda rgb: np.full((96, 64, 64), np.nan)
    with pytest.raises(ValueError, match='finite values'):
        policy.custom_predictor('mine:predict')(rgb)


def test_simulation_metrics_settled_contained_thresholds():
    def action(coverage, settled=True, containment=1.0):
        return {'execution_valid': True, 'coverage_after': coverage, 'reason': 'completed',
                'diagnostics': {'settled': settled, 'final_table_containment': containment}}
    def result(actions, failed=False):
        return {'episode': 'e', 'actions': actions, 'failed': failed, 'fatal': None,
                'collection': {'episode_reports': [{'initial_coverage': .5}]}}
    metrics = policy.simulation_metrics([
        result([action(.6), action(.81)]),
        result([action(.9, settled=False)], failed=True),
        result([action(.9, containment=.99), action(.7), action(.85)]),
    ])
    assert [metrics['success_within_actions'][str(k)]['count'] for k in (1, 2, 3)] == [0, 1, 2]
    assert metrics['total_actions_used'] == 6 and metrics['failure_count'] == 1


def test_train_reload_offline_without_genesis(tmp_path):
    pytest.importorskip('torch')
    root = release(tmp_path / 'data', [example() for _ in range(32)] + [example('validation')])
    output = tmp_path / 'fit'
    args = build_parser().parse_args(['policy', 'train', '--dataset', str(root),
                                     '--output', str(output), '--steps', '1'])
    trained = policy.run(args)
    assert trained['eligible_samples'] == 32
    report = policy.offline(root, checkpoint=output / 'model.pt')
    assert report['split'] == 'validation' and report['eligible_examples'] == 1
    assert np.isfinite([report[k] for k in ('mae', 'huber_loss', 'prediction_bias')]).all()
    with pytest.raises(FileExistsError):
        policy.run(args)
    assert trained['weights_sha256'] == file_sha256(output / 'model.pt')


def test_imports_leave_torch_and_genesis_unloaded():
    subprocess.run([sys.executable, '-c',
                    "import sys; import cloth_fling.cli, cloth_fling.release_data; "
                    "assert 'torch' not in sys.modules; assert 'genesis' not in sys.modules"],
                   check=True)


def test_export_rejects_unrelated_campaign_before_creating_output(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    release_data.write_json(source / 'protocol.json', {'version': 'single_reset_overfit_v1'})
    release_data.write_json(source / 'ledger.json', {'status': 'complete'})
    with pytest.raises(ValueError, match='original'):
        release_data.export_v1(source, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_fixture_hash_failure_stops_before_backend(tmp_path):
    root = release(tmp_path / 'data', [example()])
    (root / 'data/train.parquet').write_bytes(b'changed')
    args = build_parser().parse_args(['policy', 'collect', '--fixtures', str(root),
                                     '--output', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='corruption'):
        policy.simulate(args, backend_factory=lambda *a: pytest.fail('opened backend'))
    assert not (tmp_path / 'out').exists()


def test_split_isolation_uses_reset_families_and_episode_groups():
    train = {'split': 'train', 'reset_group_id': 'fold-20', 'reset_id': 'a', 'episode_key': 'e1'}
    test = {'split': 'test', 'reset_group_id': 'fold-40', 'reset_id': 'b', 'episode_key': 'e2'}
    release_data.check_split_isolation({'train': [train], 'test': [test]})
    for key in ('reset_group_id', 'reset_id', 'episode_key'):
        with pytest.raises(ValueError, match='leakage'):
            release_data.check_split_isolation({'train': [train], 'test': [{**test, key: train[key]}]})


def test_fixture_array_hash_validation(tmp_path):
    import hashlib

    fixture = tmp_path / 'fixtures/fold-20/fixture.json'
    fixture.parent.mkdir(parents=True)
    arrays = fixture.with_suffix('.npz')
    arrays.write_bytes(b'changed')
    release_data.write_json(fixture, {'arrays': {'path': 'fixture.npz',
                                               'sha256': hashlib.sha256(b'original').hexdigest()}})
    release_data.write_json(tmp_path / 'fixtures-manifest.json', {
        'config_sha256': release_data.CONFIG_SHA256,
        'cases': [{'id': 'fold-20', 'split': 'train', 'state_hash': 'h', 'accepted': True,
                   'cleanup': {'passed': True}, 'fixture': {
                       'path': 'fixtures/fold-20/fixture.json', 'sha256': file_sha256(fixture)}}],
    })
    release_data.release_manifest(tmp_path, version='test')
    with pytest.raises(ValueError, match='array corruption'):
        release_data.fixture_bundles(tmp_path, 'train')


def test_project_rgb_action_label_parity_and_null_ineligible(tmp_path, monkeypatch):
    from cloth_fling.data.schema import transition_schema

    row = dict.fromkeys(transition_schema().names)
    image = example()['before_rgb']['bytes']
    (tmp_path / 'rgb.png').write_bytes(image)
    row.update(split='train', before_rgb_path='rgb.png', after_rgb_path='rgb.png',
               execution_valid=True, reward=-.25, coverage_gain=-.25,
               evidence_path='evidence', action_u0=12., action_v0=13.,
               action_u1=24., action_v1=25.)
    selection = {'encoding': json.loads(example()['action_encoding_json']),
                 'valid_action_mask': np.ones((96, 64, 64), dtype=bool)}
    support = {'diagnostics': {'settled': True}, 'evidence': {'selection': selection}}
    monkeypatch.setattr(release_data, 'verify_dataset', lambda _: {'valid': True})
    monkeypatch.setattr(release_data, 'read_support', lambda *a: support)
    monkeypatch.setattr(release_data, 'read_transitions',
                        lambda _: SimpleNamespace(to_pylist=lambda: [row]))
    projected = release_data.project_episode(tmp_path, 'episode')[0]
    assert projected['supervised_target'] == -.25
    assert projected['before_rgb']['bytes'] == projected['after_rgb']['bytes'] == image
    assert projected['map_index_tvu'] == selection['encoding']['map_index_tvu']
    assert projected['to_original'] == selection['encoding']['to_original']
    assert projected['action_u0'] == 12. and projected['action_v1'] == 25.
    support['diagnostics']['settled'] = False
    projected = release_data.project_episode(tmp_path, 'episode')[0]
    assert not projected['eligible'] and projected['supervised_target'] is None
    assert projected['coverage_gain'] == -.25 and projected['failed']


def test_busy_lease_prevents_checkpoint_gpu_allocation(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from cloth_fling import reference_run

    root = tmp_path / 'data'
    root.mkdir()
    release_data.write_json(root / 'fixtures-manifest.json', {})
    release_data.write_json(root / 'release-manifest.json', {})
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(policy, 'fixture_bundles', lambda *a: [({'id': 'f'}, {})])
    monkeypatch.setattr(policy.learning, 'predictor', lambda *a, **k: pytest.fail('allocated model'))

    @contextmanager
    def busy(_):
        raise RuntimeError('lease busy')
        yield

    monkeypatch.setattr(reference_run, 'reference_lease', busy)
    args = build_parser().parse_args([
        'policy', 'evaluate', '--mode', 'sim', '--fixtures', str(root),
        '--checkpoint', str(checkpoint), '--output', str(tmp_path / 'out'), '--device', 'cuda',
    ])
    with pytest.raises(RuntimeError, match='lease busy'):
        policy.simulate(args, backend_factory=lambda *a: pytest.fail('opened scene'))
    report = json.loads((tmp_path / 'out/report.json').read_text())
    assert report['status'] == 'failed' and report['action_intents'] == 0


def test_failed_unfinalized_action_is_counted_and_scene_cleaned(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from cloth_fling import learning_pilot, reference_pilot, reference_run

    root = tmp_path / 'data'
    root.mkdir()
    release_data.write_json(root / 'fixtures-manifest.json', {})
    release_data.write_json(root / 'release-manifest.json', {})
    monkeypatch.setattr(policy, 'fixture_bundles', lambda *a: [({'id': 'f'}, {})])
    monkeypatch.setattr(reference_run, 'reference_lease', lambda _: nullcontext())
    monkeypatch.setattr(reference_pilot, 'certify_release', lambda _: {'passed': True})
    closed = []

    def failure(env, journal, *a, **kw):
        journal.claim_action('failed-episode', 0)
        raise RuntimeError('physical stop')

    monkeypatch.setattr(learning_pilot, 'episode', failure)
    args = build_parser().parse_args([
        'policy', 'collect', '--fixtures', str(root), '--output', str(tmp_path / 'out'),
    ])
    with pytest.raises(RuntimeError, match='physical stop'):
        policy.simulate(args, backend_factory=lambda _: SimpleNamespace(close=lambda: closed.append(1)))
    report = json.loads((tmp_path / 'out/report.json').read_text())
    assert report['total_actions_used'] == report['action_intents'] == 1
    assert report['actions_used_by_episode'] == {'failed-episode': 1}
    assert report['failure_count'] == 1 and closed == [1]
    assert json.loads((tmp_path / 'out/scene-cleanup.json').read_text())['passed']
