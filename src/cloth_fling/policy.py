"""Small saved-data -> RGB value model -> frozen simulation workflow."""

import importlib
import json
import time
from pathlib import Path

import numpy as np

from cloth_fling import learning
from cloth_fling.data.storage import file_sha256
from cloth_fling.release_data import (
    decode_rgb,
    eligible_arrays,
    fixture_bundles,
    project_episode,
    read_split,
    release_manifest,
    write_json,
    write_split,
)


def add_parser(sub):
    parser = sub.add_parser('policy', help='collect, train and evaluate RGB grasp values')
    commands = parser.add_subparsers(dest='policy_command', required=True)
    collect = commands.add_parser('collect', help='collect examples from training fixtures')
    collect.add_argument('--fixtures', type=Path, required=True)
    collect.add_argument('--output', type=Path, required=True)
    collect.add_argument('--episodes', type=int, default=1)
    _simulation_arguments(collect)
    train = commands.add_parser('train', help='fit the existing model from saved train examples')
    train.add_argument('--dataset', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--steps', type=int, default=2000)
    train.add_argument('--device', default='cpu')
    evaluate = commands.add_parser('evaluate', help='evaluate saved predictions or native actions')
    evaluate.add_argument('--mode', choices=('offline', 'sim'), default='offline')
    evaluate.add_argument('--dataset', type=Path)
    evaluate.add_argument('--fixtures', type=Path)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--split', choices=('validation', 'test'), default='validation')
    models = evaluate.add_mutually_exclusive_group(required=True)
    models.add_argument('--checkpoint', type=Path)
    models.add_argument('--predictor', help='Python module:function, uint8 RGB -> value maps')
    evaluate.add_argument('--device', default='cpu')
    _simulation_arguments(evaluate)


def _simulation_arguments(parser):
    parser.add_argument('--max-actions', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--max-wall-seconds', type=float, default=3600)
    parser.add_argument('--fixture-id', help='optional fixture ID within the selected split')
    parser.add_argument('--seed', type=int, default=100)


def custom_predictor(spec):
    module, separator, function = spec.partition(':')
    if not separator or not module or not function:
        raise ValueError('predictor must be module:function')
    predict = getattr(importlib.import_module(module), function)
    if not callable(predict):
        raise ValueError('predictor is not callable')

    def checked(rgb):
        if rgb.shape != (96, 64, 64, 3) or rgb.dtype != np.uint8:
            raise ValueError('predictor requires uint8 RGB [96,64,64,3]')
        values = np.asarray(predict(rgb))
        if values.shape != (96, 64, 64) or not np.isfinite(values).all():
            raise ValueError('predictor must return finite values [96,64,64]')
        return values

    return checked


def rgb_batch(row):
    """Rebuild RGB transforms from the saved RGB crop, without depth/feasibility."""
    from cloth_fling.spatial_actions import ANGLES_DEG, SCALES

    encoding = json.loads(row['action_encoding_json'])
    crop = encoding['crop']
    rgb = decode_rgb(row['before_rgb'])
    images = []
    for angle in ANGLES_DEG:
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        for scale in SCALES:
            matrix = np.eye(3)
            matrix[:2, :2] = rotation * crop['extent_px'] / (64 * scale)
            matrix[:2, 2] = crop['center_uv'] - matrix[:2, :2] @ np.array([31.5, 31.5])
            images.append(learning.transformed_rgb(rgb, matrix))
    t = row['map_index_tvu'][0]
    if not np.array_equal(images[t], learning.transformed_rgb(rgb, row['to_original'])):
        raise ValueError('saved and reconstructed RGB transform differ')
    return np.stack(images)


def offline(dataset, *, checkpoint=None, predictor=None, split='validation', device='cpu'):
    images, pixels, targets, _ = eligible_arrays(dataset, split)
    if predictor:
        predict = custom_predictor(predictor)
        rows = [r for r in read_split(dataset, split) if r['eligible']]
        values = np.asarray([predict(rgb_batch(r))[tuple(r['map_index_tvu'])] for r in rows])
    else:
        import torch

        learning.seed_training()
        model, _ = learning.load_model(checkpoint, device=device)
        values = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                x = torch.as_tensor(images[start:start + 16], device=device)
                predictions = model(x.permute(0, 3, 1, 2).float() / 255).cpu().numpy()
                uv = pixels[start:start + 16]
                values.extend(predictions[np.arange(len(uv)), uv[:, 0], uv[:, 1]])
        values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite predictions')
    error = values.astype(float) - targets.astype(float)
    absolute = np.abs(error)
    return {'mode': 'offline', 'split': split, 'eligible_examples': len(targets),
            'mae': float(absolute.mean()),
            'huber_loss': float(np.where(absolute <= .1, .5 * error**2,
                                         .1 * (absolute - .05)).mean()),
            'huber_delta': .1, 'prediction_bias': float(error.mean()),
            'dataset_manifest_sha256': file_sha256(Path(dataset) / 'release-manifest.json'),
            'checkpoint_sha256': file_sha256(checkpoint) if checkpoint else None,
            'predictor': predictor}


class SimulationJournal:
    """Fresh one-shot execution with durable action intents and no uncertain retries."""
    def __init__(self, root, seconds, action_limit, stage):
        self.root, self.seconds, self.action_limit, self.stage = root, seconds, action_limit, stage
        self.started = time.monotonic()
        self.actions = []
        self.save('started')

    def remaining(self):
        remaining = self.seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise RuntimeError('simulation wall deadline exhausted')
        return remaining

    def claim_action(self, episode, index):
        self.remaining()
        if len(self.actions) >= self.action_limit:
            raise RuntimeError('action budget exhausted')
        self.actions.append({'episode': episode, 'index': index, 'status': 'intent'})
        self.save('running')

    def save(self, status, error=None):
        write_json(self.root / 'ledger.json', {
            'status': status, 'actions': self.actions, 'error': error,
            'wall_seconds': time.monotonic() - self.started,
            'limit_wall_seconds': self.seconds, 'action_limit': self.action_limit,
        })


def simulation_metrics(results):
    successes = {k: 0 for k in (1, 2, 3)}
    terminal, used, failures = [], [], []
    for result in results:
        actions = result['actions']
        initial = result['collection']['episode_reports'][0]['initial_coverage']
        hit = 0 if not actions and initial >= .8 and not result['failed'] else None
        for i, action in enumerate(actions, 1):
            d = action['diagnostics']
            if (action['execution_valid'] and d.get('settled') is True
                    and d.get('final_table_containment') == 1.0
                    and action['coverage_after'] >= .8):
                hit = i if hit is None else hit
        for k in successes:
            successes[k] += int(hit is not None and hit <= k)
        terminal.append(actions[-1]['coverage_after'] if actions else initial)
        used.append(len(actions))
        if result['failed'] or result.get('fatal'):
            failures.append({'episode': result['episode'], 'fatal': result.get('fatal'),
                             'reasons': [a['reason'] for a in actions]})
    n = len(results)
    return {'episodes': n, 'success_within_actions': {
                str(k): {'count': count, 'rate': count / n if n else None}
                for k, count in successes.items()},
            'terminal_coverage': terminal,
            'mean_terminal_coverage': float(np.mean(terminal)) if n else None,
            'failures': failures, 'failure_count': len(failures),
            'actions_used': used, 'total_actions_used': sum(used)}


def simulate(args, *, backend_factory=None):
    from cloth_fling.benchmark_run import _reservation_deadline
    from cloth_fling.contracts import scene_config_from_dict
    from cloth_fling.frozen_laydown import frozen_laydown_spec
    from cloth_fling.learning_pilot import episode
    from cloth_fling.reference_pilot import certify_release
    from cloth_fling.reference_run import reference_lease

    if args.output.exists():
        raise FileExistsError('output directory must be fresh')
    collect = args.policy_command == 'collect'
    split = 'train' if collect else args.split
    cases = fixture_bundles(args.fixtures, split)
    if args.fixture_id:
        cases = [(c, s) for c, s in cases if c['id'] == args.fixture_id]
        if not cases:
            raise ValueError('fixture ID absent from selected split')
    if collect:
        if args.episodes < 1:
            raise ValueError('episodes must be positive')
        cases = [cases[i % len(cases)] for i in range(args.episodes)]
    if not np.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 3600:
        raise ValueError('simulation wall bound must be in (0, 3600] seconds')
    spec = frozen_laydown_spec()
    for ref, expected in spec['implementation_sha256'].items():
        if file_sha256(Path(__file__).parent / ref) != expected:
            raise ValueError(f'frozen implementation changed: {ref}')
    predict = None if collect else (custom_predictor(args.predictor) if args.predictor
                                    else learning.predictor(args.checkpoint, device=args.device))
    args.output.mkdir(parents=True, exist_ok=False)
    from cloth_fling.learning_pilot import implementation_identity
    from cloth_fling.provenance import runtime_provenance

    write_json(args.output / 'implementation.json', implementation_identity())
    write_json(args.output / 'runtime.json', runtime_provenance())
    write_json(args.output / 'fixtures-manifest.json', json.loads(
        (args.fixtures / 'fixtures-manifest.json').read_text()))
    write_json(args.output / 'intent.json', {
        'command': args.policy_command, 'split': split, 'max_actions': args.max_actions,
        'coverage_stop': .8, 'require_containment': True,
        'fixture_ids': [c['id'] for c, _ in cases],
        'source_manifest_sha256': file_sha256(args.fixtures / 'release-manifest.json'),
        'checkpoint_sha256': file_sha256(args.checkpoint) if not collect and args.checkpoint else None,
        'predictor': None if collect else args.predictor, 'seed': args.seed,
        'frozen_config_sha256': learning.CONFIG_SHA256,
    })
    journal = SimulationJournal(args.output, args.max_wall_seconds,
                                len(cases) * args.max_actions, 'policy')
    results, env = [], None
    try:
        with reference_lease(args.output):
            try:
                if backend_factory is None:
                    from cloth_fling.sim.reference import ReferenceBackend
                    backend_factory = ReferenceBackend
                with _reservation_deadline(time.monotonic() + min(180, journal.remaining())):
                    env = backend_factory(scene_config_from_dict(spec['scene']))
                for i, (case, state) in enumerate(cases):
                    results.append(episode(
                        env, journal, case, state, f'{i:04d}-{case["id"]}',
                        predict=predict, epsilon=1.0 if collect else 0.0, seed=args.seed + i,
                        coverage_stop=.8, require_containment=True, reserve_seconds=0,
                        max_actions=args.max_actions,
                    ))
            finally:
                if env is not None:
                    cleanup = certify_release(env)
                    write_json(args.output / 'scene-cleanup.json', cleanup)
                    env.close()
                    if not cleanup['passed']:
                        raise RuntimeError('native scene cleanup regression')
        report = {'mode': 'sim', 'split': split, **simulation_metrics(results)}
        if collect:
            portable = args.output / 'examples'
            portable.mkdir(exist_ok=False)
            examples = []
            for result in results:
                examples.extend(project_episode(args.output / 'episodes' / result['episode']
                                                / 'dataset', result['episode']))
            if examples:
                write_split(portable, 'train', examples)
                release_manifest(portable, version='collected', counts={
                    'train': {'recorded': len(examples),
                              'eligible': sum(r['eligible'] for r in examples)}})
            report['recorded_examples'] = len(examples)
        journal.save('complete')
        write_json(args.output / 'report.json', report)
        return report
    except BaseException as exc:
        journal.save('failed', f'{type(exc).__name__}: {exc}')
        saved = {r['episode'] for r in results}
        for path in sorted((args.output / 'episodes').glob('*/result.json')):
            result = json.loads(path.read_text())
            if result['episode'] not in saved:
                results.append(result)
        metrics = simulation_metrics(results)
        if not metrics['failures']:
            metrics['failures'].append({'error': str(exc)})
            metrics['failure_count'] = 1
        write_json(args.output / 'report.json', {
            'mode': 'sim', 'split': split, 'status': 'failed', 'error': str(exc),
            **metrics, 'action_intents': len(journal.actions),
        })
        raise


def run(args):
    if args.policy_command == 'train':
        arrays = eligible_arrays(args.dataset, 'train')
        if args.steps < 1:
            raise ValueError('steps must be positive')
        args.output.mkdir(parents=True, exist_ok=False)
        report = learning.fit_arrays(arrays, args.output / 'model.pt',
                                      device=args.device, steps=args.steps)
    elif args.policy_command == 'evaluate' and args.mode == 'offline':
        if args.dataset is None or args.fixtures is not None:
            raise ValueError('offline evaluation requires --dataset and no --fixtures')
        if args.output.exists():
            raise FileExistsError('output directory must be fresh')
        report = offline(args.dataset, checkpoint=args.checkpoint, predictor=args.predictor,
                         split=args.split, device=args.device)
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        if args.fixtures is None:
            raise ValueError('simulation requires explicit --fixtures')
        if getattr(args, 'dataset', None) is not None:
            raise ValueError('simulation accepts --fixtures, not --dataset')
        return simulate(args)
    write_json(args.output / 'report.json', report)
    return report
