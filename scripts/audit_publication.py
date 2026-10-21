"""Inspect reachable history and allowlisted release files without printing secrets."""
import argparse
import json
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--release', type=Path)
args = parser.parse_args()
patterns = {
    'private-key': rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
    'github-token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'hf-token': rb'\bhf_[A-Za-z0-9]{30,}\b',
    'aws-key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'credential-url': rb'https?://[^\s/@:]+:[^\s/@]+@',
    'generic-secret': rb'(?i)(?:api_key|access_token|client_secret|password)\s*[:=]\s*[\x22\x27][A-Za-z0-9/+_=.-]{20,}[\x22\x27]',
}
patterns = {k: re.compile(v) for k, v in patterns.items()}
objects = subprocess.check_output(['git', 'rev-list', '--objects', 'HEAD']).decode().splitlines()
objects = [line.split(' ', 1) for line in objects]
findings, artifacts = [], []
process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE)
for entry in objects:
    oid, path = entry[0], entry[1] if len(entry) > 1 else '(commit/tree)'
    process.stdin.write((oid + '\n').encode())
    process.stdin.flush()
    header = process.stdout.readline().decode().split()
    content = process.stdout.read(int(header[2]))
    process.stdout.read(1)
    if header[1] != 'blob':
        continue
    for kind, pattern in patterns.items():
        if pattern.search(content):
            findings.append({'object': oid, 'path': path, 'kind': kind})
    if (path.startswith(('runs/', '.venv/', '.env', '.cache/'))
            or path.endswith(('.pt', '.pth', '.npz', '.mp4', '.pem', '.key'))
            or len(content) > 10_000_000):
        artifacts.append({'object': oid, 'path': path, 'size_bytes': len(content)})
process.stdin.close()
process.wait()
if args.release:
    from cloth_fling.release_data import verify_release
    release = verify_release(args.release)
    allowed = re.compile(r'(?:README\.md|LICENSE|fixtures-manifest\.json|'
                         r'data/(?:train|validation|test)\.parquet|'
                         r'fixtures/fold-[^/]+/fixture\.(?:json|npz))\Z')
    for name in release['files']:
        if not allowed.fullmatch(name):
            artifacts.append({'path': name, 'kind': 'unexpected release file'})
        content = (args.release / name).read_bytes()
        for kind, pattern in patterns.items():
            if pattern.search(content):
                findings.append({'path': name, 'kind': kind})
print(json.dumps({'history_objects': len(objects), 'secret_findings': findings,
                  'unintended_artifacts': artifacts}, indent=2))
raise SystemExit(bool(findings or artifacts))
