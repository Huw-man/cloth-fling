"""Reproducible original-pilot export. Run from an installed source checkout."""
import argparse
import json

from cloth_fling.release_data import export_v1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
print(json.dumps(export_v1(args.source, args.output), indent=2))
