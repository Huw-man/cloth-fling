"""Local immutable artifacts and atomic manifest replacement; no simulator imports."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np


def relative_path(root: Path, value: str) -> Path:
    """Resolve a dataset reference, rejecting traversal and symlink escapes."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid relative dataset path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..") for p in value.split("/")):
        raise ValueError(f"invalid relative dataset path: {value!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"dataset path escapes root: {value!r}")
    return target


def jsonable(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported support value: {type(value).__name__}")


def json_bytes(value) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reset_family_hash(*, config_sha256, cloth_id, seed, reset_kind, reset_source):
    identity = {"config_sha256": config_sha256, "cloth_id": cloth_id}
    if reset_source is not None:
        identity["saved_input_sha256"] = sha256(json_bytes(reset_source))
    else:
        identity.update(generator_kind=reset_kind, generator_seed=int(seed))
    return sha256(json_bytes(identity))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(root: Path, reference: str, data: bytes) -> None:
    path = relative_path(root, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def load_json(root: Path, reference: str):
    return json.loads(relative_path(root, reference).read_text())


def write_bundle(put, payload, prefix: str) -> dict:
    """Store arrays in safe NPZ and nested structure in JSON, without pickle."""
    arrays = {}

    def encode(value):
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays are not supported")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"__npz_array__": key}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        return jsonable(value)

    document = {"payload": encode(payload), "arrays": None}
    if arrays:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        document["arrays"] = put(f"{prefix}.npz", buf.getvalue())
    return put(f"{prefix}.json", json_bytes(document))


def read_bundle(root: Path, reference: str):
    document = load_json(root, reference)
    arrays = {}
    if document["arrays"]:
        with np.load(relative_path(root, document["arrays"]["path"]), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"__npz_array__"}:
                return arrays[value["__npz_array__"]]
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(document["payload"])
