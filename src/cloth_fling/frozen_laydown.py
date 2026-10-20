"""User-frozen reference laydown, entered with two already-held native cloth pins."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from cloth_fling.contracts import GraspAction
from cloth_fling.reference_diagonal_motion import DiagonalTrajectory, execute_diagonal_action

VERSION = "reference_post_grasp_laydown_v3"
CONFIG_NAME = "reference-post-grasp-laydown-v3-frozen.json"
CONFIG_SHA256 = "9439a4c87f419711df9a01eb6f9bda24340575edc94ec8a35fef7aa0f6756647"


def frozen_laydown_spec():
    """Load the packaged immutable spec; no simulator, checkout or run directory needed."""
    data = files("cloth_fling").joinpath("profiles", CONFIG_NAME).read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIG_SHA256:
        raise ValueError("frozen laydown specification changed; use a new version")
    return json.loads(data)


def execute_frozen_laydown(env, action, *, cleanup_callback):
    """Adopt the live grasp without attach/reset/restore, then release and settle.

    `action` retains the grasp selection's provenance; its pixels are not reselected.
    Preconditions are checked before taking ownership. Once accepted, this action
    owns stepping and writes the native release/cleanup certificate through the
    required callback, including when execution fails.
    """
    if not isinstance(action, GraspAction) or not callable(cleanup_callback):
        raise TypeError("handoff requires the original GraspAction and a cleanup callback")
    spec = frozen_laydown_spec()
    if json.loads(json.dumps(env.config.to_dict())) != spec["scene"]:
        raise ValueError("frozen laydown requires its reference physical configuration")
    if getattr(env, "diagnostic_field", None) is not None:
        raise ValueError("diagnostic external loads cannot enter frozen laydown")
    for relative, expected in spec["implementation_sha256"].items():
        path = Path(__file__).parent / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen laydown implementation changed: {relative}")
    ids = tuple(env.attached_ids)
    started_at = float(env.simulated_time)
    result = execute_diagonal_action(
        env,
        action,
        DiagonalTrajectory(**spec["trajectory"]),
        expected_ids=ids,
        cleanup_callback=cleanup_callback,
        execution_mode=spec["execution_mode"],
        retain_existing_grasp=True,
    )
    result.diagnostics["frozen_laydown_version"] = VERSION
    result.evidence["frozen_laydown"] = {
        "version": VERSION,
        "specification_sha256": CONFIG_SHA256,
        "incoming_native_pin_ids": list(ids),
        "incoming_simulated_time": started_at,
        "selection_basis": spec["selection_basis"],
        "reference_evidence": spec["reference_evidence"],
    }
    return result
