"""Explicit simulator-free transition schema. Changes require a new schema version."""

import pyarrow as pa

SCHEMA_VERSION = 1


def transition_schema() -> pa.Schema:
    required = {
        "schema_version": pa.int32(),
        "transition_id": pa.int64(),
        "run_id": pa.string(),
        "episode_id": pa.int64(),
        "step_id": pa.int64(),
        "split": pa.string(),
        "reset_group_id": pa.string(),
        "cloth_id": pa.string(),
        "reset_id": pa.string(),
        "reset_sha256": pa.string(),
        "config_id": pa.string(),
        "config_sha256": pa.string(),
        "seed": pa.int64(),
        "before_observation_id": pa.string(),
        "before_rgb_path": pa.string(),
        "before_rgb_sha256": pa.string(),
        "before_support_path": pa.string(),
        "before_support_sha256": pa.string(),
        "primitive": pa.string(),
        "policy": pa.string(),
        "execution_valid": pa.bool_(),
        "terminated": pa.bool_(),
        "truncated": pa.bool_(),
        "reason": pa.string(),
        "evidence_path": pa.string(),
        "evidence_sha256": pa.string(),
    }
    optional = {
        "parent_run_id": pa.string(),
        "parent_episode_id": pa.int64(),
        "after_observation_id": pa.string(),
        "after_rgb_path": pa.string(),
        "after_rgb_sha256": pa.string(),
        "after_support_path": pa.string(),
        "after_support_sha256": pa.string(),
        "final_state_path": pa.string(),
        "final_state_sha256": pa.string(),
        "end_reason": pa.string(),
    }
    optional.update(
        dict.fromkeys(
            [
                "action_u0",
                "action_v0",
                "action_u1",
                "action_v1",
                "coverage_before",
                "coverage_after",
                "coverage_gain",
                "reward",
                "area_before_m2",
                "area_after_m2",
                "area_gain_m2",
                "flat_area_m2",
                "simulated_seconds",
                "wall_seconds",
            ],
            pa.float64(),
        )
    )
    return pa.schema(
        [pa.field(k, v, nullable=False) for k, v in required.items()]
        + [pa.field(k, v) for k, v in optional.items()],
        metadata={b"cloth_fling.schema_version": b"1", b"learned_input": b"RGB only"},
    )
