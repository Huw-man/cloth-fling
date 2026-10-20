"""Offline loading of versioned diagonal actions or contact-only failure records."""

import json

import numpy as np

from cloth_fling.data.storage import file_sha256, relative_path
from cloth_fling.reference_tune import read_hashed_bundle


def load_diagonal_replay(root):
    from cloth_fling.replay import ReplayAction, validate_action

    def checked(ref):
        if file_sha256(relative_path(root, ref["path"])) != ref["sha256"]:
            raise ValueError("diagonal replay artifact hash mismatch")
        return read_hashed_bundle(root, ref["path"])

    contact = json.loads((root / "contact-report.json").read_text())
    report_path = root / "diagonal-report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    actions = []
    if report and report.get("attempts"):
        state = checked(contact["fixture"])
        for record in report["attempts"]:
            run = checked(record["outcome"])
            label = f"{record['category']} {record['attempt']}/c{record['candidate']} " + (
                "PASS" if record["passed"] else "FAIL"
            )
            if run.get("diagnostics", {}).get("execution_mode") == "demonstration":
                label = (
                    f"exploratory demonstration {record['attempt']}: {run['reason']} (unqualified)"
                )
            actions.append(
                validate_action(
                    ReplayAction(
                        label,
                        state,
                        run["evidence"]["phase_traces"],
                        run["simulated_seconds"],
                        run["execution_valid"],
                        run["reason"],
                        run,
                        {},
                        0.30,
                    )
                )
            )
    else:
        records = [*contact["cases"], *contact.get("failure_artifacts", [])]
        for record in records:
            run = checked(record["outcome"])
            state = run.get("initial_state", run.get("failure_state"))
            if state is None:
                continue
            times = np.asarray(run.get("times", []))
            p = np.asarray(run.get("positions", []))
            v = np.asarray(run.get("velocities", []))
            if not len(times) or p.ndim != 3:
                continue
            ids = np.flatnonzero(~np.asarray(state["free"])).tolist()
            indexes = (
                list(range(0, len(times), max(1, round(0.1 / (times[1] - times[0])))))
                if len(times) > 1
                else [0]
            )
            if indexes[-1] != len(times) - 1:
                indexes.append(len(times) - 1)
            passed = bool(run.get("summary", {}).get("passed", False))
            name = record.get("name", record.get("case"))
            traces = [
                {
                    "phase": name,
                    "simulated_seconds": float(times[i]),
                    "cloth_positions_m": p[i],
                    "cloth_velocities_m_s": v[i],
                    "attachment_ids": ids,
                }
                for i in indexes
            ]
            actions.append(
                validate_action(
                    ReplayAction(
                        f"contact {name} " + ("PASS" if passed else "FAIL"),
                        state,
                        traces,
                        float(times[-1]),
                        passed,
                        run.get(
                            "error", "contact gate passed" if passed else "contact gate failed"
                        ),
                        run,
                        {},
                        0.30,
                    )
                )
            )
    origin = {
        "format": "reference_diagonal",
        "source_contact_report_sha256": file_sha256(root / "contact-report.json"),
        "source_diagonal_report_sha256": file_sha256(report_path) if report_path.exists() else None,
        "motion_evaluated": bool(report and report.get("attempts")),
    }
    return actions, origin
