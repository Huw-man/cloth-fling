import json
import subprocess
import sys

import numpy as np
import pytest
from test_reference import observation

from cloth_fling.learning import transformed_rgb
from cloth_fling.spatial_actions import build_action_batch


def test_saved_transform_reconstructs_exact_execution_rgb():
    from cloth_fling.contracts import ReferenceSceneConfig

    obs = observation()
    batch = build_action_batch(obs, ReferenceSceneConfig())
    for i in (0, 7, 32, 95):
        np.testing.assert_array_equal(transformed_rgb(obs.rgb, batch.to_original[i]), batch.rgb[i])


def test_learning_and_pilot_import_without_torch_or_genesis():
    command = "import sys; import cloth_fling.learning, cloth_fling.learning_pilot; assert 'torch' not in sys.modules; assert 'genesis' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)


def test_executed_pixel_loss_keeps_negative_labels_and_has_sparse_gradient():
    torch = pytest.importorskip("torch")
    from cloth_fling.learning import executed_loss

    predictions = torch.zeros(2, 64, 64, requires_grad=True)
    loss = executed_loss(predictions, torch.tensor([[5, 6], [8, 9]]), torch.tensor([-0.5, 0.25]))
    loss.backward()
    assert torch.count_nonzero(predictions.grad) == 2
    assert predictions.grad[0, 5, 6] > 0
    assert predictions.grad[1, 8, 9] < 0


def test_tiny_batch_overfit_and_reload(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from cloth_fling import learning

    learning.seed_training()
    rng = np.random.default_rng(9)
    images = rng.integers(0, 256, (32, 64, 64, 3), dtype=np.uint8)
    images[:] = images[0]
    pixels = np.tile([32, 32], (32, 1))
    rewards = np.full(32, -0.2, dtype=np.float32)
    monkeypatch.setattr(learning, "training_arrays", lambda _: (images, pixels, rewards, []))
    metadata = learning.fit([], tmp_path / "tiny.pt", device="cpu", steps=70)
    assert metadata["losses"][-1]["loss"] < 0.003
    predict = learning.predictor(tmp_path / "tiny.pt", device="cpu")
    values = predict(np.tile(images[:1], (96, 1, 1, 1)))
    assert values.shape == (96, 64, 64)
    assert values[0, 32, 32] == pytest.approx(-0.2, abs=0.05)
    data = json.loads((tmp_path / "tiny.json").read_text())
    data["frozen_config_sha256"] = "wrong"
    (tmp_path / "tiny.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible"):
        learning.load_model(tmp_path / "tiny.pt")


def test_training_rejects_validation_and_replays_saved_crop(monkeypatch, tmp_path):
    from cloth_fling import learning

    (tmp_path / "manifest.json").write_text("{}")
    sample = {
        "rgb": np.full((64, 64, 3), 50, dtype=np.uint8),
        "split": "validation",
        "reward": -0.25,
        "valid_action_mask": np.ones((96, 64, 64), bool),
        "action": {
            "codec": "reference_spatial_v1",
            "map_index_tvu": [4, 9, 7],
            "to_original": np.eye(3).tolist(),
        },
    }
    monkeypatch.setattr(learning, "learning_samples", lambda _: iter([sample]))
    with pytest.raises(ValueError, match="non-training"):
        learning.training_arrays([tmp_path])
    sample["split"] = "train"
    images, pixels, rewards, _ = learning.training_arrays([tmp_path])
    assert (images == 50).all()
    assert pixels.tolist() == [[9, 7]] and rewards.tolist() == [-0.25]
    sample["valid_action_mask"][4, 9, 7] = False
    with pytest.raises(ValueError, match="feasible"):
        learning.training_arrays([tmp_path])
