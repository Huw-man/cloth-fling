import numpy as np
import pytest

from cloth_fling.sim.coverage import CoverageGrid
from cloth_fling.sim.mesh import square_mesh


def test_flat_reference_and_exact_overlap_counted_once():
    p, f = square_mesh(0.5, 0.05)
    grid = CoverageGrid()
    assert grid.measure(p, f, p) == pytest.approx((1, 0.25, 0.25))
    doubled = np.concatenate((p, p + [0, 0, 0.02]))
    double_faces = np.concatenate((f, f + len(p)))
    assert grid.area(doubled, double_faces) == pytest.approx(0.25)


def test_concavity_hole_and_disconnected_regions():
    # Four small squares form a frame: neither its hole nor convex hull corners
    # may be filled by triangle union.
    p, f = square_mesh(0.2, 0.1)
    positions = np.concatenate(
        [p + shift for shift in [[-0.2, 0, 0], [0.2, 0, 0], [0, -0.2, 0], [0, 0.2, 0]]]
    )
    faces = np.concatenate([f + len(p) * i for i in range(4)])
    grid = CoverageGrid(1, 0.01)
    assert grid.area(positions, faces) == pytest.approx(0.16)
    assert not grid.mask(positions, faces)[50, 50]


def test_half_fold_area_and_negative_gain():
    p, f = square_mesh(0.5, 0.01)
    folded = p.copy()
    folded[:, 0] = -np.abs(folded[:, 0])
    grid = CoverageGrid()
    before = grid.measure(p, f, p)[0]
    after, area, flat = grid.measure(folded, f, p)
    assert (after, area, flat) == pytest.approx((0.5, 0.125, 0.25))
    assert after - before == pytest.approx(-0.5)


def test_degenerate_and_outside_triangles():
    p, f = square_mesh(0.5, 0.05)
    grid = CoverageGrid(0.6, 0.01)
    assert grid.area(p + [4, 4, 0], f) == 0
    p[:, 0] = 0
    assert grid.area(p, f) == 0


def test_area_can_exceed_flat_reference():
    p, f = square_mesh(0.5, 0.1)
    assert CoverageGrid().measure(p * 1.2, f, p)[0] == pytest.approx(1.44)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected(bad):
    p, f = square_mesh(0.5, 0.1)
    p[0, 0] = bad
    with pytest.raises(ValueError):
        CoverageGrid().area(p, f)
