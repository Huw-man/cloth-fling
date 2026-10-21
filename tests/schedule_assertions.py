"""Frozen schedules: exact structure/timing, roundoff-tolerant compiled polynomials."""

import numpy as np


def assert_schedule_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == 'segments':
            assert len(actual[key]) == len(expected[key])
            for left, right in zip(actual[key], expected[key], strict=True):
                assert left.keys() == right.keys()
                for field in left:
                    if field == 'coefficients':
                        np.testing.assert_allclose(left[field], right[field], atol=1e-12, rtol=0)
                    else:
                        assert left[field] == right[field]
        elif key in ('peak_speed_m_s', 'peak_acceleration_m_s2'):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0)
        else:
            assert actual[key] == expected[key]
