import numpy as np

from caudal.scores import crps_ensemble


def test_one_member_is_absolute_error_and_spread_lowers_crps():
    obs = np.array([10.0, 20.0])
    assert np.allclose(crps_ensemble(np.array([[12.0], [15.0]]), obs), [2.0, 5.0])
    wide = np.array([[8.0, 12.0]]); point = np.array([[10.0, 10.0]])
    assert crps_ensemble(wide, obs[:1])[0] < crps_ensemble(point + 2, obs[:1])[0]
