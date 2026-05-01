import numpy as np

import adm1_numba
from adm1_numba import ADM1_lsoda, DEFAULT_F0, DEFAULT_U0, calculate_observables, funcptr, run_stage


def test_public_imports_and_default_shapes():
    assert adm1_numba.__version__ == "0.1.0"
    assert adm1_numba.__all__ == [
        "ADM1_lsoda",
        "DEFAULT_F0",
        "DEFAULT_U0",
        "calculate_observables",
        "funcptr",
        "run_stage",
    ]
    assert ADM1_lsoda.address == funcptr
    assert DEFAULT_F0.shape == (25,)
    assert DEFAULT_U0.shape == (33,)


def test_calculate_observables_returns_expected_shapes():
    sol = np.tile(DEFAULT_U0, (3, 1))

    qch4, qco2, ph = calculate_observables(sol)

    assert qch4.shape == (3,)
    assert qco2.shape == (3,)
    assert ph.shape == (3,)
    assert np.all(np.isfinite(qch4))
    assert np.all(np.isfinite(qco2))
    assert np.all(np.isfinite(ph))


def test_run_stage_smoke():
    sol, qch4, qco2, ph = run_stage(
        DEFAULT_U0.copy(),
        deltaT=0.005,
        f0=DEFAULT_F0.copy(),
    )

    assert sol.shape == (5, 33)
    assert qch4.shape == (5,)
    assert qco2.shape == (5,)
    assert ph.shape == (5,)
    assert np.all(np.isfinite(sol))
    assert np.all(np.isfinite(ph))
