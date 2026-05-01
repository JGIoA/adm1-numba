"""Simulation utilities for the ADM1 Numba model."""

import numpy as np
from numbalsoda import lsoda

from .model import funcptr


def calculate_observables(final_sol: np.ndarray) -> tuple[float, float, float]:
    """Calculate methane flow, carbon dioxide flow, and pH from a solution array."""
    R = 0.083145  # [bar M^-1 K^-1]
    Patm = 1.013  # [bar]
    T0 = 298.15   # [K]
    T = 308.15    # [K]
    kp = 5e4     # [m3 d-1 bar-1]
    G_h2 = final_sol[:, 14]  # [kgCOD m-3]
    G_ch4 = final_sol[:, 15] # [kgCOD m-3]
    G_co2 = final_sol[:, 23] # [kmol (carbon) m-3]
    ph2  = G_h2  * R * T / 16  # [bar]
    pch4 = G_ch4 * R * T / 64  # [bar]
    pco2 = G_co2 * R * T       # [bar]
    ph2o = 0.0313 * np.exp(5290 * (1 / T0 - 1 / T))  # [bar]
    Pg = pch4 + ph2o + ph2 + pco2      # [bar]
    Qg = kp * (Pg - Patm) * Pg / Patm  # [m3/day]
    Qch4 = Qg * pch4 / Pg  # [m3/day]
    Qco2 = Qg * pco2 / Pg  # [m3/day]

    K_w = 10**-14.0 * np.exp((55900 / (100 * R)) * (1 / T0 - 1 / T))  # [M]
    S_IN = final_sol[:, 13]  # [kmol (nitrogen) m-3]
    # S_IC = final_sol[:, 24]  # [kmol (carbon) m-3]
    S_cat_i  = final_sol[:, 25] # [kmol m-3]
    S_an_i   = final_sol[:, 26] # [kmol m-3]
    S_va_i   = final_sol[:, 27] # [kgCOD m-3]
    S_bu_i   = final_sol[:, 28] # [kgCOD m-3]
    S_pro_i  = final_sol[:, 29] # [kgCOD m-3]
    S_ac_i   = final_sol[:, 30] # [kgCOD m-3]
    S_hco3_i = final_sol[:, 31] # [kmol (carbon) m-3]
    S_nh3    = final_sol[:, 32] # [kmol (nitrogen) m-3]
    S_nh4_i = S_IN - S_nh3      # [kmol (nitrogen) m-3]
    phi = (
        S_cat_i
        + S_nh4_i
        - S_hco3_i
        - S_ac_i / 64
        - S_pro_i / 112
        - S_bu_i / 160
        - S_va_i / 208
        - S_an_i
    )  # [kmol m-3]
    S_H_i = -phi * 0.5 + 0.5 * (phi**2 + 4 * K_w) ** 0.5  # [kmol (H+) m-3]
    pH = -np.log10(S_H_i)  # [-]

    return Qch4, Qco2, pH


def run_stage(
    u0: np.ndarray,
    deltaT: float,
    f0: np.ndarray,
    tint: float = 1e-3,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> tuple[np.ndarray, float, float, float]:
    """Run one ADM1 integration stage.
    
    Args:
        u0: Initial state vector.
        deltaT: Operating time in days.
        f0: Input parameters.
        tint: Integration time interval in days.
        rtol: Relative tolerance for the integration.
        atol: Absolute tolerance for the integration.
    Returns:
        sol: Raw solution vector.
        Qch4: Methane flow rate.
        Qco2: Carbon dioxide flow rate.
        pH: pH.
    """

    t_eval = np.arange(0, deltaT, tint)
    sol, success = lsoda(funcptr, u0, t_eval, f0, rtol=rtol, atol=atol)
    Qch4, Qco2, pH = calculate_observables(sol)
    return sol, Qch4, Qco2, pH
