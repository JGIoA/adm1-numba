"""Default ADM1 inputs used by the steady-state example"""

import numpy as np

# Default input parameters
# Xi, Xi_ch, Xi_li, Xi_pr, 
# Si_su, Si_aa, Si_fa, Si_va, Si_bu, 
# Si_pro, Si_ac, Si_h2, Si_ch4, Si_IN, 
# Xi_aa, Xi_fa, Xi_c4, Xi_pro, Xi_ac, 
# Xi_h2, Si_IC, Si_cat_i, Si_an_i, 
# Q, T 
DEFAULT_F0 = np.array([
    2, 5, 5, 20, 
    0.01, 0.001, 0.001, 0.001, 0.001, 
    0.001, 0.001, 1e-8, 1e-5, 0.01, 
    0.01, 0.01, 0.01, 0.01, 0.01, 
    0.01, 0.04, 0.04, 0.02, 
    170, 308.15
])


# Default initial conditions
# dXdt, dX_chdt, dX_lidt, dX_prdt, \
# dS_sudt, dS_aadt, dS_fadt, dS_vadt, dS_budt, dS_prodt, dS_acdt, \
# dS_h2dt, dS_ch4dt, dS_INdt, \
# dG_h2dt, dG_ch4dt, dX_sudt, dX_aadt, dX_fadt, dX_c4dt, dX_prodt, dX_acdt, dX_h2dt, dG_co2dt, dS_ICdt,\
# dS_cat_idt, dS_an_idt, dS_va_idt, dS_bu_idt, dS_pro_idt, dS_ac_idt, dS_hco3_idt, dS_nh3dt
DEFAULT_U0 = np.array([
    0.31, 0.028, 0.029, 0.1,
    0.012, 0.0053, 0.099, 0.012, 0.013, 0.016, 0.2,
    2.3e-7, 0.055, 0.13,
    1.2e-5, 1.63, 0.42, 1.18, 0.24, 0.43, 0.14, 0.76, 0.32, 0.014, 0.15,
    0.04, 0.02, 0.011, 0.013, 0.016, 0.2, 0.14, 0.0041
])