"""ADM1 reactor model with numba"""

import numba
import numpy as np
from numbalsoda import lsoda_sig


@numba.cfunc(lsoda_sig)
def ADM1_lsoda(t, Z, dz, p):
  ## 1. Input parameters
    Xi,    Xi_ch, Xi_li, Xi_pr, Si_su, Si_aa, Si_fa, Si_va, Si_bu, Si_pro,  Si_ac, Si_h2, Si_ch4,Si_IN, Xi_aa, Xi_fa, Xi_c4, Xi_pro, Xi_ac, Xi_h2, Si_IC, Si_cat_i, Si_an_i, Q, T = (
    p[0],  p[1],  p[2],  p[3],  p[4],  p[5],   p[6],  p[7],   p[8],  p[9],  p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18],  p[19], p[20],  p[21],   p[22],  p[23], p[24])
    # Q: input flow rate [m3/day]
    # T: temperature [K]
    V  = 3400. # Volume of reactor [m3]
    Vg = 300.  # Gas volume of reactor [m3]

  ## 2. Constants
    # Thermal Constants
    R    = 0.083145  # Gas constant [bar M-1 K-1]
    Patm = 1.013     # Atmospheric pressure [bar]
    T0   = 298.15    # Reference temperature [K]
    # T    = 308.15  # Operating temperature [K]

    # Stoichiometric Constants
    # Inhibitor
    Nxc  = 0.0376 / 14 # [-]
    Naa  = 0.007       # [kmol (nitrogen) kg-1 COD]
    NI   = 0.06 / 14   # [kmol (nitrogen) kg-1 COD]
    Nbac = 0.08 / 14   # [kmol (nitrogen) kg-1 COD]

    # Inhibitor and Composites
    fsi_xc = 0.1  # [-]
    fxi_xc = 0.2  # [-]
    fch_xc = 0.2  # [-] 
    fli_xc = 0.3  # [-]
    fpr_xc = 0.2  # [-] 
    
    # Liquid Phase
    ffa_li = 0.95 # [-]
    fh2_su = 0.19 # [-]
    fbu_su = 0.13 # [-]
    fpro_su= 0.27 # [-]
    fac_su = 0.41 # [-]
    fh2_aa = 0.06 # [-]
    fva_aa = 0.23 # [-]
    fbu_aa = 0.26 # [-]
    fpro_aa= 0.05 # [-]
    fac_aa = 0.40 # [-]
    Ysu    = 0.1  # [-]
    Yaa    = 0.08 # [-]
    Yfa    = 0.06 # [-]
    Yc4    = 0.06 # [-]
    Ypro   = 0.04 # [-]
    Yac    = 0.05 # [-]
    Yh2    = 0.06 # [-]

    # Rate Constants
    kdis  = 0.5  # [d-1] 
    kh_ch = 10   # [d-1] 
    kh_pr = 10   # [d-1] 
    kh_li = 10   # [d-1] 

    # Liquid Phase
    k_msu  = 30      # [d-1]
    K_Ssu  = 0.5     # [kgCOD m-3]
    k_maa  = 50      # [d-1]
    K_Saa  = 0.3     # [kgCOD m-3]
    k_mfa  = 6       # [d-1]
    K_Sfa  = 0.4     # [kgCOD m-3]
    k_mc4  = 20      # [d-1]
    K_Sc4  = 0.2     # [kgCOD m-3]
    k_mpro = 13      # [d-1]
    K_Spro = 0.1     # [kgCOD m-3]
    k_mac  = 8       # [d-1]
    K_Sac  = 0.15    # [kgCOD m-3]
    k_mh2  = 35      # [d-1]
    K_Sh2  = 7e-6    # [kgCOD m-3]
    kdec_Xsu = 0.02  # [d-1]
    kdec_Xaa = 0.02  # [d-1]
    kdec_Xfa = 0.02  # [d-1]
    kdec_Xc4 = 0.02  # [d-1]
    kdec_Xpro= 0.02  # [d-1]
    kdec_Xac = 0.02  # [d-1]
    kdec_Xh2 = 0.02  # [d-1]

    # Inhibitor
    KS_IN    = 1e-4   # [M]
    K_Ih2fa  = 5e-6   # [kgCOD m-3]
    K_Ih2c4  = 1e-5   # [kgCOD m-3]
    K_Ih2pro = 3.5e-6 # [kgCOD m-3]
    K_Inh3   = 0.0018 # [M]

    # Gas Phase
    kp    = 5e4      # [m3 d-1 bar-1] (Need calibration for different reactors)
    KLa   = 200      # [d-1] 
    KHco2 =  0.035 * np.exp((-19410 / (100 * R)) * (1/T0 - 1/T))      # [Mliq bar-1] ~0.027147 @ 308.15K
    KHch4 = 0.0014 * np.exp((-14240 / (100 * R)) * (1/T0 - 1/T))      # [Mliq bar-1] ~0.001162 @ 308.15K
    KHh2  = 7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1/T0 - 1/T)) # [Mliq bar-1] ~7.38e-4 @ 308.15K

    # pH and ion
    pH_UL_aa =  5.5 # [-]
    pH_LL_aa =  4   # [-]
    pH_UL_ac =  7   # [-]
    pH_LL_ac =  6   # [-]
    pH_UL_h2 =  6   # [-]
    pH_LL_h2 =  5   # [-]
    K_w =  10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T0 - 1 / T)) # [M] ~2.08e-14 @ 308.15K
    K_a_va  = 1.38e-5 # [M]
    K_a_bu  = 1.5e-5  # [M]
    K_a_pro = 1.32e-5 # [M]
    K_a_ac  = 1.74e-5 # [M]
    K_a_co2 =  10 ** -6.35 * np.exp(( 7646 / (100 * R)) * (1 / T0 - 1 / T)) # [M] ~4.94e-7 @ 308.15K
    K_a_IN  =  10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T0 - 1 / T)) # [M] ~1.11e-9 @ 308.15K
    k_A_B_va  =  10e10 # [M-1 d-1]
    k_A_B_bu  =  10e10 # [M-1 d-1]
    k_A_B_pro =  10e10 # [M-1 d-1]
    k_A_B_ac  =  10e10 # [M-1 d-1]
    k_A_B_co2 =  10e10 # [M-1 d-1]
    k_A_B_IN  =  10e10 # [M-1 d-1]

    # Carbon balance
    C_xc  =  0.02786 # [kmole (carbon) kg-1COD]
    C_sI  =  0.03    # [kmole (carbon) kg-1COD]
    C_ch  =  0.0313  # [kmole (carbon) kg-1COD]
    C_pr  =  0.03    # [kmole (carbon) kg-1COD]
    C_li  =  0.022   # [kmole (carbon) kg-1COD]
    C_xI  =  0.03    # [kmole (carbon) kg-1COD]
    C_su  =  0.0313  # [kmole (carbon) kg-1COD]
    C_aa  =  0.03    # [kmole (carbon) kg-1COD]
    C_fa  =  0.0217  # [kmole (carbon) kg-1COD]
    C_bu  =  0.025   # [kmole (carbon) kg-1COD]
    C_pro =  0.0268  # [kmole (carbon) kg-1COD]
    C_ac  =  0.0313  # [kmole (carbon) kg-1COD]
    C_bac =  0.0313  # [kmole (carbon) kg-1COD]
    C_ch4 =  0.0156  # [kmole (carbon) kg-1COD]
    C_va  =  0.024   # [kmole (carbon) kg-1COD]

  ## 3. Reactor states
    # Unit: [kgCOD m-3] if not specified
    # Particulate Matter
    X    = Z[0]  # 13 Composite 
    X_ch = Z[1]  # 14 Carbohydrates 
    X_li = Z[2]  # 16 Lipids 
    X_pr = Z[3]  # 15 Proteins 

    # # Liquid Phase 
    S_su = Z[4]  # 1 Sugars 
    S_aa = Z[5]  # 2 Amino acids 
    S_fa = Z[6]  # 3 Long chain fatty acids 
    S_va = Z[7]  # 4 Valerate 
    S_bu = Z[8]  # 5 Butyrate 
    S_pro= Z[9]  # 6 Propionate 
    S_ac = Z[10] # 7 Acetate 
    S_h2 = Z[11] # 8 Hydrogen 
    S_ch4= Z[12] # 9 Methane 
    S_IN = Z[13] #11 Inorganic nitrogen [kmol (nitrogen) m-3]

    # Gas States 
    G_h2  = Z[14] # 33 Hydrogen 
    G_ch4 = Z[15] # 34 Methane

    # Particulate Matter 
    X_su     = Z[16] # 17 Sugars
    X_aa     = Z[17] # 18 Amino acids
    X_fa     = Z[18] # 19 Fatty acids
    X_c4     = Z[19] # 20 C4 acids
    X_pro    = Z[20] # 21 Propionate
    X_ac     = Z[21] # 22 Acetate
    X_h2     = Z[22] # 23 Hydrogen

    # Ion & Others 
    G_co2    = Z[23] # 35 Carbon dioxide [kmol (carbon) m-3] 
    S_IC     = Z[24] # 10 Inorganic carbon [kmol (carbon) m-3] 
    S_cat_i  = Z[25] # 25 Cation [kmol m-3]
    S_an_i   = Z[26] # 26 Anion [kmol m-3]
    S_va_i   = Z[27] # 27 Valerate
    S_bu_i   = Z[28] # 28 Butyrate
    S_pro_i  = Z[29] # 29 Propionate
    S_ac_i   = Z[30] # 30 Acetate
    S_hco3_i = Z[31] # 31 Bicarbonate [kmol (carbon) m-3]
    S_nh3    = Z[32] # 32 Ammonia [kmol (nitrogen) m-3]

  ## 4. Reaction rates
    # pH 
    S_nh4_i = S_IN - S_nh3  # Ammonium [kmol (nitrogen) m-3]
    S_co2 = S_IC - S_hco3_i # Carbonate [kmol (carbon) m-3]
    phi = S_cat_i + S_nh4_i - S_hco3_i - S_ac_i/64 - S_pro_i/112 - S_bu_i/160 - S_va_i/208 - S_an_i # [kmol m-3]
    S_H_i = -phi*0.5 + 0.5*(phi**2+4*K_w)**0.5 # Hydrogen ion [kmol m-3]

    # Gas phase
    ph2o  = 0.0313 * np.exp(5290 * (1/T0 - 1/T)) # Water vapor pressure [bar] ~0.0557 @ 308.15K
    ph2  = G_h2  * R * T / 16     # Hydrogen pressure [bar]
    pch4 = G_ch4 * R * T / 64     # Methane pressure [bar]
    pco2 = G_co2 * R * T          # Carbon dioxide pressure [bar]
    Pg = pch4 + ph2o + ph2 + pco2 # Gas pressure [bar]
    Qg = kp * (Pg-Patm) * Pg/Patm # Gas flow rate [m3/day]
    Qg = Qg if Qg>=0 else 0       # Ensure Qg is non-negative

    pT8  = KLa * (S_h2  - 16 * KHh2 * ph2)   # Hydrogen gas transfer rate
    pT9  = KLa * (S_ch4 - 64 * KHch4 * pch4) # Methane gas transfer rate
    pT10 = KLa * (S_co2 - KHco2 * pco2)      # Carbon dioxide gas transfer rate

    # Inhibitor contributions
    Iaa = 10**(-(3/(pH_UL_aa - pH_LL_aa))*(pH_LL_aa+pH_UL_aa)/2) / (S_H_i**(3/(pH_UL_aa - pH_LL_aa)) + 10**(-(3/(pH_UL_aa - pH_LL_aa))*(pH_LL_aa+pH_UL_aa)/2)) # Amino acids
    Iac = 10**(-(3/(pH_UL_ac - pH_LL_ac))*(pH_LL_ac+pH_UL_ac)/2) / (S_H_i**(3/(pH_UL_ac - pH_LL_ac)) + 10**(-(3/(pH_UL_ac - pH_LL_ac))*(pH_LL_ac+pH_UL_ac)/2)) # Acetate
    Ih2 = 10**(-(3/(pH_UL_h2 - pH_LL_h2))*(pH_LL_h2+pH_UL_h2)/2) / (S_H_i**(3/(pH_UL_h2 - pH_LL_h2)) + 10**(-(3/(pH_UL_h2 - pH_LL_h2))*(pH_LL_h2+pH_UL_h2)/2)) # Hydrogen
    I5  = 1 / (1 + (KS_IN/S_IN)) * Iaa 
    I6  = I5
    I7  = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_h2 /K_Ih2fa))  * Iaa
    I8  = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_h2 /K_Ih2c4))  * Iaa
    I9  = I8
    I10 = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_h2 /K_Ih2pro)) * Iaa
    I11 = 1 / (1 + (KS_IN/S_IN)) * 1 / (1 + (S_nh3/K_Inh3))   * Iac
    I12 = 1 / (1 + (KS_IN/S_IN)) * Ih2

    # Particulate Matter
    p1 = kdis  * X     # Disintegration rate
    p2 = kh_ch * X_ch  # Hydrolysis of carbohydrates rate
    p3 = kh_pr * X_pr  # Hydrolysis of proteins rate
    p4 = kh_li * X_li  # Hydrolysis of lipids rate

    # Liquid Phase
    p5  = k_msu * S_su / (K_Ssu + S_su) * X_su * I5 # Uptake of sugars
    p6  = k_maa * S_aa / (K_Saa + S_aa) * X_aa * I6 # Uptake of amino acids
    p7  = k_mfa * S_fa / (K_Sfa + S_fa) * X_fa * I7 # Uptake of long chain fatty acids
    p8  = k_mc4 * S_va / (K_Sc4 + S_va) * S_va / (S_bu + S_va + 1e-6) * X_c4 * I8 # Uptake of valerate
    p9  = k_mc4 * S_bu / (K_Sc4 + S_bu) * S_bu / (S_bu + S_va + 1e-6) * X_c4 * I9 # Uptake of butyrate
    p10 = k_mpro* S_pro/ (K_Spro+ S_pro)* X_pro * I10 # Uptake of propionate
    p11 = k_mac * S_ac / (K_Sac + S_ac) * X_ac * I11  # Uptake of acetate
    p12 = k_mh2 * S_h2 / (K_Sh2 + S_h2) * X_h2 * I12  # Uptake of hydrogen gas

    # Decay in Particulate Matter
    p13 = kdec_Xsu * X_su  # Decay of sugars
    p14 = kdec_Xaa * X_aa  # Decay of amino acids
    p15 = kdec_Xfa * X_fa  # Decay of long chain fatty acids
    p16 = kdec_Xc4 * X_c4  # Decay of C4 acids
    p17 = kdec_Xpro* X_pro # Decay of propionate
    p18 = kdec_Xac * X_ac  # Decay of acetate
    p19 = kdec_Xh2 * X_h2  # Decay of hydrogen gas

    # Carbon Balance
    s1  = (-1 * C_xc + fsi_xc * C_sI + fch_xc * C_ch + fpr_xc * C_pr + fli_xc * C_li + fxi_xc * C_xI)
    s2  = (-1 * C_ch + C_su)
    s3  = (-1 * C_pr + C_aa)
    s4  = (-1 * C_li + (1 - ffa_li) * C_su + ffa_li * C_fa)
    s5  = (-1 * C_su + (1 - Ysu) * (fbu_su * C_bu + fpro_su * C_pro + fac_su * C_ac) + Ysu * C_bac)
    s6  = (-1 * C_aa + (1 - Yaa) * (fva_aa * C_va + fbu_aa * C_bu + fpro_aa * C_pro + fac_aa * C_ac) + Yaa * C_bac)
    s7  = (-1 * C_fa + (1 - Yfa) * 0.7 * C_ac + Yfa * C_bac)
    s8  = (-1 * C_va + (1 - Yc4) * 0.54 * C_pro + (1 - Yc4) * 0.31 * C_ac + Yc4 * C_bac)
    s9  = (-1 * C_bu + (1 - Yc4) * 0.8 * C_ac + Yc4 * C_bac)
    s10 = (-1 * C_pro + (1 - Ypro) * 0.57 * C_ac + Ypro * C_bac)
    s11 = (-1 * C_ac + (1 - Yac) * C_ch4 + Yac * C_bac)
    s12 = ((1 - Yh2) * C_ch4 + Yh2 * C_bac)
    s13 = (-1 * C_bac + C_xc)
    sigma =  (s1 * p1 + s2 * p2 + s3 * p3 + s4 * p4
              + s5 * p5 + s6 * p6 + s7 * p7 + s8 * p8
              + s9 * p9 + s10 * p10 + s11 * p11 + s12 * p12
              + s13 * (p13 + p14 + p15 + p16 + p17 + p18 + p19)
    )

    # Acid-Base Rates
    pA4  = k_A_B_va  * (S_va_i   * (K_a_va + S_H_i)  - K_a_va*S_va)   # Valerate
    pA5  = k_A_B_bu  * (S_bu_i   * (K_a_bu + S_H_i)  - K_a_bu*S_bu)   # Butyrate
    pA6  = k_A_B_pro * (S_pro_i  * (K_a_pro + S_H_i) - K_a_pro*S_pro) # Propionate
    pA7  = k_A_B_ac  * (S_ac_i   * (K_a_ac + S_H_i)  - K_a_ac*S_ac)   # Acetate
    pA10 = k_A_B_co2 * (S_hco3_i * (K_a_co2 + S_H_i) - K_a_co2*S_IC)  # Bicarbonate
    pA11 = k_A_B_IN  * (S_nh3    * (K_a_IN + S_H_i)  - K_a_IN*S_IN)   # Ammonia

  ## 5. Rate Equations (ODEs)

    # Particulate Matter
    dXdt = Q/V * (Xi - X) - p1 + p13+p14+p15+p16+p17+p18+p19 # 13 Disintegration
    dX_chdt = Q/V * (Xi_ch - X_ch) + fch_xc*p1 - p2          # 14 Carbohydrates
    dX_prdt = Q/V * (Xi_pr - X_pr) + fpr_xc*p1 - p3          # 15 Proteins
    dX_lidt = Q/V * (Xi_li - X_li) + fli_xc*p1 - p4          # 16 Lipids

    # Particulate Matter
    dX_sudt = Q/V * (0 - X_su) + Ysu*p5 - p13                # 17 Sugars
    dX_aadt = Q/V * (Xi_aa - X_aa) + Yaa*p6 - p14            # 18 Amino acids
    dX_fadt = Q/V * (Xi_fa - X_fa) + Yfa*p7 - p15            # 19 Fatty acids
    dX_c4dt = Q/V * (Xi_c4 - X_c4) + Yc4*p8 + Yc4*p9 - p16   # 20 C4 acids
    dX_prodt= Q/V * (Xi_pro - X_pro)+ Ypro*p10 - p17         # 21 Propionate
    dX_acdt = Q/V * (Xi_ac - X_ac) + Yac*p11 - p18           # 22 Acetate
    dX_h2dt = Q/V * (Xi_h2 - X_h2) + Yh2*p12 - p19           # 23 Hydrogen

    # Liquid Phase
    dS_sudt = Q/V * (Si_su - S_su) + p2 + (1-ffa_li)*p4 - p5 # 1 Sugars
    dS_aadt = Q/V * (Si_aa - S_aa) + p3 - p6                 # 2 Amino acids
    dS_fadt = Q/V * (Si_fa - S_fa) + ffa_li*p4 - p7          # 3 Fatty acids
    dS_vadt = Q/V * (Si_va - S_va) + (1-Yaa)*fva_aa*p6 - p8  # 4 Valerate
    dS_budt = Q/V * (Si_bu - S_bu) + (1-Ysu)*fbu_su*p5 + (1-Yaa)*fbu_aa*p6 - p9 # 5 Butyrate
    dS_prodt= Q/V * (Si_pro - S_pro)+(1-Ysu)*fpro_su*p5+ (1-Yaa)*fpro_aa*p6 + (1-Yc4)*0.54*p8 - p10 # 6 Propionate
    dS_acdt =(Q/V * (Si_ac - S_ac) + (1-Ysu)*fac_su*p5 + (1-Yaa)*fac_aa*p6
              + (1-Yfa)*0.7*p7 + (1-Yc4)*0.31*p8 + (1-Yc4)*0.8*p9
              + (1-Ypro)*0.57*p10 - p11) # 7 Acetate
    dS_h2dt =(Q/V * (Si_h2 - S_h2) + (1-Ysu)*fh2_su*p5 + (1-Yaa)*fh2_aa*p6
              + (1-Yfa)*0.3*p7 + (1-Yc4)*0.15*p8 + (1-Yc4)*0.2*p9
              + (1-Ypro)*0.43*p10 - p12 - pT8) # 8 Hydrogen
    dS_ch4dt= Q/V * (Si_ch4 - S_ch4) + (1-Yac)*p11 + (1-Yh2)*p12 - pT9 # 9 Methane

    dS_ICdt = Q/V  * (Si_IC - S_IC) - sigma - pT10 # 10 Inorganic carbon
    dS_INdt = (Q/V * (Si_IN - S_IN)
            - Ysu*Nbac*p5 + (Naa-Yaa*Nbac)*p6 - Yfa*Nbac*p7 - Yc4*Nbac*p8
            - Yc4*Nbac*p9 - Ypro*Nbac*p10 - Yac*Nbac*p11 - Yh2*Nbac*p12
            + (Nbac-Nxc)*(p13+p14+p15+p16+p17+p18+p19)
            + (Nxc - fxi_xc*NI - fsi_xc*NI - fpr_xc*Naa)*p1) # 11 Inorganic nitrogen

    # Gas Phase 
    dG_h2dt  = - G_h2  * Qg/Vg + pT8 *V/Vg # 33 Hydrogen
    dG_ch4dt = - G_ch4 * Qg/Vg + pT9 *V/Vg # 34 Methane
    dG_co2dt = - G_co2 * Qg/Vg + pT10*V/Vg # 35 Carbon dioxide

    # Ion Phase
    dS_cat_idt  = Q/V  * (Si_cat_i - S_cat_i) # 25 Cation
    dS_an_idt   = Q/V  * (Si_an_i - S_an_i)   # 26 Anion
    dS_va_idt   = -pA4  # 27 Valerate
    dS_bu_idt   = -pA5  # 28 Butyrate
    dS_pro_idt  = -pA6  # 29 Propionate
    dS_ac_idt   = -pA7  # 30 Acetate
    dS_hco3_idt = -pA10 # 31 Bicarbonate
    dS_nh3dt    = -pA11 # 32 Ammonia

  ## 6. Output
    dz[0], dz[1], dz[2], dz[3], \
    dz[4], dz[5], dz[6], dz[7], dz[8], dz[9], dz[10], \
    dz[11], dz[12], dz[13], \
    dz[14], dz[15], dz[16], dz[17], dz[18], dz[19], dz[20], dz[21] ,dz[22], dz[23] ,dz[24], \
    dz[25], dz[26], dz[27], dz[28], dz[29] ,dz[30], dz[31] ,dz[32]  =  \
    dXdt, dX_chdt, dX_lidt, dX_prdt, \
    dS_sudt, dS_aadt, dS_fadt, dS_vadt, dS_budt, dS_prodt, dS_acdt, \
    dS_h2dt, dS_ch4dt, dS_INdt, \
    dG_h2dt, dG_ch4dt, dX_sudt, dX_aadt, dX_fadt, dX_c4dt, dX_prodt, dX_acdt, dX_h2dt, dG_co2dt, dS_ICdt,\
    dS_cat_idt, dS_an_idt, dS_va_idt, dS_bu_idt, dS_pro_idt, dS_ac_idt, dS_hco3_idt, dS_nh3dt


funcptr = ADM1_lsoda.address
