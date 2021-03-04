#module CombustionModel

import Models
using ParametricModels
import Sundials

ϵ = 0
@parameterspace mutable struct Combustion
    m_1 = 1.00797, constant
    m_2 = 2.01594, constant
    m_3 = 15.9994, constant
    m_4 = 31.9988, constant
    m_5 = 17.00737, constant
    m_6 = 18.01534, constant
    m_7 = 33.00677, constant
    m_8 = 34.01474, constant
    m_9 = 28.0134, constant
    y_1_init = 0.0, constant
    y_2_init = 0.02852097, constant
    y_3_init = 0.0, constant
    y_4_init = 0.2263619, constant
    y_5_init = 0.0, constant
    y_6_init = 0.0, constant
    y_7_init = 0.0, constant
    y_8_init = 0.0, constant
    y_9_init = 0.74511713, constant
    R = 83145100.0, constant
    T_init = 1200.0, constant
    press = 1013250.0, constant
    A_1 = 32.878, identity
    B_1 = ϵ, constant
    Ta_1 = 8272.9, identity
    A_2 = 26.88, identity
    B_2 = 0.41, identity
    Ta_2 = 203.97, identity
    A_3 = 10.845, identity
    B_3 = 2.67, identity
    Ta_3 = 3165.2, identity
    A_4 = 10.502, identity
    B_4 = 2.61, identity
    Ta_4 = 2456.9, identity
    A_5 = 19.181, identity
    B_5 = 1.51, identity
    Ta_5 = 1726.0, identity
    A_6 = 23.786, identity
    B_6 = 1.14, identity
    Ta_6 = 9643.8, identity
    A_7 = 26.938, identity
    B_7 = ϵ, constant
    Ta_7 = 8197.4, identity
    A_8 = 31.886, identity
    B_8 = 0.32, identity
    Ta_8 = 16823.0, identity
    A_9 = 45.269, identity
    B_9 = 1.4, identity
    Ta_9 = 52526.0, identity
    A_10 = 44.078, identity
    B_10 = 1.43, identity
    Ta_10 = 29.11, identity
    A_11 = 36.358, identity
    B_11 = 0.5, identity
    Ta_11 = ϵ, constant
    A_12 = 43.204, identity
    B_12 = 0.94, identity
    Ta_12 = 60265.0, identity
    A_13 = 42.99, identity
    B_13 = 1.0, identity
    Ta_13 = ϵ, constant
    A_14 = 43.837, identity
    B_14 = 1.03, identity
    Ta_14 = 51788.0, identity
    A_15 = 51.463, identity
    B_15 = 2.0, identity
    Ta_15 = ϵ, constant
    A_16 = 57.259, identity
    B_16 = 2.35, identity
    Ta_16 = 60414.0, identity
    A_17 = 45.66, identity
    B_17 = 1.42, identity
    Ta_17 = ϵ, constant
    A_18 = 49.885, identity
    B_18 = 1.9, identity
    Ta_18 = 25210.0, identity
    A_19 = 31.822, identity
    B_19 = ϵ, constant
    Ta_19 = 1071.9, identity
    A_20 = 28.788, identity
    B_20 = 0.51, identity
    Ta_20 = 28359.0, identity
    A_21 = 32.767, identity
    B_21 = ϵ, constant
    Ta_21 = 437.8, identity
    A_22 = 23.391, identity
    B_22 = 0.86, identity
    Ta_22 = 18539.0, identity
    A_23 = 30.487, identity
    B_23 = ϵ, constant
    Ta_23 = 201.29, identity
    A_24 = 27.11, identity
    B_24 = 0.45, identity
    Ta_24 = 26377.0, identity
    A_25 = 37.213, identity
    B_25 = 1.0, identity
    Ta_25 = ϵ, constant
    A_26 = 38.784, identity
    B_26 = 0.86, identity
    Ta_26 = 35205.0, identity
    A_27 = 28.736, identity
    B_27 = ϵ, constant
    Ta_27 = 24.3142, identity
    A_28 = 31.591, identity
    B_28 = 0.12, identity
    Ta_28 = 19970.0, identity
    A_29 = 39.326, identity
    B_29 = ϵ, constant
    Ta_29 = 22896.0, identity
    A_30 = 22.872, identity
    B_30 = 1.47, identity
    Ta_30 = 3482.2, identity
    A_31 = 29.934, identity
    B_31 = ϵ, constant
    Ta_31 = 1806.5, identity
    A_32 = 19.274, identity
    B_32 = 1.12, identity
    Ta_32 = 35842.0, identity
    A_33 = 31.5, identity
    B_33 = ϵ, constant
    Ta_33 = 4006.0, identity
    A_34 = 25.612, identity
    B_34 = 0.63, identity
    Ta_34 = 12017.0, identity
    A_35 = 16.072, identity
    B_35 = 2.0, identity
    Ta_35 = 1997.8, identity
    A_36 = 9.8405, identity
    B_36 = 2.58, identity
    Ta_36 = 9305.9, identity
    A_37 = 29.588, identity
    B_37 = ϵ, constant
    Ta_37 = 719.6, identity
    A_38 = 28.305, identity
    B_38 = 0.26, identity
    Ta_38 = 16654.0, identity
    h1_1 = 82487670.0, identity
    h1_2 = 25473.6599, identity
    h1_3 = 2.50000001, identity
    h1_4 = -1.15421486e-11, identity
    h1_5 = 5.38539827e-15, identity
    h1_6 = -1.18378809e-18, identity
    h1_7 = 9.96394714e-23, identity
    h2_1 = 41243840.0, identity
    h2_2 = -950.158922, identity
    h2_3 = 3.3372792, identity
    h2_4 = -2.47012365e-05, identity
    h2_5 = 1.66485593e-07, identity
    h2_6 = -4.48915985e-11, identity
    h2_7 = 4.00510752e-15, identity
    h3_1 = 5196764.0, identity
    h3_2 = 29217.5791, identity
    h3_3 = 2.569420778, identity
    h3_4 = -4.29870569e-05, identity
    h3_5 = 1.39828196e-08, identity
    h3_6 = -2.50444497e-12, identity
    h3_7 = 2.45667382e-16, identity
    h4_1 = 2598382.0, identity
    h4_2 = -1088.45772, identity
    h4_3 = 3.28253784, identity
    h4_4 = 0.00074154377, identity
    h4_5 = -2.52655556e-07, identity
    h4_6 = 5.23676387e-11, identity
    h4_7 = -4.33435588e-15, identity
    h5_1 = 4888769.0, identity
    h5_2 = 3718.85774, identity
    h5_3 = 2.86472886, identity
    h5_4 = 0.0005285224, identity
    h5_5 = -8.63609193e-08, identity
    h5_6 = 7.63046685e-12, identity
    h5_7 = -2.66391752e-16, identity
    h6_1 = 4615239.0, identity
    h6_2 = -30004.2971, identity
    h6_3 = 3.03399249, identity
    h6_4 = 0.00108845902, identity
    h6_5 = -5.46908393e-08, identity
    h6_6 = -2.42604967e-11, identity
    h6_7 = 3.36401984e-15, identity
    h7_1 = 2519032.0, identity
    h7_2 = 111.856713, identity
    h7_3 = 4.0172109, identity
    h7_4 = 0.00111991006, identity
    h7_5 = -2.11219383e-07, identity
    h7_6 = 2.85615925e-11, identity
    h7_7 = -2.1581707e-15, identity
    h8_1 = 2444384.0, identity
    h8_2 = -17861.7877, identity
    h8_3 = 4.16500285, identity
    h8_4 = 0.00245415847, identity
    h8_5 = -6.33797417e-07, identity
    h8_6 = 9.27964965e-11, identity
    h8_7 = -5.7581661e-15, identity
    h9_1 = 2968047.0, identity
    h9_2 = -922.7977, identity
    h9_3 = 2.92664, identity
    h9_4 = 0.0007439884, identity
    h9_5 = -1.89492e-07, identity
    h9_6 = 2.5242595e-11, identity
    h9_7 = -1.3506702e-15, identity
    cp1_1 = 82487670.0, identity
    cp1_2 = 2.50000001, identity
    cp1_3 = -2.30842973e-11, identity
    cp1_4 = 1.61561948e-14, identity
    cp1_5 = -4.73515235e-18, identity
    cp1_6 = 4.98197357e-22, identity
    cp2_1 = 41243840.0, identity
    cp2_2 = 3.3372792, identity
    cp2_3 = -4.94024731e-05, identity
    cp2_4 = 4.99456778e-07, identity
    cp2_5 = -1.79566394e-10, identity
    cp2_6 = 2.00255376e-14, identity
    cp3_1 = 5196764.0, identity
    cp3_2 = 2.56942078, identity
    cp3_3 = -8.59741137e-05, identity
    cp3_4 = 4.19484589e-08, identity
    cp3_5 = -1.00177799e-11, identity
    cp3_6 = 1.22833691e-15, identity
    cp4_1 = 2598382.0, identity
    cp4_2 = 3.28253784, identity
    cp4_3 = 0.00148308754, identity
    cp4_4 = -7.57966669e-07, identity
    cp4_5 = 2.09470555e-10, identity
    cp4_6 = -2.16717794e-14, identity
    cp5_1 = 4888769.0, identity
    cp5_2 = 2.86472886, identity
    cp5_3 = 0.00105650448, identity
    cp5_4 = -2.59082758e-07, identity
    cp5_5 = 3.05218674e-11, identity
    cp5_6 = -1.33195876e-15, identity
    cp6_1 = 4615239.0, identity
    cp6_2 = 3.03399249, identity
    cp6_3 = 0.00217691804, identity
    cp6_4 = -1.64072518e-07, identity
    cp6_5 = -9.7041987e-11, identity
    cp6_6 = 1.68200992e-14, identity
    cp7_1 = 2519032.0, identity
    cp7_2 = 4.0172109, identity
    cp7_3 = 0.00223982013, identity
    cp7_4 = -6.3365815e-07, identity
    cp7_5 = 1.1424637e-10, identity
    cp7_6 = -1.07908535e-14, identity
    cp8_1 = 2444384.0, identity
    cp8_2 = 4.16500285, identity
    cp8_3 = 0.00490831694, identity
    cp8_4 = -1.90139225e-06, identity
    cp8_5 = 3.71185986e-10, identity
    cp8_6 = -2.87908305e-14, identity
    cp9_1 = 2968047.0, identity
    cp9_2 = 2.92664, identity
    cp9_3 = 0.0014879768, identity
    cp9_4 = -5.68476e-07, identity
    cp9_5 = 1.0097038e-10, identity
    cp9_6 = -6.753351e-15, identity
end

function inp(ps::Combustion{T}, _t) where T <: Real
    return T[]
end

function ic(ps::Combustion{T}) where T <: Real
    return T[ps.y_1_init, ps.y_2_init, ps.y_3_init, ps.y_4_init, ps.y_5_init, ps.y_6_init, ps.y_7_init, ps.y_8_init, ps.y_9_init, ps.T_init]
end

function rhs(ps::Combustion{T}, _t, _x, _dx) where T <: Real
    H1 = _x[10] .^ 5 .* ps.h1_1 .* ps.h1_7 + _x[10] .^ 4 .* ps.h1_1 .* ps.h1_6 + _x[10] .^ 3 .* ps.h1_1 .* ps.h1_5 + _x[10] .^ 2 .* ps.h1_1 .* ps.h1_4 + _x[10] .* ps.h1_1 .* ps.h1_3 + ps.h1_1 .* ps.h1_2
    H2 = _x[10] .^ 5 .* ps.h2_1 .* ps.h2_7 + _x[10] .^ 4 .* ps.h2_1 .* ps.h2_6 + _x[10] .^ 3 .* ps.h2_1 .* ps.h2_5 + _x[10] .^ 2 .* ps.h2_1 .* ps.h2_4 + _x[10] .* ps.h2_1 .* ps.h2_3 + ps.h2_1 .* ps.h2_2
    H3 = _x[10] .^ 5 .* ps.h3_1 .* ps.h3_7 + _x[10] .^ 4 .* ps.h3_1 .* ps.h3_6 + _x[10] .^ 3 .* ps.h3_1 .* ps.h3_5 + _x[10] .^ 2 .* ps.h3_1 .* ps.h3_4 + _x[10] .* ps.h3_1 .* ps.h3_3 + ps.h3_1 .* ps.h3_2
    H4 = _x[10] .^ 5 .* ps.h4_1 .* ps.h4_7 + _x[10] .^ 4 .* ps.h4_1 .* ps.h4_6 + _x[10] .^ 3 .* ps.h4_1 .* ps.h4_5 + _x[10] .^ 2 .* ps.h4_1 .* ps.h4_4 + _x[10] .* ps.h4_1 .* ps.h4_3 + ps.h4_1 .* ps.h4_2
    H5 = _x[10] .^ 5 .* ps.h5_1 .* ps.h5_7 + _x[10] .^ 4 .* ps.h5_1 .* ps.h5_6 + _x[10] .^ 3 .* ps.h5_1 .* ps.h5_5 + _x[10] .^ 2 .* ps.h5_1 .* ps.h5_4 + _x[10] .* ps.h5_1 .* ps.h5_3 + ps.h5_1 .* ps.h5_2
    H6 = _x[10] .^ 5 .* ps.h6_1 .* ps.h6_7 + _x[10] .^ 4 .* ps.h6_1 .* ps.h6_6 + _x[10] .^ 3 .* ps.h6_1 .* ps.h6_5 + _x[10] .^ 2 .* ps.h6_1 .* ps.h6_4 + _x[10] .* ps.h6_1 .* ps.h6_3 + ps.h6_1 .* ps.h6_2
    H7 = _x[10] .^ 5 .* ps.h7_1 .* ps.h7_7 + _x[10] .^ 4 .* ps.h7_1 .* ps.h7_6 + _x[10] .^ 3 .* ps.h7_1 .* ps.h7_5 + _x[10] .^ 2 .* ps.h7_1 .* ps.h7_4 + _x[10] .* ps.h7_1 .* ps.h7_3 + ps.h7_1 .* ps.h7_2
    H8 = _x[10] .^ 5 .* ps.h8_1 .* ps.h8_7 + _x[10] .^ 4 .* ps.h8_1 .* ps.h8_6 + _x[10] .^ 3 .* ps.h8_1 .* ps.h8_5 + _x[10] .^ 2 .* ps.h8_1 .* ps.h8_4 + _x[10] .* ps.h8_1 .* ps.h8_3 + ps.h8_1 .* ps.h8_2
    H9 = _x[10] .^ 5 .* ps.h9_1 .* ps.h9_7 + _x[10] .^ 4 .* ps.h9_1 .* ps.h9_6 + _x[10] .^ 3 .* ps.h9_1 .* ps.h9_5 + _x[10] .^ 2 .* ps.h9_1 .* ps.h9_4 + _x[10] .* ps.h9_1 .* ps.h9_3 + ps.h9_1 .* ps.h9_2
    CP1 = _x[10] .^ 4 .* ps.cp1_1 .* ps.cp1_6 + _x[10] .^ 3 .* ps.cp1_1 .* ps.cp1_5 + _x[10] .^ 2 .* ps.cp1_1 .* ps.cp1_4 + _x[10] .* ps.cp1_1 .* ps.cp1_3 + ps.cp1_1 .* ps.cp1_2
    CP2 = _x[10] .^ 4 .* ps.cp2_1 .* ps.cp2_6 + _x[10] .^ 3 .* ps.cp2_1 .* ps.cp2_5 + _x[10] .^ 2 .* ps.cp2_1 .* ps.cp2_4 + _x[10] .* ps.cp2_1 .* ps.cp2_3 + ps.cp2_1 .* ps.cp2_2
    CP3 = _x[10] .^ 4 .* ps.cp3_1 .* ps.cp3_6 + _x[10] .^ 3 .* ps.cp3_1 .* ps.cp3_5 + _x[10] .^ 2 .* ps.cp3_1 .* ps.cp3_4 + _x[10] .* ps.cp3_1 .* ps.cp3_3 + ps.cp3_1 .* ps.cp3_2
    CP4 = _x[10] .^ 4 .* ps.cp4_1 .* ps.cp4_6 + _x[10] .^ 3 .* ps.cp4_1 .* ps.cp4_5 + _x[10] .^ 2 .* ps.cp4_1 .* ps.cp4_4 + _x[10] .* ps.cp4_1 .* ps.cp4_3 + ps.cp4_1 .* ps.cp4_2
    CP5 = _x[10] .^ 4 .* ps.cp5_1 .* ps.cp5_6 + _x[10] .^ 3 .* ps.cp5_1 .* ps.cp5_5 + _x[10] .^ 2 .* ps.cp5_1 .* ps.cp5_4 + _x[10] .* ps.cp5_1 .* ps.cp5_3 + ps.cp5_1 .* ps.cp5_2
    CP6 = _x[10] .^ 4 .* ps.cp6_1 .* ps.cp6_6 + _x[10] .^ 3 .* ps.cp6_1 .* ps.cp6_5 + _x[10] .^ 2 .* ps.cp6_1 .* ps.cp6_4 + _x[10] .* ps.cp6_1 .* ps.cp6_3 + ps.cp6_1 .* ps.cp6_2
    CP7 = _x[10] .^ 4 .* ps.cp7_1 .* ps.cp7_6 + _x[10] .^ 3 .* ps.cp7_1 .* ps.cp7_5 + _x[10] .^ 2 .* ps.cp7_1 .* ps.cp7_4 + _x[10] .* ps.cp7_1 .* ps.cp7_3 + ps.cp7_1 .* ps.cp7_2
    CP8 = _x[10] .^ 4 .* ps.cp8_1 .* ps.cp8_6 + _x[10] .^ 3 .* ps.cp8_1 .* ps.cp8_5 + _x[10] .^ 2 .* ps.cp8_1 .* ps.cp8_4 + _x[10] .* ps.cp8_1 .* ps.cp8_3 + ps.cp8_1 .* ps.cp8_2
    CP9 = _x[10] .^ 4 .* ps.cp9_1 .* ps.cp9_6 + _x[10] .^ 3 .* ps.cp9_1 .* ps.cp9_5 + _x[10] .^ 2 .* ps.cp9_1 .* ps.cp9_4 + _x[10] .* ps.cp9_1 .* ps.cp9_3 + ps.cp9_1 .* ps.cp9_2
    rho = ps.press ./ (_x[10] .* _x[1] .* ps.R ./ ps.m_1 + _x[10] .* _x[2] .* ps.R ./ ps.m_2 + _x[10] .* _x[3] .* ps.R ./ ps.m_3 + _x[10] .* _x[4] .* ps.R ./ ps.m_4 + _x[10] .* _x[5] .* ps.R ./ ps.m_5 + _x[10] .* _x[6] .* ps.R ./ ps.m_6 + _x[10] .* _x[7] .* ps.R ./ ps.m_7 + _x[10] .* _x[8] .* ps.R ./ ps.m_8 + _x[10] .* _x[9] .* ps.R ./ ps.m_9)
    C1 = _x[1] .* rho ./ ps.m_1
    C2 = _x[2] .* rho ./ ps.m_2
    C3 = _x[3] .* rho ./ ps.m_3
    C4 = _x[4] .* rho ./ ps.m_4
    C5 = _x[5] .* rho ./ ps.m_5
    C6 = _x[6] .* rho ./ ps.m_6
    C7 = _x[7] .* rho ./ ps.m_7
    C8 = _x[8] .* rho ./ ps.m_8
    C9 = _x[9] .* rho ./ ps.m_9
    thd = ps.press ./ (_x[10] .* ps.R)
    R1 = C1 .* C4 .*  exp.(ps.A_1) .*  exp.(-ps.Ta_1 ./ _x[10]) .*  exp.(ps.B_1 .* log(_x[10]))
    R2 = C3 .* C5 .*  exp.(ps.A_2) .*  exp.(ps.Ta_2 ./ _x[10]) .*  exp.(ps.B_2 .* log(_x[10]))
    R3 = C2 .* C3 .*  exp.(ps.A_3) .*  exp.(-ps.Ta_3 ./ _x[10]) .*  exp.(ps.B_3 .* log(_x[10]))
    R4 = C1 .* C5 .*  exp.(ps.A_4) .*  exp.(-ps.Ta_4 ./ _x[10]) .*  exp.(ps.B_4 .* log(_x[10]))
    R5 = C2 .* C5 .*  exp.(ps.A_5) .*  exp.(-ps.Ta_5 ./ _x[10]) .*  exp.(ps.B_5 .* log(_x[10]))
    R6 = C1 .* C6 .*  exp.(ps.A_6) .*  exp.(-ps.Ta_6 ./ _x[10]) .*  exp.(ps.B_6 .* log(_x[10]))
    R7 = C5 .^ 2 .*  exp.(ps.A_7) .*  exp.(-ps.Ta_7 ./ _x[10]) .*  exp.(ps.B_7 .* log(_x[10]))
    R8 = C3 .* C6 .*  exp.(ps.A_8) .*  exp.(-ps.Ta_8 ./ _x[10]) .*  exp.(-ps.B_8 .* log(_x[10]))
    R9 = C2 .* C9 .*  exp.(ps.A_9) .*  exp.(-ps.Ta_9 ./ _x[10]) .*  exp.(-ps.B_9 .* log(_x[10]))
    R10 = C1 .^ 2 .* C9 .*  exp.(ps.A_10) .*  exp.(-ps.Ta_10 ./ _x[10]) .*  exp.(-ps.B_10 .* log(_x[10]))
    R11 = C3 .^ 2 .* C9 .*  exp.(ps.A_11) .*  exp.(-ps.Ta_11 ./ _x[10]) .*  exp.(-ps.B_11 .* log(_x[10]))
    R12 = C4 .* C9 .*  exp.(ps.A_12) .*  exp.(-ps.Ta_12 ./ _x[10]) .*  exp.(-ps.B_12 .* log(_x[10]))
    R13 = C1 .* C3 .* thd .*  exp.(ps.A_13) .*  exp.(-ps.Ta_13 ./ _x[10]) .*  exp.(-ps.B_13 .* log(_x[10]))
    R14 = C5 .* thd .*  exp.(ps.A_14) .*  exp.(-ps.Ta_14 ./ _x[10]) .*  exp.(-ps.B_14 .* log(_x[10]))
    R15 = C1 .* C5 .* C9 .*  exp.(ps.A_15) .*  exp.(-ps.Ta_15 ./ _x[10]) .*  exp.(-ps.B_15 .* log(_x[10]))
    R16 = C6 .* C9 .*  exp.(ps.A_16) .*  exp.(-ps.Ta_16 ./ _x[10]) .*  exp.(-ps.B_16 .* log(_x[10]))
    R17 = C1 .* C4 .* C9 .*  exp.(ps.A_17) .*  exp.(-ps.Ta_17 ./ _x[10]) .*  exp.(-ps.B_17 .* log(_x[10]))
    R18 = C7 .* C9 .*  exp.(ps.A_18) .*  exp.(-ps.Ta_18 ./ _x[10]) .*  exp.(-ps.B_18 .* log(_x[10]))
    R19 = C1 .* C7 .*  exp.(ps.A_19) .*  exp.(-ps.Ta_19 ./ _x[10]) .*  exp.(ps.B_19 .* log(_x[10]))
    R20 = C2 .* C4 .*  exp.(ps.A_20) .*  exp.(-ps.Ta_20 ./ _x[10]) .*  exp.(ps.B_20 .* log(_x[10]))
    R21 = C1 .* C7 .*  exp.(ps.A_21) .*  exp.(-ps.Ta_21 ./ _x[10]) .*  exp.(ps.B_21 .* log(_x[10]))
    R22 = C5 .^ 2 .*  exp.(ps.A_22) .*  exp.(-ps.Ta_22 ./ _x[10]) .*  exp.(ps.B_22 .* log(_x[10]))
    R23 = C3 .* C7 .*  exp.(ps.A_23) .*  exp.(ps.Ta_23 ./ _x[10]) .*  exp.(ps.B_23 .* log(_x[10]))
    R24 = C4 .* C5 .*  exp.(ps.A_24) .*  exp.(-ps.Ta_24 ./ _x[10]) .*  exp.(ps.B_24 .* log(_x[10]))
    R25 = C5 .* C7 .*  exp.(ps.A_25) .*  exp.(-ps.Ta_25 ./ _x[10]) .*  exp.(-ps.B_25 .* log(_x[10]))
    R26 = C4 .* C6 .*  exp.(ps.A_26) .*  exp.(-ps.Ta_26 ./ _x[10]) .*  exp.(-ps.B_26 .* log(_x[10]))
    R27 = C7 .^ 2 .*  exp.(ps.A_27) .*  exp.(-ps.A_27 .* ps.Ta_27 ./ _x[10])
    R28 = C4 .* C8 .*  exp.(ps.A_28) .*  exp.(-ps.Ta_28 ./ _x[10]) .*  exp.(-ps.B_28 .* log(_x[10]))
    R29 = C8 .* C9 .*  exp.(ps.A_29) .*  exp.(-ps.Ta_29 ./ _x[10]) .*  exp.(ps.B_29 .* log(_x[10]))
    R30 = C5 .^ 2 .* C9 .*  exp.(ps.A_30) .*  exp.(ps.Ta_30 ./ _x[10]) .*  exp.(ps.B_30 .* log(_x[10]))
    R31 = C1 .* C8 .*  exp.(ps.A_31) .*  exp.(-ps.Ta_31 ./ _x[10]) .*  exp.(ps.B_31 .* log(_x[10]))
    R32 = C5 .* C6 .*  exp.(ps.A_32) .*  exp.(-ps.Ta_32 ./ _x[10]) .*  exp.(ps.B_32 .* log(_x[10]))
    R33 = C1 .* C8 .*  exp.(ps.A_33) .*  exp.(-ps.Ta_33 ./ _x[10]) .*  exp.(ps.B_33 .* log(_x[10]))
    R34 = C2 .* C7 .*  exp.(ps.A_34) .*  exp.(-ps.Ta_34 ./ _x[10]) .*  exp.(ps.B_34 .* log(_x[10]))
    R35 = C3 .* C8 .*  exp.(ps.A_35) .*  exp.(-ps.Ta_35 ./ _x[10]) .*  exp.(ps.B_35 .* log(_x[10]))
    R36 = C5 .* C7 .*  exp.(ps.A_36) .*  exp.(-ps.Ta_36 ./ _x[10]) .*  exp.(ps.B_36 .* log(_x[10]))
    R37 = C5 .* C8 .*  exp.(ps.A_37) .*  exp.(-ps.Ta_37 ./ _x[10]) .*  exp.(ps.B_37 .* log(_x[10]))
    R38 = C6 .* C7 .*  exp.(ps.A_38) .*  exp.(-ps.Ta_38 ./ _x[10]) .*  exp.(ps.B_38 .* log(_x[10]))
    dS1 = -R1 - 2*R10 - R13 + R14 - R15 + R16 - R17 + R18 - R19 + R2 + R20 - R21 + R22 + R3 - R31 + R32 - R33 + R34 - R4 + R5 - R6 + 2*R9
    dS2 = R10 + R19 - R20 - R3 + R33 - R34 + R4 - R5 + R6 - R9
    dS3 = R1 - 2*R11 + 2*R12 - R13 + R14 - R2 - R23 + R24 - R3 - R35 + R36 + R4 + R7 - R8
    dS4 = -R1 + R11 - R12 - R17 + R18 + R19 + R2 - R20 + R23 - R24 + R25 - R26 + R27 - R28
    dS5 = R1 + R13 - R14 - R15 + R16 - R2 + 2*R21 - 2*R22 + R23 - R24 - R25 + R26 + 2*R29 + R3 - 2*R30 + R31 - R32 + R35 - R36 - R37 + R38 - R4 - R5 + R6 - 2*R7 + 2*R8
    dS6 = R15 - R16 + R25 - R26 + R31 - R32 + R37 - R38 + R5 - R6 + R7 - R8
    dS7 = R17 - R18 - R19 + R20 - R21 + R22 - R23 + R24 - R25 + R26 - 2*R27 + 2*R28 + R33 - R34 + R35 - R36 + R37 - R38
    dS8 = R27 - R28 - R29 + R30 - R31 + R32 - R33 + R34 - R35 + R36 - R37 + R38
    dS9 = 0
    CP_AVG = CP1 .* _x[1] + CP2 .* _x[2] + CP3 .* _x[3] + CP4 .* _x[4] + CP5 .* _x[5] + CP6 .* _x[6] + CP7 .* _x[7] + CP8 .* _x[8] + CP9 .* _x[9]
    _inp = inp(ps, _t)
    _dx[1] = dS1 .* ps.m_1 ./ rho
    _dx[2] = dS2 .* ps.m_2 ./ rho
    _dx[3] = dS3 .* ps.m_3 ./ rho
    _dx[4] = dS4 .* ps.m_4 ./ rho
    _dx[5] = dS5 .* ps.m_5 ./ rho
    _dx[6] = dS6 .* ps.m_6 ./ rho
    _dx[7] = dS7 .* ps.m_7 ./ rho
    _dx[8] = dS8 .* ps.m_8 ./ rho
    _dx[9] = dS9 .* ps.m_9 ./ rho
    _dx[10] = -H1 .* dS1 .* ps.m_1 ./ (CP_AVG .* rho) - H2 .* dS2 .* ps.m_2 ./ (CP_AVG .* rho) - H3 .* dS3 .* ps.m_3 ./ (CP_AVG .* rho) - H4 .* dS4 .* ps.m_4 ./ (CP_AVG .* rho) - H5 .* dS5 .* ps.m_5 ./ (CP_AVG .* rho) - H6 .* dS6 .* ps.m_6 ./ (CP_AVG .* rho) - H7 .* dS7 .* ps.m_7 ./ (CP_AVG .* rho) - H8 .* dS8 .* ps.m_8 ./ (CP_AVG .* rho) - H9 .* dS9 .* ps.m_9 ./ (CP_AVG .* rho)
    nothing
end

function obs(ps::Combustion{T}, _t, _x) where T <: Real
    return T[_x[1], _x[2], _x[3], _x[4], _x[5], _x[6], _x[7], _x[8], _x[9], _x[10] - ps.T_init]
end

f(ps::Combustion{T}, t) where T <: Real = solve_ode(ps, ic, rhs, obs, t, Sundials.CVODE_BDF(), abstol = 1e-12, reltol = 1e-12)


t = range(0, stop = 0.001, length = 1001) |> collect
ydata = f(Combustion(), t)
weights = 1 ./ sqrt.(0.01.*abs.(ydata).^2 .+ 0.01)

data = ParametricModels.WLSData("Combustion", ParametricModels.ModelArgs(t), ydata, weights)
pmodel = PModel(Combustion, parameter_transforms, f, data)
model = Models.Model(pmodel)

x = ParametricModels.xvalues(pmodel)
# j = model.jacobian(x);

function objective(x)
    return 0.5*sum(abs2.(model.r(x)))
end

function gradient(x)
    return model.jacobian(x)' * model.r(x)
end

#end # module
