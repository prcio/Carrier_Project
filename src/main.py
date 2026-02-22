#!/usr/bin/env python3
"""
Carrier Deploying Objects Simulation
Implements all deliverables from the paper:
  R1  - 2D az/el plot of true + noisy ejected-object positions
  R2  - Table of ranges, center, carrier range, vce
  R3  - 3D positions of carrier and ejected objects
  T1  - Table 1: standard deviations of speed estimates (position + Doppler)
  MLE - MLE estimates of vdelta and vL from the noisy measurements
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# ENU <-> spherical  (r, az, el)
# az: clockwise from North   el: above horizontal
# ---------------------------------------------------------------------------

def cart2sph_ENU(xyz: np.ndarray) -> np.ndarray:
    """Cartesian ENU -> (range, azimuth, elevation)."""
    x = xyz[..., 0]   # East
    y = xyz[..., 1]   # North
    z = xyz[..., 2]   # Up

    r   = np.sqrt(x*x + y*y + z*z)
    rho = np.sqrt(x*x + y*y)

    az = np.arctan2(x, y)       # clockwise from North  (East is positive)
    el = np.arctan2(z, rho)     # above horizontal

    return np.stack([r, az, el], axis=-1)


def sph2cart_ENU(rae: np.ndarray) -> np.ndarray:
    """(range, azimuth, elevation) -> Cartesian ENU."""
    r  = rae[..., 0]
    az = rae[..., 1]
    el = rae[..., 2]

    rho = r * np.cos(el)
    z   = r * np.sin(el)
    x   = rho * np.sin(az)   # East
    y   = rho * np.cos(az)   # North

    return np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# Rotation matrices  –  Eq. (7), (8), (10) of the paper
# ---------------------------------------------------------------------------

def A1(alpha_c: float) -> np.ndarray:
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    return np.array([[ca, sa, 0.],
                     [-sa, ca, 0.],
                     [0., 0., 1.]])


def A2(eps_c: float) -> np.ndarray:
    ce, se = np.cos(eps_c), np.sin(eps_c)
    return np.array([[1., 0., 0.],
                     [0., se,  ce],
                     [0., -ce, se]])


def A_total(alpha_c: float, eps_c: float) -> np.ndarray:
    """
    Full rotation matrix A = A2 @ A1  (Eq. 10).
    Columns: [1_x'', 1_y'', 1_z'' = 1_vc]
    A is orthogonal so A^{-1} = A^T.
    """
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    ce, se = np.cos(eps_c),   np.sin(eps_c)

    # Eq. (10) – written out explicitly
    A = np.array([
        [ ca,  sa*se,  sa*ce],
        [-sa,  ca*se,  ca*ce],
        [ 0.,    -ce,    se ],
    ], dtype=float)
    return A


# ---------------------------------------------------------------------------
# Measurement noise covariance in Cartesian  (for sigma_w)
# ---------------------------------------------------------------------------

def cartesian_noise_MSE(r: float, az: float, el: float,
                         sigma_r: float, sigma_az: float, sigma_el: float) -> float:
    """
    Linearised Cartesian position noise MSE  sigma_w^2 = tr(R_xyz).
    Jacobian of  sph2cart  at (r, az, el).
    """
    cos_el = np.cos(el)
    sin_el = np.sin(el)
    cos_az = np.cos(az)
    sin_az = np.sin(az)

    # Jacobian  d(x,y,z)/d(r, az, el)
    # x = r cos_el sin_az
    # y = r cos_el cos_az
    # z = r sin_el
    J = np.array([
        [cos_el*sin_az,  r*cos_el*cos_az,  -r*sin_el*sin_az],
        [cos_el*cos_az, -r*cos_el*sin_az,  -r*sin_el*cos_az],
        [sin_el,         0.,                r*cos_el        ],
    ])

    R_sph = np.diag([sigma_r**2, sigma_az**2, sigma_el**2])
    R_xyz = J @ R_sph @ J.T
    return float(np.trace(R_xyz))


# ---------------------------------------------------------------------------
# Simulation parameters  (Section 5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    n:           int   = 12
    t0:          float = 0.0
    t_rho:       float = 1.0     # resolution time
    g:           float = 9.81

    v_delta:     float = 20.0    # lateral ejection speed  [m/s]
    v_L:         float = 200.0   # longitudinal ejection speed  [m/s]

    xc_m:   tuple = (40e3, 40e3, 40e3)        # carrier position at ejection  [m]
    vc_mps: tuple = (-3.2e3, 0.0, -3.0e3)     # carrier velocity  [m/s]

    alpha_c_deg: float = -90.0
    eps_c_deg:   float = -43.0

    # Radar measurement noise
    sigma_r:  float = 5.0        # range  [m]
    sigma_az: float = 0.15e-3    # azimuth  [rad]
    sigma_el: float = 0.15e-3    # elevation  [rad]
    sigma_D:  float = 1.0        # Doppler  [m/s]

    # Track errors for carrier
    sigma_p:  float = 5.0        # carrier position RMSE  [m]
    sigma_vc: float = 5.0        # carrier velocity RMSE  [m/s]

    me_over_mc: float = 0.1
    seed:       int   = 0


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    n  = p.n
    dt = p.t_rho - p.t0

    xc = np.array(p.xc_m,   dtype=float)
    vc = np.array(p.vc_mps, dtype=float)

    alpha_c = np.deg2rad(p.alpha_c_deg)
    eps_c   = np.deg2rad(p.eps_c_deg)

    # -----------------------------------------------------------------------
    # Step 1 – unit circle dispersion velocities in ENU  (Eq. 4, 6)
    # -----------------------------------------------------------------------
    angles = 2.0 * np.pi * np.arange(1, n + 1) / n   # i = 1..n
    v_unit_circle = np.stack([
        np.sin(angles),         # x  (East)
        np.cos(angles),         # y  (North)
        np.zeros_like(angles),  # z  (Up)
    ], axis=1)   # shape (n, 3)

    # -----------------------------------------------------------------------
    # Step 2 – rotate to be orthogonal to vc  (Eq. 9, 10)
    # -----------------------------------------------------------------------
    A     = A_total(alpha_c, eps_c)   # (3,3)
    A_inv = A.T                        # A is orthogonal

    # A @ v_unit_circle[i]  maps the unit-circle directions to carrier-frame
    v_unit_disp = (A @ v_unit_circle.T).T   # (n, 3)

    # -----------------------------------------------------------------------
    # Carrier unit vector and speeds
    # -----------------------------------------------------------------------
    vc_norm = np.linalg.norm(vc)
    one_vc  = vc / vc_norm

    # -----------------------------------------------------------------------
    # Step 3 – dispersion velocities  (Eq. 12)
    # -----------------------------------------------------------------------
    v_disp = p.v_delta * v_unit_disp   # (n, 3)

    # -----------------------------------------------------------------------
    # Step 4 – initial object velocities  (Eq. 13, 14)
    # -----------------------------------------------------------------------
    v_long   = p.v_L * one_vc            # (3,)  longitudinal part
    v_obj0   = vc[None, :] + v_disp + v_long[None, :]   # (n, 3)

    # -----------------------------------------------------------------------
    # Step 4 – object positions at t_rho  (Eq. 15)
    # -----------------------------------------------------------------------
    grav_term = np.array([0., 0., -p.g]) * (dt**2) / 2.0   # (3,)
    x_obj_true = xc[None, :] + v_obj0 * dt + grav_term[None, :]   # (n, 3)

    # -----------------------------------------------------------------------
    # Step 5 – carrier position after ejection  (Eq. 16, 43)
    # -----------------------------------------------------------------------
    vce = (vc_norm - p.me_over_mc * p.v_L) / (1.0 - p.me_over_mc)   # Eq. 43
    v_car_after = vce * one_vc
    x_car_true  = xc + v_car_after * dt + grav_term   # (3,)

    # -----------------------------------------------------------------------
    # Step 6A – noiseless spherical measurements  (Eq. 17)
    # -----------------------------------------------------------------------
    u_obj_true = cart2sph_ENU(x_obj_true)   # (n, 3)  [r, az, el]

    # -----------------------------------------------------------------------
    # Step 6B – noisy spherical measurements
    # -----------------------------------------------------------------------
    noise = np.stack([
        rng.normal(0., p.sigma_r,  size=n),
        rng.normal(0., p.sigma_az, size=n),
        rng.normal(0., p.sigma_el, size=n),
    ], axis=1)
    u_obj_noisy = u_obj_true + noise   # (n, 3)

    # -----------------------------------------------------------------------
    # Step 7 – convert to Cartesian  (Eq. 22)
    # -----------------------------------------------------------------------
    z_obj_true  = sph2cart_ENU(u_obj_true)    # (n, 3)
    z_obj_noisy = sph2cart_ENU(u_obj_noisy)   # (n, 3)

    # -----------------------------------------------------------------------
    # Step 8 – centre of object measurements  (Eq. 24)
    # -----------------------------------------------------------------------
    z0_true  = z_obj_true.mean(axis=0)    # (3,)
    z0_noisy = z_obj_noisy.mean(axis=0)   # (3,)

    # -----------------------------------------------------------------------
    # Step 9 – rotate into (ideally) circular shape  (Eq. 25)
    # -----------------------------------------------------------------------
    y_true  = (A_inv @ (z_obj_true  - z0_true ).T).T   # (n, 3)
    y_noisy = (A_inv @ (z_obj_noisy - z0_noisy).T).T   # (n, 3)

    # -----------------------------------------------------------------------
    # sigma_w  (position measurement MSE in Cartesian)
    # Use the mean true range/az/el for a representative estimate
    # -----------------------------------------------------------------------
    r_mean  = float(u_obj_true[:, 0].mean())
    az_mean = float(u_obj_true[:, 1].mean())
    el_mean = float(u_obj_true[:, 2].mean())
    sigma_w_sq = cartesian_noise_MSE(r_mean, az_mean, el_mean,
                                      p.sigma_r, p.sigma_az, p.sigma_el)
    sigma_w = float(np.sqrt(sigma_w_sq))

    return {
        "params":       asdict(p),
        "dt":           dt,
        "A":            A,
        "A_inv":        A_inv,
        "one_vc":       one_vc,
        "vc_norm":      vc_norm,
        "v_unit_disp":  v_unit_disp,
        # true positions
        "x_obj_true":   x_obj_true,
        "x_car_true":   x_car_true,
        # spherical measurements
        "u_obj_true":   u_obj_true,
        "u_obj_noisy":  u_obj_noisy,
        # Cartesian measurements
        "z_obj_true":   z_obj_true,
        "z_obj_noisy":  z_obj_noisy,
        # centres
        "z0_true":      z0_true,
        "z0_noisy":     z0_noisy,
        # rotated/centred
        "y_true":       y_true,
        "y_noisy":      y_noisy,
        # derived scalars
        "vce":          vce,
        "sigma_w":      sigma_w,
    }


# ---------------------------------------------------------------------------
# Table 1 – theoretical standard deviations  (Eqs. 33, 41, 52, 57)
# ---------------------------------------------------------------------------

def compute_table1_sigmas(p: Params, sim: dict) -> dict:
    dt      = sim["dt"]                # t_rho - t0
    n       = p.n
    sigma_w = sim["sigma_w"]          # Cartesian position noise [m]

    # ------------------------------------------------------------------
    # From position measurements
    # ------------------------------------------------------------------

    # Dispersion speed s.d.  (Eq. 33):
    #   sigma^2_vdelta = (2 sigma_w^2 / n) / (t_rho - t0)^2
    sigma_vdelta_pos = np.sqrt((2.0 * sigma_w**2 / n) / dt**2)

    # Longitudinal speed s.d.  (Eq. 41):
    #   sigma^2_vL = (sigma_w^2/n + sigma_p^2) / (t_rho - t0)^2  +  sigma_vc^2
    sigma_vL_pos = np.sqrt(
        (sigma_w**2 / n + p.sigma_p**2) / dt**2 + p.sigma_vc**2
    )

    # ------------------------------------------------------------------
    # Baseline: 2-point differencing per object  (Eq. 34):
    #   sigma^2_v2pt = 2 sigma_w^2 / (t_rho - t0)^2
    # ------------------------------------------------------------------
    sigma_v2pt = np.sqrt(2.0 * sigma_w**2 / dt**2)

    # ------------------------------------------------------------------
    # From Doppler measurements
    # ------------------------------------------------------------------

    # LOS unit vector from radar (origin) to carrier
    xc     = np.array(p.xc_m, dtype=float)
    one_LOS = xc / np.linalg.norm(xc)   # Eq. 44

    # Projection of carrier velocity unit vector on LOS  (Eq. 46)
    one_vc = sim["one_vc"]
    phi_cLOS = float(one_vc @ one_LOS)

    # Longitudinal speed from Doppler  (Eq. 52):
    #   sigma^2_vLD = sigma_D^2 / phi_cLOS^2  +  sigma_vc^2
    sigma_vL_doppler = np.sqrt(p.sigma_D**2 / phi_cLOS**2 + p.sigma_vc**2)

    # Dispersion speed from Doppler  (Eq. 57):
    #   sigma^2_vdD = (1/n^2) * sum_{i=1}^{n/2}  2*sigma_D^2 / phi_di^2
    # where phi_di = 1_vdi . 1_LOS  (Eq. 53)
    v_unit_disp = sim["v_unit_disp"]   # (n, 3)
    phi_disp = v_unit_disp @ one_LOS   # (n,)  projection of each disp unit vec on LOS

    # Only first n/2 objects (opposite pairs cancel longitudinal velocity)
    n_half   = n // 2
    phi_half = phi_disp[:n_half]       # phi_delta_i for i = 1..n/2

    # Eq. 57:  sigma^2_vdD = (1/n^2) * sum_{i=1}^{n/2}  2 sigma_D^2 / phi_di^2
    sum_term = np.sum(2.0 * p.sigma_D**2 / phi_half**2)
    sigma_vdelta_doppler = np.sqrt(sum_term / n**2)

    return {
        "sigma_vdelta_pos":     sigma_vdelta_pos,
        "sigma_vL_pos":         sigma_vL_pos,
        "sigma_vdelta_doppler": sigma_vdelta_doppler,
        "sigma_vL_doppler":     sigma_vL_doppler,
        "sigma_v2pt_baseline":  sigma_v2pt,
        "phi_cLOS":             phi_cLOS,
    }


# ---------------------------------------------------------------------------
# MLE estimates from noisy position measurements  (Eqs. 32, 40)
# ---------------------------------------------------------------------------

def compute_mle_estimates(p: Params, sim: dict) -> dict:
    """
    Estimate v_delta and v_L from the noisy measurements using the ML
    estimators described in Sections 4.1 and 4.2.
    """
    dt      = sim["dt"]
    n       = p.n
    n_half  = n // 2
    y_noisy = sim["y_noisy"]     # (n, 3)  rotated+centred Cartesian meas.
    z0_noisy = sim["z0_noisy"]   # (3,)    centre of noisy measurements

    # ------------------------------------------------------------------
    # v_delta MLE  (Eq. 32):
    #   v_hat_delta = mean_{i=1}^{n/2}  ||y_i - y_{i+n/2}|| / (2 dt)
    #
    # Objects i and i + n/2 are on opposite sides of the circle.
    # (indices 0-based: i and i + n_half)
    # ------------------------------------------------------------------
    diffs = y_noisy[:n_half] - y_noisy[n_half:]   # (n/2, 3)
    d_i   = np.linalg.norm(diffs, axis=1)          # (n/2,)  Eq. 28 (noisy distances)

    v_hat_delta = d_i.mean() / (2.0 * dt)          # Eq. 32

    # ------------------------------------------------------------------
    # v_L MLE  (Eq. 40):
    #   v_hat_L = ||z0(t_rho) - x_hat_c(t0)|| / dt  -  v_hat_c
    #
    # We use the true (noise-free) carrier position estimate x_hat_c = xc
    # and the true carrier speed v_hat_c = vc_norm (in practice these come
    # from the carrier track; here we use the true values per the paper setup).
    # ------------------------------------------------------------------
    xc       = np.array(p.xc_m, dtype=float)
    vc_norm  = sim["vc_norm"]

    v_hat_L = np.linalg.norm(z0_noisy - xc) / dt - vc_norm   # Eq. 40

    return {
        "v_hat_delta": float(v_hat_delta),
        "v_hat_L":     float(v_hat_L),
        "d_i":         d_i,
    }


# ---------------------------------------------------------------------------
# Monte Carlo validation of theoretical sigmas
# ---------------------------------------------------------------------------

def run_monte_carlo(p: Params, n_trials: int = 1000) -> dict:
    v_hat_deltas = np.zeros(n_trials)
    v_hat_Ls     = np.zeros(n_trials)

    for k in range(n_trials):
        sim_k = run_sim(Params(**{**asdict(p), "seed": k}))
        mle_k = compute_mle_estimates(p, sim_k)
        v_hat_deltas[k] = mle_k["v_hat_delta"]
        v_hat_Ls[k]     = mle_k["v_hat_L"]

    return {
        "v_hat_deltas":         v_hat_deltas,
        "v_hat_Ls":             v_hat_Ls,
        "empirical_mean_delta": float(v_hat_deltas.mean()),
        "empirical_mean_L":     float(v_hat_Ls.mean()),
        "empirical_std_delta":  float(v_hat_deltas.std()),
        "empirical_std_L":      float(v_hat_Ls.std()),
        "n_trials":             n_trials,
    }


def plot_mc(outdir: Path, p: Params, mc: dict, sigmas: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, vals, true_val, theo_sigma, label in zip(
        axes,
        [mc["v_hat_deltas"], mc["v_hat_Ls"]],
        [p.v_delta,          p.v_L],
        [sigmas["sigma_vdelta_pos"], sigmas["sigma_vL_pos"]],
        ["$\\hat{v}_\\delta$", "$\\hat{v}_L$"],
    ):
        ax.hist(vals, bins=40, color='steelblue', alpha=0.7, density=True)
        ax.axvline(true_val,               color='green',  lw=1.5, ls='-',  label=f'True = {true_val:.1f}')
        ax.axvline(vals.mean(),            color='tomato', lw=1.5, ls='--', label=f'Mean = {vals.mean():.2f}')
        ax.axvline(true_val + theo_sigma,  color='gray',   lw=1.0, ls=':',  label=f'Theo. σ = {theo_sigma:.3f}')
        ax.axvline(true_val - theo_sigma,  color='gray',   lw=1.0, ls=':')
        ax.set_xlabel(f"{label} (m/s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}:  emp. σ = {vals.std():.4f},  theo. σ = {theo_sigma:.4f} m/s")
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle(f"Monte Carlo ({mc['n_trials']} trials) — position-based MLE")
    plt.tight_layout()
    plt.savefig(outdir / "MC.png", dpi=200)
    plt.close(fig)


def print_mc(p: Params, mc: dict, sigmas: dict) -> str:
    lines = [
        "=" * 65,
        f"Monte Carlo validation  (N = {mc['n_trials']} trials)",
        "=" * 65,
        f"{'':<38} {'v_delta':>12} {'v_L':>12}",
        "-" * 65,
        f"{'True value (m/s)':<38} {p.v_delta:>12.4f} {p.v_L:>12.4f}",
        f"{'MC mean (m/s)':<38} {mc['empirical_mean_delta']:>12.4f} {mc['empirical_mean_L']:>12.4f}",
        "-" * 65,
        f"{'Empirical std (m/s)':<38} {mc['empirical_std_delta']:>12.4f} {mc['empirical_std_L']:>12.4f}",
        f"{'Theoretical std — pos. (m/s)':<38} {sigmas['sigma_vdelta_pos']:>12.4f} {sigmas['sigma_vL_pos']:>12.4f}",
        "-" * 65,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# R2 – table of ranges
# ---------------------------------------------------------------------------

def compute_R2(p: Params, sim: dict) -> dict:
    u_obj_true = sim["u_obj_true"]   # (n, 3)  [r, az, el]
    z0_true    = sim["z0_true"]      # (3,)  Cartesian centre
    x_car_true = sim["x_car_true"]   # (3,)

    ranges_obj  = u_obj_true[:, 0]                          # (n,)
    range_centre = np.linalg.norm(z0_true)                  # scalar
    range_carrier = np.linalg.norm(x_car_true)              # scalar

    return {
        "ranges_obj":   ranges_obj,
        "range_centre": range_centre,
        "range_carrier": range_carrier,
        "vce":          sim["vce"],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_R1(outdir: Path, sim: dict) -> None:
    """
    R1 – 2D az/el plot (mrad) of true and noisy ejected object positions,
         plus the centre of the true measurements.
    """
    u_obj_true  = sim["u_obj_true"]    # (n, 3)
    u_obj_noisy = sim["u_obj_noisy"]   # (n, 3)

    # True positions in mrad
    az_true = u_obj_true[:, 1] * 1e3
    el_true = u_obj_true[:, 2] * 1e3

    # Noisy positions in mrad
    az_noisy = u_obj_noisy[:, 1] * 1e3
    el_noisy = u_obj_noisy[:, 2] * 1e3

    # Centre of true positions (converted to spherical)
    z0_true   = sim["z0_true"]
    u_z0_true = cart2sph_ENU(z0_true)
    az_ctr    = u_z0_true[1] * 1e3
    el_ctr    = u_z0_true[2] * 1e3

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw lines connecting each true point to its noisy counterpart first
    for i in range(len(az_true)):
        ax.plot([az_true[i], az_noisy[i]], [el_true[i], el_noisy[i]],
                color='gray', linewidth=0.7, alpha=0.6, zorder=1)

    ax.scatter(az_true,  el_true,  marker='o', color='steelblue', s=50,
               label='True positions', zorder=3)
    ax.scatter(az_noisy, el_noisy, marker='+', color='tomato', s=80,
               label='Noisy positions', zorder=3)
    ax.scatter(az_ctr,   el_ctr,   marker='*', s=150, color='green',
               label='Centre (true)', zorder=4)

    # Label object indices next to true points only
    for i in range(len(az_true)):
        ax.annotate(str(i+1), (az_true[i], el_true[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7,
                    color='steelblue')

    # Axis limits: include all points (true + noisy) with a tidy margin
    all_az = np.concatenate([az_true, az_noisy, [az_ctr]])
    all_el = np.concatenate([el_true, el_noisy, [el_ctr]])
    pad_az = 0.10 * (all_az.max() - all_az.min())
    pad_el = 0.10 * (all_el.max() - all_el.min())
    ax.set_xlim(all_az.min() - pad_az, all_az.max() + pad_az)
    ax.set_ylim(all_el.min() - pad_el, all_el.max() + pad_el)

    ax.set_xlabel("Azimuth (mrad)")
    ax.set_ylabel("Elevation (mrad)")
    ax.set_title("R1 – Ejected objects: az/el at $t_\\rho = 1\\,$s\n"
                 "(true = circles, noisy = crosses, centre = star)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "R1.png", dpi=200)
    plt.close(fig)


def plot_R3(outdir: Path, sim: dict) -> None:
    """
    R3 – 3D plot of ejected object positions + carrier position at t_rho.
    """
    x_obj = sim["x_obj_true"]   # (n, 3)  [m]
    x_car = sim["x_car_true"]   # (3,)    [m]

    # Convert to km for readability
    x_obj_km = x_obj / 1e3
    x_car_km = x_car / 1e3

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(x_obj_km[:, 0], x_obj_km[:, 1], x_obj_km[:, 2],
               s=60, marker='o', color='steelblue', label='Ejected objects')
    ax.scatter([x_car_km[0]], [x_car_km[1]], [x_car_km[2]],
               s=120, marker='X', color='tomato', label='Carrier')

    # Label object indices
    for i in range(len(x_obj_km)):
        ax.text(x_obj_km[i, 0], x_obj_km[i, 1], x_obj_km[i, 2],
                f' {i+1}', fontsize=7)

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Up (km)")
    ax.set_title("R3 – 3D positions at $t_\\rho = 1\\,$s\n"
                 "(carrier = X, objects = circles)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "R3.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print deliverables to stdout and to a text file
# ---------------------------------------------------------------------------

def print_R2(p: Params, r2: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("R2 – True ranges of ejected objects at t_rho = 1 s")
    lines.append("=" * 60)
    for i, rng in enumerate(r2["ranges_obj"]):
        lines.append(f"  Object {i+1:2d}:  range = {rng/1e3:8.4f} km")
    lines.append(f"  Centre:     range = {r2['range_centre']/1e3:8.4f} km")
    lines.append(f"  Carrier:    range = {r2['range_carrier']/1e3:8.4f} km")
    lines.append(f"  vce (carrier speed after ejection): {r2['vce']:.4f} m/s")
    lines.append("")
    return "\n".join(lines)


def print_table1(p: Params, sigmas: dict, mle: dict, mc: dict = None) -> str:
    W = 72
    lines = ["=" * W, "Table 1 – Speed estimate standard deviations (m/s)", "=" * W]

    if mc is None:
        lines.append(f"{'Method':<38} {'v_delta':>14} {'v_L':>14}")
        lines.append("-" * W)
        lines.append(f"{'Theoretical — position (Eq 33,41)':<38} {sigmas['sigma_vdelta_pos']:>14.4f} {sigmas['sigma_vL_pos']:>14.4f}")
        lines.append(f"{'Theoretical — Doppler  (Eq 57,52)':<38} {sigmas['sigma_vdelta_doppler']:>14.4f} {sigmas['sigma_vL_doppler']:>14.4f}")
        lines.append("-" * W)
        lines.append(f"{'Baseline 2-pt diff.   (Eq 34)':<38} {sigmas['sigma_v2pt_baseline']:>14.4f} {sigmas['sigma_v2pt_baseline']:>14.4f}")
    else:
        lines.append(f"{'Method':<38} {'v_delta theo':>13} {'v_delta emp':>13} {'v_L theo':>13} {'v_L emp':>13}")
        lines.append("-" * W)
        lines.append(f"{'Position meas. (Eq 33,41)':<38} "
                     f"{sigmas['sigma_vdelta_pos']:>13.4f} {mc['empirical_std_delta']:>13.4f} "
                     f"{sigmas['sigma_vL_pos']:>13.4f} {mc['empirical_std_L']:>13.4f}")
        lines.append(f"{'Doppler meas. (Eq 57,52)':<38} "
                     f"{sigmas['sigma_vdelta_doppler']:>13.4f} {'N/A':>13} "
                     f"{sigmas['sigma_vL_doppler']:>13.4f} {'N/A':>13}")
        lines.append("-" * W)
        lines.append(f"{'Baseline 2-pt diff. (Eq 34)':<38} "
                     f"{sigmas['sigma_v2pt_baseline']:>13.4f} {'N/A':>13} "
                     f"{sigmas['sigma_v2pt_baseline']:>13.4f} {'N/A':>13}")
        lines.append("")
        lines.append(f"  MC trials = {mc['n_trials']}   |   "
                     f"mean v_delta = {mc['empirical_mean_delta']:.4f} m/s  (true = {p.v_delta:.2f})   |   "
                     f"mean v_L = {mc['empirical_mean_L']:.4f} m/s  (true = {p.v_L:.2f})")

    lines += [
        "",
        f"  phi_cLOS = {sigmas['phi_cLOS']:.6f}",
        "",
        "-" * W,
        "Single-run MLE (seed=0):",
        f"  v_delta: true={p.v_delta:.2f}  est={mle['v_hat_delta']:.4f}  err={mle['v_hat_delta']-p.v_delta:+.4f} m/s",
        f"  v_L:     true={p.v_L:.2f}  est={mle['v_hat_L']:.4f}  err={mle['v_hat_L']-p.v_L:+.4f} m/s",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Carrier deploying objects simulation")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--mc-trials", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = Params(seed=args.seed)

    sim    = run_sim(p)
    sigmas = compute_table1_sigmas(p, sim)
    mle    = compute_mle_estimates(p, sim)
    r2     = compute_R2(p, sim)

    print(f"Running Monte Carlo ({args.mc_trials} trials)...")
    mc = run_monte_carlo(p, n_trials=args.mc_trials)

    np.savez(str(outdir / "arrays.npz"), **{
        k: v for k, v in sim.items()
        if isinstance(v, np.ndarray)
    })

    plot_R1(outdir, sim)
    plot_R3(outdir, sim)
    plot_mc(outdir, p, mc, sigmas)

    out_r2 = print_R2(p, r2)
    out_t1 = print_table1(p, sigmas, mle, mc=mc)
    out_mc = print_mc(p, mc, sigmas)

    report = out_r2 + "\n" + out_t1 + "\n" + out_mc

    print(report)

    with open(outdir / "results.txt", "w") as f:
        f.write(report)

    print(f"\nFiles saved to: {outdir}/")
    print("  R1.png  – az/el plot (true + noisy)")
    print("  R3.png  – 3D positions")
    print("  MC.png  – Monte Carlo histograms")
    print("  results.txt")
    print("  arrays.npz")


if __name__ == "__main__":
    main()
