#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def cart2sph_ENU(xyz: np.ndarray) -> np.ndarray:
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    r = np.sqrt(x * x + y * y + z * z)
    rho = np.sqrt(x * x + y * y)

    az = np.arctan2(x, y)  # clockwise from North (ENU)
    el = np.arctan2(z, rho)

    return np.stack([r, az, el], axis=-1)


def sph2cart_ENU(rae: np.ndarray) -> np.ndarray:
    r = rae[..., 0]
    az = rae[..., 1]
    el = rae[..., 2]

    rho = r * np.cos(el)
    z = r * np.sin(el)
    x = rho * np.sin(az)
    y = rho * np.cos(az)

    return np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# Rotation matrices  (Eqs. 7–10)
# ---------------------------------------------------------------------------


def A1(alpha_c: float) -> np.ndarray:
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])


def A2(eps_c: float) -> np.ndarray:
    ce, se = np.cos(eps_c), np.sin(eps_c)
    return np.array([[1.0, 0.0, 0.0], [0.0, se, ce], [0.0, -ce, se]])


def A_total(alpha_c: float, eps_c: float) -> np.ndarray:
    # Eq. (10)
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    ce, se = np.cos(eps_c), np.sin(eps_c)

    return np.array(
        [
            [ca, sa * se, sa * ce],
            [-sa, ca * se, ca * ce],
            [0.0, -ce, se],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Cartesian noise MSE via linearised Jacobian
# ---------------------------------------------------------------------------


def cartesian_noise_MSE(
    r: float, az: float, el: float, sigma_r: float, sigma_az: float, sigma_el: float
) -> float:
    cos_el = np.cos(el)
    sin_el = np.sin(el)
    cos_az = np.cos(az)
    sin_az = np.sin(az)

    # Jacobian of sph2cart at (r, az, el)
    J = np.array(
        [
            [cos_el * sin_az, r * cos_el * cos_az, -r * sin_el * sin_az],
            [cos_el * cos_az, -r * cos_el * sin_az, -r * sin_el * cos_az],
            [sin_el, 0.0, r * cos_el],
        ]
    )

    R_sph = np.diag([sigma_r**2, sigma_az**2, sigma_el**2])
    R_xyz = J @ R_sph @ J.T
    return float(np.trace(R_xyz))


# ---------------------------------------------------------------------------
# Simulation parameters  (Section 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    n: int = 12
    t0: float = 0.0
    t_rho: float = 1.0
    g: float = 9.81

    v_delta: float = 20.0  # lateral speed
    v_L: float = 200.0  # longitudinal speed

    xc_m: tuple = (40e3, 40e3, 40e3)
    vc_mps: tuple = (-3.2e3, 0.0, -3.0e3)

    alpha_c_deg: float = -90.0
    eps_c_deg: float = -43.0

    # Measurement noise
    sigma_r: float = 5.0
    sigma_az: float = 0.15e-3
    sigma_el: float = 0.15e-3
    sigma_D: float = 1.0

    # Carrier track errors (Eqs. 40–41, 51–52)
    sigma_p: float = 5.0
    sigma_vc: float = 5.0

    me_over_mc: float = 0.1
    seed: int = 0


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_sim(p: Params) -> dict:
    rng = np.random.default_rng(p.seed)

    n = p.n
    dt = p.t_rho - p.t0

    xc = np.array(p.xc_m, dtype=float)
    vc = np.array(p.vc_mps, dtype=float)

    alpha_c = np.deg2rad(p.alpha_c_deg)
    eps_c = np.deg2rad(p.eps_c_deg)

    # Unit circle of dispersion directions
    angles = 2.0 * np.pi * np.arange(1, n + 1) / n
    v_unit_circle = np.stack(
        [np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=1
    )

    # Rotate into plane orthogonal to carrier velocity (Eq. 12)
    A = A_total(alpha_c, eps_c)
    A_inv = A.T
    v_unit_disp = (A @ v_unit_circle.T).T

    vc_norm = np.linalg.norm(vc)
    one_vc = vc / vc_norm

    v_disp = p.v_delta * v_unit_disp
    v_long = p.v_L * one_vc

    # Object velocities and positions (Eqs. 13–16)
    v_obj0 = vc[None, :] + v_disp + v_long[None, :]
    grav_term = np.array([0.0, 0.0, -p.g]) * (dt**2) / 2.0
    x_obj_true = xc[None, :] + v_obj0 * dt + grav_term[None, :]

    # Carrier after ejection (Eq. 43)
    vce = (vc_norm - p.me_over_mc * p.v_L) / (1.0 - p.me_over_mc)
    v_car_after = vce * one_vc
    x_car_true = xc + v_car_after * dt + grav_term

    # Spherical measurements (Eq. 17)
    u_obj_true = cart2sph_ENU(x_obj_true)

    # Add measurement noise
    noise = np.stack(
        [
            rng.normal(0.0, p.sigma_r, size=n),
            rng.normal(0.0, p.sigma_az, size=n),
            rng.normal(0.0, p.sigma_el, size=n),
        ],
        axis=1,
    )
    u_obj_noisy = u_obj_true + noise

    # Back to Cartesian (Eq. 22)
    z_obj_true = sph2cart_ENU(u_obj_true)
    z_obj_noisy = sph2cart_ENU(u_obj_noisy)

    # Object centroid (Eq. 24)
    z0_true = z_obj_true.mean(axis=0)
    z0_noisy = z_obj_noisy.mean(axis=0)

    # Rotate into circular frame (Eq. 25)
    y_true = (A_inv @ (z_obj_true - z0_true).T).T
    y_noisy = (A_inv @ (z_obj_noisy - z0_noisy).T).T

    # Noisy carrier estimates (Fix 2)
    xc_hat = xc + rng.normal(0.0, p.sigma_p / np.sqrt(3), size=3)
    vc_hat_norm = np.linalg.norm(vc) + rng.normal(0.0, p.sigma_vc)
    # vc_hat_vec = vc + rng.normal(0.0, p.sigma_vc / np.sqrt(3), size=3)
    # vc_hat_norm = np.linalg.norm(vc_hat_vec)

    # LOS geometry
    one_LOS_carrier = xc / np.linalg.norm(xc)
    one_LOS_obj = x_obj_true / np.linalg.norm(x_obj_true, axis=1, keepdims=True)

    phi_disp = np.sum(v_unit_disp * one_LOS_obj, axis=1)  # Eq. 53
    phi_cLOS = float(one_vc @ one_LOS_carrier)

    # Doppler measurements (Secs. 4.3–4.4)
    z_D_true = np.sum(v_obj0 * one_LOS_obj, axis=1)
    z_D_unresolved_true = float(np.mean(v_obj0 @ one_LOS_carrier))

    z_D_noisy = z_D_true + rng.normal(0.0, p.sigma_D, size=n)
    z_D_unresolved_noisy = z_D_unresolved_true + rng.normal(0.0, p.sigma_D)

    z_Dc_before_noisy = vc_norm * phi_cLOS + rng.normal(0.0, p.sigma_D)
    z_Dc_after_noisy = vce * phi_cLOS + rng.normal(0.0, p.sigma_D)

    # Representative Cartesian noise level sigma_w
    r_mean = float(u_obj_true[:, 0].mean())
    az_mean = float(u_obj_true[:, 1].mean())
    el_mean = float(u_obj_true[:, 2].mean())
    sigma_w = float(
        np.sqrt(
            cartesian_noise_MSE(
                r_mean, az_mean, el_mean, p.sigma_r, p.sigma_az, p.sigma_el
            )
        )
    )

    return {
        "params": asdict(p),
        "dt": dt,
        "A": A,
        "A_inv": A_inv,
        "one_vc": one_vc,
        "one_LOS_carrier": one_LOS_carrier,
        "one_LOS_obj": one_LOS_obj,
        "phi_cLOS": phi_cLOS,
        "phi_disp": phi_disp,
        "vc_norm": vc_norm,
        "v_unit_disp": v_unit_disp,
        "v_obj0": v_obj0,
        "x_obj_true": x_obj_true,
        "x_car_true": x_car_true,
        "u_obj_true": u_obj_true,
        "u_obj_noisy": u_obj_noisy,
        "z_obj_true": z_obj_true,
        "z_obj_noisy": z_obj_noisy,
        "z0_true": z0_true,
        "z0_noisy": z0_noisy,
        "y_true": y_true,
        "y_noisy": y_noisy,
        "z_D_true": z_D_true,
        "z_D_noisy": z_D_noisy,
        "z_D_unresolved_noisy": z_D_unresolved_noisy,
        "z_Dc_before_noisy": z_Dc_before_noisy,
        "z_Dc_after_noisy": z_Dc_after_noisy,
        "xc_hat": xc_hat,
        "vc_hat_norm": vc_hat_norm,
        "vce": vce,
        "sigma_w": sigma_w,
    }


# ---------------------------------------------------------------------------
# Table 1 – theoretical standard deviations  (Eqs. 33, 41, 52, 57)
# ---------------------------------------------------------------------------


def compute_table1_sigmas(p: Params, sim: dict) -> dict:
    dt = sim["dt"]
    n = p.n
    sigma_w = sim["sigma_w"]

    # From position measurements

    # FIX 1: Dispersion speed s.d. corrected formula (Eq. 33)
    # Correct: sigma_w^2 / (n * dt^2)   [was: 2*sigma_w^2/n / dt^2]
    sigma_vdelta_pos = np.sqrt(sigma_w**2 / (n * dt**2))

    # Longitudinal speed s.d. (Eq. 41)
    sigma_vL_pos = np.sqrt((sigma_w**2 / n + p.sigma_p**2) / dt**2 + p.sigma_vc**2)

    # Two-point differencing baseline
    sigma_v2pt = np.sqrt(2.0 * sigma_w**2 / dt**2)

    # ------------------------------------------------------------------
    # From Doppler measurements
    # ------------------------------------------------------------------

    phi_cLOS = sim["phi_cLOS"]

    # Long speed from Doppler (Eq. 52)
    sigma_vL_doppler = np.sqrt(p.sigma_D**2 / phi_cLOS**2 + p.sigma_vc**2)

    # v_delta from Doppler (Eq. 57) — using per-target phi_disp (Fix 3)
    phi_disp = sim["phi_disp"]  # already per-target (Fix 3)
    n_h = n // 2
    phi_half = phi_disp[:n_h]

    # Eq. 57
    sum_term = np.sum(2.0 * p.sigma_D**2 / phi_half**2)
    sigma_vdelta_doppler = np.sqrt(sum_term / n**2)

    return {
        "sigma_vdelta_pos": sigma_vdelta_pos,
        "sigma_vL_pos": sigma_vL_pos,
        "sigma_vdelta_doppler": sigma_vdelta_doppler,
        "sigma_vL_doppler": sigma_vL_doppler,
        "sigma_v2pt_baseline": sigma_v2pt,
        "phi_cLOS": phi_cLOS,
    }


def compute_mle_estimates(p: Params, sim: dict) -> dict:
    """
    Estimate v_delta and v_L using position-based ML estimators, sec. 4.1 and 4.2.
    Uses noisy carrier position/velocity estimates (Fix 2).
    """

    dt = sim["dt"]
    n = p.n
    n_half = n // 2
    y_noisy = sim["y_noisy"]

    # Eq. 32: objects are on opposite sides of circle
    diffs = y_noisy[:n_half] - y_noisy[n_half:]
    d_i = np.linalg.norm(diffs, axis=1)

    v_hat_delta = d_i.sum() / (n / 2) / (2.0 * dt)

    # v_L MLE Eq. 40 — use noisy carrier position/speed estimates (Fix 2)
    xc_hat = sim["xc_hat"]
    vc_hat_norm = sim["vc_hat_norm"]
    z0_noisy = sim["z0_noisy"]

    v_hat_L = np.linalg.norm(z0_noisy - xc_hat) / dt - vc_hat_norm  # Eq. 40

    return {
        "v_hat_delta": float(v_hat_delta),
        "v_hat_L": float(v_hat_L),
        "d_i": d_i,
    }


def compute_doppler_mle_estimates(p: Params, sim: dict) -> dict:
    """
    Estimate v_delta and v_L using Doppler-based ML estimators, sec. 4.3 and 4.4.
    Uses noisy carrier velocity estimate (Fix 2) and per-target LOS phi_disp (Fix 3).
    """
    n = p.n
    n_half = n // 2

    phi_cLOS = sim["phi_cLOS"]
    phi_disp = sim["phi_disp"]  # per-target phi values (Fix 3)
    z_D_noisy = sim["z_D_noisy"]
    vc_hat_norm = sim["vc_hat_norm"]  # noisy carrier speed estimate (Fix 2)

    # --- v_L from Doppler (Eq. 51) ---
    # Use the unresolved blob Doppler measurement (carrier LOS projection), Eq. 49/51
    z_D_unresolved_noisy = sim["z_D_unresolved_noisy"]
    v_hat_L_doppler = z_D_unresolved_noisy / phi_cLOS - vc_hat_norm  # Eq. 51

    # --- v_delta from Doppler (Eq. 56) — per-target phi (Fix 3) ---
    diffs_D = z_D_noisy[:n_half] - z_D_noisy[n_half:]  # shape (n/2,)
    phi_half = phi_disp[:n_half]

    v_hat_delta_doppler = float(np.sum(diffs_D / (2.0 * phi_half)) / (n / 2))

    return {
        "v_hat_delta_doppler": v_hat_delta_doppler,
        "v_hat_L_doppler": float(v_hat_L_doppler),
    }


def run_monte_carlo(p: Params, n_trials: int = 1000) -> dict:
    v_hat_deltas = np.zeros(n_trials)
    v_hat_Ls = np.zeros(n_trials)
    v_hat_deltas_D = np.zeros(n_trials)
    v_hat_Ls_D = np.zeros(n_trials)

    for k in range(n_trials):
        sim_k = run_sim(Params(**{**asdict(p), "seed": k}))

        # Position-based
        mle_k = compute_mle_estimates(p, sim_k)
        v_hat_deltas[k] = mle_k["v_hat_delta"]
        v_hat_Ls[k] = mle_k["v_hat_L"]

        # Doppler-based
        mle_D_k = compute_doppler_mle_estimates(p, sim_k)
        v_hat_deltas_D[k] = mle_D_k["v_hat_delta_doppler"]
        v_hat_Ls_D[k] = mle_D_k["v_hat_L_doppler"]

    return {
        # Position-based
        "v_hat_deltas": v_hat_deltas,
        "v_hat_Ls": v_hat_Ls,
        "empirical_mean_delta": float(v_hat_deltas.mean()),
        "empirical_mean_L": float(v_hat_Ls.mean()),
        "empirical_std_delta": float(v_hat_deltas.std()),
        "empirical_std_L": float(v_hat_Ls.std()),
        # Doppler-based
        "v_hat_deltas_D": v_hat_deltas_D,
        "v_hat_Ls_D": v_hat_Ls_D,
        "empirical_mean_delta_D": float(v_hat_deltas_D.mean()),
        "empirical_mean_L_D": float(v_hat_Ls_D.mean()),
        "empirical_std_delta_D": float(v_hat_deltas_D.std()),
        "empirical_std_L_D": float(v_hat_Ls_D.std()),
        "n_trials": n_trials,
    }


def plot_mc(outdir: Path, p: Params, mc: dict, sigmas: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    configs = [
        (
            axes[0, 0],
            mc["v_hat_deltas"],
            p.v_delta,
            sigmas["sigma_vdelta_pos"],
            "$\\hat{v}_\\delta$ (position)",
            "Position-based",
            r"$\hat{\sigma}_{\hat{v}_\delta}$",
            r"$\sigma_{\hat{v}_\delta}^{\mathrm{theo}}$",
        ),
        (
            axes[0, 1],
            mc["v_hat_Ls"],
            p.v_L,
            sigmas["sigma_vL_pos"],
            "$\\hat{v}_L$ (position)",
            "Position-based",
            r"$\hat{\sigma}_{\hat{v}_L}$",
            r"$\sigma_{\hat{v}_L}^{\mathrm{theo}}$",
        ),
        (
            axes[1, 0],
            mc["v_hat_deltas_D"],
            p.v_delta,
            sigmas["sigma_vdelta_doppler"],
            "$\\hat{v}_\\delta$ (Doppler)",
            "Doppler-based",
            r"$\hat{\sigma}_{\hat{v}_\delta}$",
            r"$\sigma_{\hat{v}_\delta}^{\mathrm{theo}}$",
        ),
        (
            axes[1, 1],
            mc["v_hat_Ls_D"],
            p.v_L,
            sigmas["sigma_vL_doppler"],
            "$\\hat{v}_L$ (Doppler)",
            "Doppler-based",
            r"$\hat{\sigma}_{\hat{v}_L}$",
            r"$\sigma_{\hat{v}_L}^{\mathrm{theo}}$",
        ),
    ]

    for ax, vals, true_val, theo_sigma, label, subtitle, emp_sym, theo_sym in configs:
        ax.hist(vals, bins=40, color="steelblue", alpha=0.7, density=True)
        ax.axvline(
            true_val, color="green", lw=1.5, ls="-", label=f"True = {true_val:.1f}"
        )
        ax.axvline(
            vals.mean(),
            color="tomato",
            lw=1.5,
            ls="--",
            label=f"Mean = {vals.mean():.2f}",
        )
        ax.axvline(
            true_val + theo_sigma,
            color="gray",
            lw=1.0,
            ls=":",
            label=theo_sym + f" = {theo_sigma:.3f}",
        )
        ax.axvline(true_val - theo_sigma, color="gray", lw=1.0, ls=":")

        ax.set_xlabel(f"{label} (m/s)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{subtitle} — {label}\n"
            + emp_sym
            + f" $= {vals.std():.4f}$,  "
            + theo_sym
            + f" $= {theo_sigma:.4f}$ m/s"
        )
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(outdir / "MC.png", dpi=200)
    plt.close(fig)


def print_mc(p: Params, mc: dict, sigmas: dict) -> str:
    lines = [
        "=" * 75,
        f"Monte Carlo validation  (N = {mc['n_trials']} trials)",
        "=" * 75,
        f"{'':<38} {'v_delta':>12} {'v_L':>12}",
        "-" * 75,
        f"{'True value (m/s)':<38} {p.v_delta:>12.4f} {p.v_L:>12.4f}",
        "",
        "  --- Position-based ---",
        f"{'MC mean (m/s)':<38} {mc['empirical_mean_delta']:>12.4f} {mc['empirical_mean_L']:>12.4f}",
        f"{'Empirical std (m/s)':<38} {mc['empirical_std_delta']:>12.4f} {mc['empirical_std_L']:>12.4f}",
        f"{'Theoretical std (m/s)':<38} {sigmas['sigma_vdelta_pos']:>12.4f} {sigmas['sigma_vL_pos']:>12.4f}",
        "",
        "  --- Doppler-based ---",
        f"{'MC mean (m/s)':<38} {mc['empirical_mean_delta_D']:>12.4f} {mc['empirical_mean_L_D']:>12.4f}",
        f"{'Empirical std (m/s)':<38} {mc['empirical_std_delta_D']:>12.4f} {mc['empirical_std_L_D']:>12.4f}",
        f"{'Theoretical std (m/s)':<38} {sigmas['sigma_vdelta_doppler']:>12.4f} {sigmas['sigma_vL_doppler']:>12.4f}",
        "-" * 75,
        "",
    ]
    return "\n".join(lines)


def compute_R2(p: Params, sim: dict) -> dict:
    u_obj_true = sim["u_obj_true"]
    z0_true = sim["z0_true"]
    x_car_true = sim["x_car_true"]

    ranges_obj = u_obj_true[:, 0]
    range_centre = np.linalg.norm(z0_true)
    range_carrier = np.linalg.norm(x_car_true)

    return {
        "ranges_obj": ranges_obj,
        "range_centre": range_centre,
        "range_carrier": range_carrier,
        "vce": sim["vce"],
    }


def plot_R1(outdir: Path, sim: dict) -> None:
    """2D az/el view of true vs noisy object positions."""

    u_obj_true = sim["u_obj_true"]
    u_obj_noisy = sim["u_obj_noisy"]

    az_true = u_obj_true[:, 1] * 1e3
    el_true = u_obj_true[:, 2] * 1e3

    az_noisy = u_obj_noisy[:, 1] * 1e3
    el_noisy = u_obj_noisy[:, 2] * 1e3

    z0_true = sim["z0_true"]
    u_z0_true = cart2sph_ENU(z0_true)
    az_ctr = u_z0_true[1] * 1e3
    el_ctr = u_z0_true[2] * 1e3

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(len(az_true)):
        ax.plot(
            [az_true[i], az_noisy[i]],
            [el_true[i], el_noisy[i]],
            color="gray",
            linewidth=0.7,
            alpha=0.6,
        )

    ax.scatter(
        az_true, el_true, marker="o", color="steelblue", s=50, label="True positions"
    )
    ax.scatter(
        az_noisy, el_noisy, marker="+", color="tomato", s=80, label="Noisy positions"
    )
    ax.scatter(az_ctr, el_ctr, marker="*", s=150, color="green", label="Centre (true)")

    for i in range(len(az_true)):
        ax.annotate(
            str(i + 1),
            (az_true[i], el_true[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="steelblue",
        )

    all_az = np.concatenate([az_true, az_noisy, [az_ctr]])
    all_el = np.concatenate([el_true, el_noisy, [el_ctr]])

    pad_az = 0.10 * (all_az.max() - all_az.min())
    pad_el = 0.10 * (all_el.max() - all_el.min())

    ax.set_xlim(all_az.min() - pad_az, all_az.max() + pad_az)
    ax.set_ylim(all_el.min() - pad_el, all_el.max() + pad_el)

    ax.set_xlabel("Azimuth (mrad)")
    ax.set_ylabel("Elevation (mrad)")
    #   ax.set_title(
    #       "R1 – Ejected objects at $t_\\rho = 1\\,$s\n"
    #       "(true = circles, noisy = crosses, centre = star)"
    #   )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(outdir / "R1.png", dpi=200)
    plt.close(fig)


def plot_R3(outdir: Path, sim: dict) -> None:
    """3D positions of objects and carrier at t_rho."""

    x_obj = sim["x_obj_true"]
    x_car = sim["x_car_true"]

    x_obj_km = x_obj / 1e3
    x_car_km = x_car / 1e3

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x_obj_km[:, 0],
        x_obj_km[:, 1],
        x_obj_km[:, 2],
        s=60,
        marker="o",
        color="steelblue",
        label="Ejected objects",
    )

    ax.scatter(
        [x_car_km[0]],
        [x_car_km[1]],
        [x_car_km[2]],
        s=120,
        marker="X",
        color="tomato",
        label="Carrier",
    )

    for i in range(len(x_obj_km)):
        ax.text(x_obj_km[i, 0], x_obj_km[i, 1], x_obj_km[i, 2], f" {i+1}", fontsize=7)

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Up (km)")
    #   ax.set_title(
    #       "R3 – 3D positions at $t_\\rho = 1\\,$s\n" "(carrier = X, objects = circles)"
    #   )
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / "R3.png", dpi=200)
    plt.close(fig)


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


def print_table1(p: Params, sigmas: dict, mle: dict, mle_D: dict, mc: dict) -> str:
    W = 80
    lines = ["=" * W, "Table 1 – Speed estimate standard deviations (m/s)", "=" * W]

    lines.append(
        f"{'Method':<38} {'v_delta theo':>13} {'v_delta emp':>13} {'v_L theo':>13} {'v_L emp':>13}"
    )
    lines.append("-" * W)
    lines.append(
        f"{'Position meas. (Eq 33,41)':<38} "
        f"{sigmas['sigma_vdelta_pos']:>13.4f} {mc['empirical_std_delta']:>13.4f} "
        f"{sigmas['sigma_vL_pos']:>13.4f} {mc['empirical_std_L']:>13.4f}"
    )
    lines.append(
        f"{'Doppler meas. (Eq 57,52)':<38} "
        f"{sigmas['sigma_vdelta_doppler']:>13.4f} {mc['empirical_std_delta_D']:>13.4f} "
        f"{sigmas['sigma_vL_doppler']:>13.4f} {mc['empirical_std_L_D']:>13.4f}"
    )
    lines.append("-" * W)
    lines.append(
        f"{'Baseline 2-pt diff. (Eq 34)':<38} "
        f"{sigmas['sigma_v2pt_baseline']:>13.4f} {'N/A':>13} "
        f"{sigmas['sigma_v2pt_baseline']:>13.4f} {'N/A':>13}"
    )
    lines += [
        "",
        f"  MC trials = {mc['n_trials']}   |   "
        f"mean v_delta = {mc['empirical_mean_delta']:.4f} m/s  (true = {p.v_delta:.2f})   |   "
        f"mean v_L = {mc['empirical_mean_L']:.4f} m/s  (true = {p.v_L:.2f})",
        f"  Doppler MC: "
        f"mean v_delta_D = {mc['empirical_mean_delta_D']:.4f} m/s   |   "
        f"mean v_L_D = {mc['empirical_mean_L_D']:.4f} m/s",
        "",
        f"  phi_cLOS = {sigmas['phi_cLOS']:.6f}",
        "",
        "-" * W,
        "Single-run MLE (seed=0):",
        f"  [Position] v_delta: true={p.v_delta:.2f}  est={mle['v_hat_delta']:.4f}  err={mle['v_hat_delta']-p.v_delta:+.4f} m/s",
        f"  [Position] v_L:     true={p.v_L:.2f}  est={mle['v_hat_L']:.4f}  err={mle['v_hat_L']-p.v_L:+.4f} m/s",
        f"  [Doppler]  v_delta: true={p.v_delta:.2f}  est={mle_D['v_hat_delta_doppler']:.4f}  err={mle_D['v_hat_delta_doppler']-p.v_delta:+.4f} m/s",
        f"  [Doppler]  v_L:     true={p.v_L:.2f}  est={mle_D['v_hat_L_doppler']:.4f}  err={mle_D['v_hat_L_doppler']-p.v_L:+.4f} m/s",
        "",
        "CORRECTIONS APPLIED:",
        "  Fix 1: Eq.(33) sigma_vdelta_pos = sigma_w / (sqrt(n) * dt)  [was sqrt(2)*sigma_w/(sqrt(n)*dt)]",
        "  Fix 2: Carrier position/velocity estimates include Gaussian noise (sigma_p, sigma_vc)",
        "  Fix 3: Per-target LOS vectors used in Eq.(53) phi_disp[i] = 1_vdi . 1_LOS,i",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Carrier deploying objects simulation")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mc-trials", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = Params(seed=args.seed)

    sim = run_sim(p)
    sigmas = compute_table1_sigmas(p, sim)
    mle = compute_mle_estimates(p, sim)
    mle_D = compute_doppler_mle_estimates(p, sim)
    r2 = compute_R2(p, sim)

    print(f"Running Monte Carlo ({args.mc_trials} trials)...")
    mc = run_monte_carlo(p, n_trials=args.mc_trials)

    np.savez(
        str(outdir / "arrays.npz"),
        **{k: v for k, v in sim.items() if isinstance(v, np.ndarray)},
    )

    plot_R1(outdir, sim)
    plot_R3(outdir, sim)
    plot_mc(outdir, p, mc, sigmas)

    out_r2 = print_R2(p, r2)
    out_t1 = print_table1(p, sigmas, mle, mle_D, mc=mc)
    out_mc = print_mc(p, mc, sigmas)

    report = out_r2 + "\n" + out_t1 + "\n" + out_mc

    print(report)

    with open(outdir / "results.txt", "w") as f:
        f.write(report)

    print(f"\nFiles saved to: {outdir}/")
    print("  R1.png  – az/el plot (true + noisy)")
    print("  R3.png  – 3D positions")
    print("  MC.png  – Monte Carlo histograms (position + Doppler)")
    print("  results.txt")
    print("  arrays.npz")
    print(f"sigma_w = {sim['sigma_w']:.4f} m")


if __name__ == "__main__":
    main()
