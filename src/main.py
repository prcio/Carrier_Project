#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Coordinate transforms (ENU)
# -----------------------------
def cart2sph_ENU(xyz: np.ndarray) -> np.ndarray:
    """
    ENU Cartesian -> spherical [r, az, el]
      az: clockwise from North (ENU: x=East, y=North)
      el: elevation above horizontal plane (ENU)
    """
    x = xyz[..., 0]  # East
    y = xyz[..., 1]  # North
    z = xyz[..., 2]  # Up

    r = np.sqrt(x * x + y * y + z * z)
    rho = np.sqrt(x * x + y * y)

    # azimuth clockwise from North:
    az = np.arctan2(x, y)
    # elevation above horizontal:
    el = np.arctan2(z, rho)

    return np.stack([r, az, el], axis=-1)


def sph2cart_ENU(rae: np.ndarray) -> np.ndarray:
    """
    Spherical [r, az, el] -> ENU Cartesian consistent with cart2sph_ENU.
    """
    r = rae[..., 0]
    az = rae[..., 1]
    el = rae[..., 2]

    rho = r * np.cos(el)
    z = r * np.sin(el)

    x = rho * np.sin(az)  # East
    y = rho * np.cos(az)  # North

    return np.stack([x, y, z], axis=-1)


# -----------------------------
# Rotations (paper eq 7–10)
# -----------------------------
def A1(alpha_c: float) -> np.ndarray:
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def A2(eps_c: float) -> np.ndarray:
    ce, se = np.cos(eps_c), np.sin(eps_c)
    return np.array([[1.0, 0.0, 0.0], [0.0, se, ce], [0.0, -ce, se]], dtype=float)


def A_total(alpha_c: float, eps_c: float) -> np.ndarray:
    # paper eq (9): 1vδi = A2 * A1 * 1vi
    A = A2(eps_c) @ A1(alpha_c)
    return A


# -----------------------------
# Parameters
# -----------------------------
@dataclass(frozen=True)
class Params:
    # Scenario
    n: int = 12
    t0: float = 0.0
    t_rho: float = 1.0
    g: float = 9.81

    # Speeds
    v_delta: float = 20.0  # m/s lateral (dispersion)
    v_L: float = 200.0  # m/s longitudinal ejection speed

    # Carrier state at ejection (paper gives km and km/s; we store meters, m/s)
    xc_m: tuple[float, float, float] = (40e3, 40e3, 40e3)
    vc_mps: tuple[float, float, float] = (-3.2e3, 0.0, -3.0e3)

    # Carrier velocity pointing angles used for rotation (paper)
    alpha_c_deg: float = -90.0
    eps_c_deg: float = -43.0

    # Radar measurement noise (paper Sec. 5)
    sigma_r: float = 5.0  # m
    sigma_az: float = 0.15e-3  # rad
    sigma_el: float = 0.15e-3  # rad

    # “Position measurement RMSE” used in Table 1 formulas (paper lists sigma_w = 10m)
    sigma_w: float = 10.0  # m

    # Carrier track uncertainties (paper Sec. 5)
    sigma_p: float = 5.0  # m (carrier position RMSE)
    sigma_vc: float = 5.0  # m/s (carrier speed RMSE)

    # Doppler noise (paper Sec. 5)
    sigma_D: float = 1.0  # m/s

    # Momentum fraction (paper Sec. 5)
    me_over_mc: float = 0.1

    # Random seed
    seed: int = 0


# -----------------------------
# Core simulation (Sec. 6.1 A)
# -----------------------------
def run_sim(p: Params):
    rng = np.random.default_rng(p.seed)

    n = p.n
    dt = p.t_rho - p.t0

    xc = np.array(p.xc_m, dtype=float)
    vc = np.array(p.vc_mps, dtype=float)

    alpha_c = np.deg2rad(p.alpha_c_deg)
    eps_c = np.deg2rad(p.eps_c_deg)

    # (Option A step 1) unit dispersion vectors around radar (paper eq 6 with eq 4)
    angles = 2.0 * np.pi * np.arange(1, n + 1) / n
    v_unit_circle = np.stack(
        [np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=1
    )

    # (Option A step 2) rotate around carrier (paper eq 9)
    A = A_total(alpha_c, eps_c)
    v_unit_disp_carrier = (A @ v_unit_circle.T).T  # (n,3)

    # sanity: A is orthogonal -> inv = A.T
    A_inv = A.T

    # carrier direction unit vector (from vc vector)
    vc_norm = np.linalg.norm(vc)
    one_vc = vc / vc_norm

    # (Option A step 3) dispersion velocities with known v_delta (paper eq 12)
    v_disp = p.v_delta * v_unit_disp_carrier

    # longitudinal ejection velocity along carrier direction (paper eq 14)
    v_long = p.v_L * one_vc

    # object initial velocities (paper eq 13)
    v_obj0 = vc + v_disp + v_long  # (n,3)

    # gravity term for position propagation
    grav_term = np.array([0.0, 0.0, -p.g]) * (dt * dt) / 2.0

    # (Option A step 4) object positions at t_rho (paper eq 15)
    x_obj_true = xc + v_obj0 * dt + grav_term

    # carrier speed after ejection (paper eq 43)
    vce = (vc_norm - p.me_over_mc * p.v_L) / (1.0 - p.me_over_mc)
    v_car_after = vce * one_vc

    # (Option A step 5) carrier position at t_rho (paper eq 16, using vce)
    x_car_true = xc + v_car_after * dt + grav_term

    # (Option A step 6A) spherical measurements (zero noise)
    u_obj_true = cart2sph_ENU(x_obj_true)

    # (Option A step 6B) spherical measurements with noise (paper eq 19–21)
    noise = np.stack(
        [
            rng.normal(0.0, p.sigma_r, size=n),
            rng.normal(0.0, p.sigma_az, size=n),
            rng.normal(0.0, p.sigma_el, size=n),
        ],
        axis=1,
    )
    u_obj_noisy = u_obj_true + noise

    # (Option A step 7) convert spherical -> Cartesian (paper eq 22)
    z_obj_true_from_u = sph2cart_ENU(u_obj_true)  # should ~match x_obj_true
    z_obj_noisy = sph2cart_ENU(u_obj_noisy)

    # (Option A step 8) center (paper eq 24)
    z0_true = z_obj_true_from_u.mean(axis=0)
    z0_noisy = z_obj_noisy.mean(axis=0)

    # (Option A step 9) rotate centered measurements into circular shape (paper eq 25/26)
    y_true = (A_inv @ (z_obj_true_from_u - z0_true).T).T
    y_noisy = (A_inv @ (z_obj_noisy - z0_noisy).T).T

    # For R1 center in spherical (they want az/el of center)
    u_center_true = cart2sph_ENU(z0_true[None, :])[0]
    u_center_noisy = cart2sph_ENU(z0_noisy[None, :])[0]
    u_carrier_true = cart2sph_ENU(x_car_true[None, :])[0]

    return {
        "params": asdict(p),
        "A": A,
        "A_inv": A_inv,
        "v_unit_circle": v_unit_circle,
        "v_unit_disp_carrier": v_unit_disp_carrier,
        "v_disp": v_disp,
        "v_long": v_long,
        "v_obj0": v_obj0,
        "x_obj_true": x_obj_true,
        "x_car_true": x_car_true,
        "u_obj_true": u_obj_true,
        "u_obj_noisy": u_obj_noisy,
        "z_obj_true_from_u": z_obj_true_from_u,
        "z_obj_noisy": z_obj_noisy,
        "z0_true": z0_true,
        "z0_noisy": z0_noisy,
        "y_true": y_true,
        "y_noisy": y_noisy,
        "u_center_true": u_center_true,
        "u_center_noisy": u_center_noisy,
        "u_carrier_true": u_carrier_true,
        "vc_norm": vc_norm,
        "vce": vce,
        "one_vc": one_vc,
        "xc": xc,
        "vc": vc,
    }


# -----------------------------
# Table 1 sigma computations
# -----------------------------
def compute_table1_sigmas(p: Params, sim: dict) -> dict[str, float]:
    """
    Computes Table 1 standard deviations (m/s) using the paper’s formulas, in a dimensionally consistent way.

    - Position-based:
        sigma_vdelta_pos ~ sqrt( (2*sigma_w^2/n) / dt^2 )
        sigma_vL_pos ~ sqrt( (sigma_w^2/n + sigma_p^2)/dt^2 + sigma_vc^2 )

    - Doppler-based:
        sigma_vL_dopp ~ sqrt( sigma_D^2 / phi_cLOS^2 + sigma_vc^2 )
        sigma_vdelta_dopp ~ sqrt( (1/n^2) * sum_{i=1..n/2} (2*sigma_D^2)/(phi_delta_i_LOS^2) )

    NOTE:
    LOS is taken to the carrier at ejection position xc (consistent with the “at ejection” geometry).
    """
    dt = p.t_rho - p.t0
    n = p.n

    # LOS to carrier at ejection
    xc = sim["xc"]
    one_LOS = xc / np.linalg.norm(xc)

    one_vc = sim["one_vc"]
    phi_cLOS = float(one_vc @ one_LOS)

    # dispersion unit vectors at carrier (eq 9 output)
    one_vdelta_i = sim["v_unit_disp_carrier"]

    # phi_delta_i_LOS (eq 53)
    phi_delta = (one_vdelta_i @ one_LOS).astype(float)  # (n,)

    # opposite pairs i and i+n/2, and for perfect symmetry phi(i) = -phi(i+n/2)
    # in sigma formula (57) we need i = 1..n/2
    phi_half = phi_delta[: n // 2]

    # Position-based sigmas
    sigma_vdelta_pos = np.sqrt((2.0 * p.sigma_w**2 / n) / (dt**2))
    sigma_vL_pos = np.sqrt(
        ((p.sigma_w**2 / n) + p.sigma_p**2) / (dt**2) + p.sigma_vc**2
    )

    # Doppler-based sigmas
    # Guard against pathological phi_cLOS ~ 0
    eps = 1e-12
    sigma_vL_dopp = np.sqrt((p.sigma_D**2) / (max(phi_cLOS**2, eps)) + p.sigma_vc**2)

    # Guard against tiny phi_delta components; in a real scenario you’d handle geometry/visibility.
    denom = np.maximum(phi_half**2, eps)
    sigma_vdelta_dopp = np.sqrt((1.0 / (n**2)) * np.sum((2.0 * p.sigma_D**2) / denom))

    # Baseline two-point differencing overall speed sd (paper compares to (34)); dimensionally consistent:
    sigma_v_2pt = np.sqrt((2.0 * p.sigma_w**2) / (dt**2))

    return {
        "sigma_vdelta_from_position": float(sigma_vdelta_pos),
        "sigma_vL_from_position": float(sigma_vL_pos),
        "sigma_vdelta_from_doppler": float(sigma_vdelta_dopp),
        "sigma_vL_from_doppler": float(sigma_vL_dopp),
        "sigma_v_baseline_2pt": float(sigma_v_2pt),
        "phi_cLOS": float(phi_cLOS),
    }


# -----------------------------
# R1–R3 outputs
# -----------------------------
def save_ranges_table(outdir: Path, sim: dict):
    u_obj_true = sim["u_obj_true"]
    z0_true = sim["z0_true"]
    x_car_true = sim["x_car_true"]
    vce = sim["vce"]

    u_center = cart2sph_ENU(z0_true[None, :])[0]
    u_car = cart2sph_ENU(x_car_true[None, :])[0]

    # CSV: index, range_m
    lines = ["item,range_m"]
    for i, r in enumerate(u_obj_true[:, 0], start=1):
        lines.append(f"object_{i},{r:.6f}")
    lines.append(f"center,{u_center[0]:.6f}")
    lines.append(f"carrier,{u_car[0]:.6f}")
    lines.append(f"carrier_speed_after_vce_mps,{vce:.6f}")

    (outdir / "ranges_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_R1(outdir: Path, sim: dict):
    # True az/el of objects at t_rho, plus center
    u_obj_true = sim["u_obj_true"]
    u_center_true = sim["u_center_true"]

    # Convert to mrad for plot scale
    az_mrad = u_obj_true[:, 1] * 1e3
    el_mrad = u_obj_true[:, 2] * 1e3
    azc_mrad = u_center_true[1] * 1e3
    elc_mrad = u_center_true[2] * 1e3

    plt.figure()
    plt.scatter(az_mrad, el_mrad)
    plt.scatter([azc_mrad], [elc_mrad], marker="x")
    plt.xlabel("Azimuth (mrad)")
    plt.ylabel("Elevation (mrad)")
    plt.title("R1: True az-el of ejected objects at t_rho (ellipse)")
    plt.tight_layout()
    plt.savefig(outdir / "R1_az_el_true.png", dpi=200)
    plt.close()

    # Overlay noisy positions too (requested in “Noisy simulations (i)”)
    u_obj_noisy = sim["u_obj_noisy"]
    azn_mrad = u_obj_noisy[:, 1] * 1e3
    eln_mrad = u_obj_noisy[:, 2] * 1e3

    plt.figure()
    plt.scatter(az_mrad, el_mrad, label="true")
    plt.scatter(azn_mrad, eln_mrad, label="noisy", alpha=0.6)
    plt.scatter([azc_mrad], [elc_mrad], marker="x", label="center(true)")
    plt.xlabel("Azimuth (mrad)")
    plt.ylabel("Elevation (mrad)")
    plt.title("R1: True + noisy az-el overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "R1_az_el_true_with_noisy_overlay.png", dpi=200)
    plt.close()


def plot_R3(outdir: Path, sim: dict):
    x_obj = sim["x_obj_true"]
    x_car = sim["x_car_true"]
    xc0 = sim["xc"]  # carrier at ejection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_obj[:, 0], x_obj[:, 1], x_obj[:, 2], label="objects @ t_rho")
    ax.scatter([x_car[0]], [x_car[1]], [x_car[2]], marker="x", label="carrier @ t_rho")
    ax.scatter([xc0[0]], [xc0[1]], [xc0[2]], marker="^", label="carrier @ t0")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("R3: 3D geometry at t_rho")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "R3_3d_geometry.png", dpi=200)
    plt.close(fig)


def save_table1(outdir: Path, sigmas: dict):
    lines = ["quantity,value"]
    for k, v in sigmas.items():
        lines.append(f"{k},{v:.12g}")
    (outdir / "table1_sigmas.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument(
        "--seed", type=int, default=0, help="Random seed for noisy measurements"
    )

    # Optional overrides if you need them quickly:
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--t_rho", type=float, default=1.0)
    ap.add_argument("--v_delta", type=float, default=20.0)
    ap.add_argument("--v_L", type=float, default=200.0)
    ap.add_argument("--sigma_r", type=float, default=5.0)
    ap.add_argument(
        "--sigma_az_mrad", type=float, default=0.15, help="az std dev in mrad"
    )
    ap.add_argument(
        "--sigma_el_mrad", type=float, default=0.15, help="el std dev in mrad"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = Params(
        n=args.n,
        t_rho=args.t_rho,
        v_delta=args.v_delta,
        v_L=args.v_L,
        sigma_r=args.sigma_r,
        sigma_az=args.sigma_az_mrad * 1e-3,
        sigma_el=args.sigma_el_mrad * 1e-3,
        seed=args.seed,
    )

    sim = run_sim(p)
    sigmas = compute_table1_sigmas(p, sim)

    # Save arrays for inspection / downstream work
    np.savez(
        outdir / "arrays_true_and_meas.npz",
        **{k: v for k, v in sim.items() if k != "params"},
        params=np.array([str(sim["params"])], dtype=object),
    )

    # R1–R3 + tables
    plot_R1(outdir, sim)
    plot_R3(outdir, sim)
    save_ranges_table(outdir, sim)
    save_table1(outdir, sigmas)

    # Console summary
    print(f"Wrote outputs to: {outdir.resolve()}")
    print("Key outputs:")
    print("  - arrays_true_and_meas.npz")
    print("  - ranges_table.csv")
    print("  - table1_sigmas.csv")
    print("  - R1_az_el_true.png")
    print("  - R1_az_el_true_with_noisy_overlay.png")
    print("  - R3_3d_geometry.png")
    print("")
    print("Carrier speed after ejection vce (m/s):", sim["vce"])
    print("Table1 sigmas (m/s):")
    for k in [
        "sigma_vdelta_from_position",
        "sigma_vL_from_position",
        "sigma_vdelta_from_doppler",
        "sigma_vL_from_doppler",
        "sigma_v_baseline_2pt",
    ]:
        print(f"  {k}: {sigmas[k]}")


if __name__ == "__main__":
    main()
