#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ENU <-> spherical (pls dont touch unless you like debugging trig)
def cart2sph_ENU(xyz: np.ndarray) -> np.ndarray:
    x = xyz[..., 0]  # East
    y = xyz[..., 1]  # North
    z = xyz[..., 2]  # Up

    r = np.sqrt(x * x + y * y + z * z)
    rho = np.sqrt(x * x + y * y)

    az = np.arctan2(x, y)  # cw from North
    el = np.arctan2(z, rho)  # above horiz

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


# rotations from the pdf. yes i know.
def A1(alpha_c: float) -> np.ndarray:
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])


def A2(eps_c: float) -> np.ndarray:
    ce, se = np.cos(eps_c), np.sin(eps_c)
    return np.array([[1.0, 0.0, 0.0], [0.0, se, ce], [0.0, -ce, se]])


def A_total(alpha_c: float, eps_c: float) -> np.ndarray:
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    ce, se = np.cos(eps_c), np.sin(eps_c)

    # Eq (10) from the paper (rotA). Columns are orth, last col = 1_vc
    A = np.array(
        [
            [ca, sa * se, sa * ce],
            [-sa, ca * se, ca * ce],
            [0.0, -ce, se],
        ],
        dtype=float,
    )
    return A


@dataclass(frozen=True)
class Params:
    n: int = 12
    t0: float = 0.0
    t_rho: float = 1.0
    g: float = 9.81

    v_delta: float = 20.0
    v_L: float = 200.0

    xc_m: tuple[float, float, float] = (40e3, 40e3, 40e3)
    vc_mps: tuple[float, float, float] = (-3.2e3, 0.0, -3.0e3)

    alpha_c_deg: float = -90.0
    eps_c_deg: float = -43.0

    sigma_r: float = 5.0
    sigma_az: float = 0.15e-3
    sigma_el: float = 0.15e-3

    sigma_w: float = 10.0
    sigma_p: float = 5.0
    sigma_vc: float = 5.0
    sigma_D: float = 1.0

    me_over_mc: float = 0.1
    seed: int = 0


def run_sim(p: Params):
    rng = np.random.default_rng(p.seed)

    n = p.n
    dt = p.t_rho - p.t0

    xc = np.array(p.xc_m)
    vc = np.array(p.vc_mps)

    alpha_c = np.deg2rad(p.alpha_c_deg)
    eps_c = np.deg2rad(p.eps_c_deg)

    angles = 2 * np.pi * np.arange(1, n + 1) / n
    v_unit_circle = np.stack(
        [np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=1
    )

    A = A_total(alpha_c, eps_c)
    A_inv = A.T

    v_unit_disp = (A @ v_unit_circle.T).T

    vc_norm = np.linalg.norm(vc)
    one_vc = vc / vc_norm

    v_disp = p.v_delta * v_unit_disp
    v_long = p.v_L * one_vc

    v_obj0 = vc + v_disp + v_long

    grav_term = np.array([0.0, 0.0, -p.g]) * (dt * dt) / 2.0

    x_obj_true = xc + v_obj0 * dt + grav_term

    vce = (vc_norm - p.me_over_mc * p.v_L) / (1.0 - p.me_over_mc)
    v_car_after = vce * one_vc
    x_car_true = xc + v_car_after * dt + grav_term

    u_obj_true = cart2sph_ENU(x_obj_true)

    noise = np.stack(
        [
            rng.normal(0.0, p.sigma_r, size=n),
            rng.normal(0.0, p.sigma_az, size=n),
            rng.normal(0.0, p.sigma_el, size=n),
        ],
        axis=1,
    )
    u_obj_noisy = u_obj_true + noise

    z_obj_true = sph2cart_ENU(u_obj_true)
    z_obj_noisy = sph2cart_ENU(u_obj_noisy)

    z0_true = z_obj_true.mean(axis=0)
    z0_noisy = z_obj_noisy.mean(axis=0)

    y_true = (A_inv @ (z_obj_true - z0_true).T).T
    y_noisy = (A_inv @ (z_obj_noisy - z0_noisy).T).T

    return {
        "params": asdict(p),
        "x_obj_true": x_obj_true,
        "x_car_true": x_car_true,
        "u_obj_true": u_obj_true,
        "u_obj_noisy": u_obj_noisy,
        "z_obj_noisy": z_obj_noisy,
        "z0_true": z0_true,
        "y_true": y_true,
        "vce": vce,
    }


def compute_table1_sigmas(p: Params, sim: dict):
    dt = p.t_rho - p.t0
    n = p.n

    sigma_vdelta_pos = np.sqrt((2 * p.sigma_w**2 / n) / (dt**2))
    sigma_vL_pos = np.sqrt(
        ((p.sigma_w**2 / n) + p.sigma_p**2) / (dt**2) + p.sigma_vc**2
    )

    sigma_v_2pt = np.sqrt((2 * p.sigma_w**2) / (dt**2))

    return {
        "sigma_vdelta_from_position": sigma_vdelta_pos,
        "sigma_vL_from_position": sigma_vL_pos,
        "sigma_v_baseline_2pt": sigma_v_2pt,
    }


def plot_R1(outdir: Path, sim: dict):
    u_obj_true = sim["u_obj_true"]
    az = u_obj_true[:, 1] * 1e3
    el = u_obj_true[:, 2] * 1e3

    plt.figure()
    plt.scatter(az, el)
    plt.xlabel("Az (mrad)")
    plt.ylabel("El (mrad)")
    plt.title("R1 - ellipse thing")
    plt.tight_layout()
    plt.savefig(outdir / "R1.png", dpi=200)
    plt.close()


def plot_R3(outdir: Path, sim: dict):
    x_obj = sim["x_obj_true"]
    x_car = sim["x_car_true"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_obj[:, 0], x_obj[:, 1], x_obj[:, 2])
    ax.scatter([x_car[0]], [x_car[1]], [x_car[2]], marker="x")
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_zlabel("U")
    ax.set_title("3d mess")
    plt.tight_layout()
    plt.savefig(outdir / "R3.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = Params(seed=args.seed)
    sim = run_sim(p)
    sigmas = compute_table1_sigmas(p, sim)

    np.savez(outdir / "arrays.npz", **sim)

    plot_R1(outdir, sim)
    plot_R3(outdir, sim)

    print("done. files in", outdir)
    print("vce:", sim["vce"])
    print("sigmas:", sigmas)


if __name__ == "__main__":
    main()
