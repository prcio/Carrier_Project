#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib

from main import A_total, Params, sph2cart_ENU
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cart2sph_ENU(xyz: np.ndarray) -> np.ndarray
    x = xyz[ ..., 0]
    y = xyz[ ..., 1]
    z = xyz[ ..., 2]

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
# Rotation matrices  –  Eq. (7), (8), (10) 
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
    ca, sa = np.cos(alpha_c), np.sin(alpha_c)
    ce, se = np.cos(eps_c),   np.sin(eps_c)

    # Eq. (10)
    A = np.array([
        [ ca,  sa*se,  sa*ce],
        [-sa,  ca*se,  ca*ce],
        [ 0.,    -ce,    se ],
    ], dtype=float)
    return A



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

    n = p.n
    dt = p.t_rho - p.t0

    xc = np.array(p.xc_m, dtype=float)
    vc = np.array(p.vc_mps, dtype=float)

    alpha_c = np.deg2rad(p.alpha_c_deg)
    eps_c   = np.deg2rad(p.eps_c_deg)

    # Unit circle
    angles = 2.0 * np.pi * np.arange(1, n+1) / n 
    v_unit_circle = np.stack([
        np.sin(angles),
        np.cos(angles),
        np.zeros_like(angles),
    ], axis=1)

    # 2 - rotate to be orth. to v_c (Eq. 4, 6)
    A = A_total(alpha_c, eps_c)
    A_inv = A.T

    v_unit_disp = (A @ v_unit_circle.T).T

    vc_norm = np.linalg.norm(vc)
    one_vc = vc / vc_norm
    
    # 3 - disp velocities (Eq. 12)
    v_disp = p.v_delta * v_unit_disp
    
    # 4 - initial obj velocities & positions (Eq. 13, 14, 15)
    v_long = p.v_L * one_vc
    v_obj0 = vc [None, :] + v_disp + v_long[None, :]
    
    grav_term = np.array([0., 0., -p.g]) * (dt**2) / 2.0
    x_obj_true = xc[None, :] + v_obj0 * dt + grav_term[None, :]

    # 5 - carrier position after election (Eq. 16, 43)
    vce = (vc_norm - p.me_over_mc * p.v_L) / (1.0 - p.me_over_mc) # Eq. 43
    v_car_after = vce * one_vc
    x_car_true = xc + v_car_after * dt + grav_term

    # 6A - noiseless spherical measurements Eq. 17
    u_obj_true = cart2sph_ENU(x_obj_true)

    # 6B - adding noise
    noise = np.stack([
        rng.normal(0., p.sigma_r, size = n),
        rng.normal(0., p.sigma_az, size = n),
        rng.normal(0., p.sigma_el, size = n),
    ], axis = 1)
    u_obj_noisy = u_obj_true + noise
    
    # 7 - convert to cartesian (Eq. 22)
    z_obj_true = sph2cart_ENU(u_obj_true)
    z_obj_noisy = sph2cart_ENU(u_obj_noisy)

    # 8 - center of object measurements (Eq. 24)
    z0_true = z_obj_true.mean(axis=0)
    z0_noisy = z_obj_noisy.mean(axis=0)

