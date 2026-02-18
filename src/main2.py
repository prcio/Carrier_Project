from __future__ import annotations
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
