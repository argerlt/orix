#
# Copyright 2018-2026 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

"""Numba functions used in the quaternion module."""

# TODO: Move all Numba functions in the quaternion module here.
# The idea is to in the future allow Numba to be an optional dependency.

import numba as nb
import numpy as np

# --------------------------- orientation ---------------------------- #


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ori_angle_with_outer_sym(
    qu1: np.ndarray, qu2: np.ndarray, qu_sym: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Return pairwise symmetry-reduced misorientation angles.

    Avoids materializing the large intermediate misorientation and
    dot-product arrays that the NumPy path requires, reducing memory
    usage from O(n * m * (4 + s)) to O(n * m).

    Parameters
    ----------
    qu1
        Array of shape (n, 4) with unit quaternion components of the
        first set of orientations.
    qu2
        Array of shape (m, 4) with unit quaternion components of the
        second set of orientations.
    qu_sym
        Array of shape (s, 4) with unit quaternion components of the
        *proper* symmetry elements.

    Returns
    -------
    angles
        Array of shape (n, m) with pairwise misorientation angles in
        radians.
    """
    n = qu1.shape[0]
    m = qu2.shape[0]
    s = qu_sym.shape[0]
    out = np.empty((n, m), dtype=np.float64)

    for i in nb.prange(n):
        ai = qu1[i, 0]
        bi = qu1[i, 1]
        ci = qu1[i, 2]
        di = qu1[i, 3]

        for j in range(m):
            aj = qu2[j, 0]
            bj = qu2[j, 1]
            cj = qu2[j, 2]
            dj = qu2[j, 3]

            # Misorientation: m = q_j * conj(q_i),
            # conj(q_i) = (ai, -bi, -ci, -di)
            mw = aj * ai + bj * bi + cj * ci + dj * di
            mx = bj * ai - aj * bi + dj * ci - cj * di
            my = cj * ai - dj * bi - aj * ci + bj * di
            mz = dj * ai + cj * bi - bj * ci - aj * di

            # Find max |dot(m, s)| over all proper symmetry elements.
            # Improper elements are pre-filtered by the caller.
            max_dp = 0.0
            for k in range(s):
                dp = (
                    mw * qu_sym[k, 0]
                    + mx * qu_sym[k, 1]
                    + my * qu_sym[k, 2]
                    + mz * qu_sym[k, 3]
                )
                if dp < 0.0:
                    dp = -dp
                if dp > max_dp:
                    max_dp = dp

            # Clamp to [0, 1] to guard against floating-point overshoot
            if max_dp > 1.0:
                max_dp = 1.0

            out[i, j] = np.arccos(2.0 * max_dp * max_dp - 1.0)

    return out
