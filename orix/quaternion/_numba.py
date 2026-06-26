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

# TODO: Move all Numba functions in the quaternion module here. The
# idea is to in the future allow Numba to be an optional dependency.

import numba as nb
import numpy as np

# --------------------------- orientation ---------------------------- #


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ori_angle_with_outer_sym(
    qu1: np.ndarray, qu2: np.ndarray, sym_ops: np.ndarray
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
    sym_ops
        Array of shape (s, 4) with unit quaternion components of the
        *proper* symmetry operations.

    Returns
    -------
    angles
        Array of shape (n, m) with pairwise misorientation angles in
        radians.
    """
    n = qu1.shape[0]
    m = qu2.shape[0]
    s = sym_ops.shape[0]
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

            # Misorientation: M = q_j * conj(q_i),
            # conj(q_i) = (ai, -bi, -ci, -di)
            mw = aj * ai + bj * bi + cj * ci + dj * di
            mx = bj * ai - aj * bi + dj * ci - cj * di
            my = cj * ai - dj * bi - aj * ci + bj * di
            mz = dj * ai + cj * bi - bj * ci - aj * di

            # Find max |dot(M, s)| over all proper symmetry elements.
            # Improper elements are pre-filtered by the caller.
            max_dp = 0.0
            for k in range(s):
                dp = (
                    mw * sym_ops[k, 0]
                    + mx * sym_ops[k, 1]
                    + my * sym_ops[k, 2]
                    + mz * sym_ops[k, 3]
                )
                if dp < 0.0:
                    dp = -dp
                if dp > max_dp:
                    max_dp = dp

            # Clamp to [0, 1] to guard against floating-point overshoot.
            # The round() call matches the Dask path's da.round(dp, 12):
            # it collapses values like 1 - O(ε) to exactly 1.0, which
            # prevents the diagonal distance from being a small positive
            # number (~3e-8) instead of exactly 0.
            max_dp = round(max_dp, 12)
            if max_dp > 1.0:
                max_dp = 1.0

            out[i, j] = np.arccos(2.0 * max_dp * max_dp - 1.0)

    return out


# ------------------------- misorientation --------------------------- #


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _mori_distance_matrix(
    qu: np.ndarray, sym_ops: np.ndarray
) -> np.ndarray:  # pragma: no cover
    r"""Return the pairwise symmetry-reduced misorientation distance
    matrix.

    Given misorientations :math:`M_i` and symmetry elements
    :math:`s \in S`, compute

    .. math::

        D_{ij} = \arccos(2 \cdot (\max_{l,r \in S} |\langle M_i s_l M_j^{-1}, s_r \rangle|)^2 - 1)

    This is equivalent to the full formula
    :math:`\max_{k,l,p}|\langle s_k M_i s_l M_j^{-1}, s_p \rangle|` used
    by the Dask path, because left-invariance of the quaternion dot
    product collapses the independent maximisation over :math:`k` and
    :math:`p` into a single index.

    Memory usage is O(n*n).

    Parameters
    ----------
    qu
        Array of shape (n, 4) with unit quaternion components.
    sym_ops
        Array of shape (s, 4) with unit quaternion components of all
        symmetry elements (proper *and* improper).

    Returns
    -------
    angles
        Array of shape (n, n) with pairwise misorientation angles in
        radians.
    """
    qu_size = qu.shape[0]
    sym_size = sym_ops.shape[0]
    distance_matrix = np.empty((qu_size, qu_size), dtype=np.float64)
    

    for i in nb.prange(qu_size):
        ai = qu[i, 0]
        bi = qu[i, 1]
        ci = qu[i, 2]
        di = qu[i, 3]

        for j in range(i, qu_size):
            # M_j^{-1} = conj(M_j) for unit quaternions
            wj = qu[j, 0]
            xj = -qu[j, 1]
            yj = -qu[j, 2]
            zj = -qu[j, 3]

            max_dp = 0.0

            for l in range(sym_size):
                sla = sym_ops[l, 0]
                slb = sym_ops[l, 1]
                slc = sym_ops[l, 2]
                sld = sym_ops[l, 3]

                # t = M_i * s_l
                t0 = ai * sla - bi * slb - ci * slc - di * sld
                t1 = ai * slb + bi * sla + ci * sld - di * slc
                t2 = ai * slc - bi * sld + ci * sla + di * slb
                t3 = ai * sld + bi * slc - ci * slb + di * sla

                # u = t * M_j^{-1} = (M_i * s_l) * conj(M_j)
                u0 = t0 * wj - t1 * xj - t2 * yj - t3 * zj
                u1 = t0 * xj + t1 * wj + t2 * zj - t3 * yj
                u2 = t0 * yj - t1 * zj + t2 * wj + t3 * xj
                u3 = t0 * zj + t1 * yj - t2 * xj + t3 * wj

                # Find max |<u, s_r>| over all r in S
                for r in range(sym_size):
                    dp = (
                        u0 * sym_ops[r, 0]
                        + u1 * sym_ops[r, 1]
                        + u2 * sym_ops[r, 2]
                        + u3 * sym_ops[r, 3]
                    )
                    if dp < 0.0:
                        dp = -dp
                    if dp > max_dp:
                        max_dp = dp

            if max_dp > 1.0:
                max_dp = 1.0

            max_dp = round(max_dp, 12)
            if max_dp > 1.0:
                max_dp = 1.0

            distance = np.arccos(2.0 * max_dp * max_dp - 1.0)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            

<<<<<<< Updated upstream
    return out

@nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _mori_nearest_neighbor(
    qu: np.ndarray,
    ref: np.ndarray,
    sym_ops: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    r"""Return the symmetry-reduced nearest neighbors to a reference
    misorientation.

    This is equivalent to _mori_distance_matrix, but calculated
    relative to only a single reference point, and returning the
    misorientations, not their dot products.

    Memory usage is O(n).

    Parameters
    ----------
    qu
        Array of shape (n, 4) with unit quaternion components.
    ref
        Array of shape (1, 4) of a reference unit quternion that
        distances are calculated relative to.
    sym_ops
        Array of shape (s, 4) with unit quaternion components of all
        symmetry elements (proper *and* improper).

    Returns
    -------
    angles
        Array of shape (n, n) with pairwise misorientation angles in
        radians.
    """
    n = qu.shape[0]
    s = sym_ops.shape[0]
    out = np.empty((n, 4), dtype=np.float64)

    # M_j^{-1} = conj(M_j) for unit quaternions
    refw = ref[0, 0]
    refx = -ref[0, 1]
    refy = -ref[0, 2]
    refz = -ref[0, 3]

    for i in nb.prange(n):
        ai = qu[i, 0]
        bi = qu[i, 1]
        ci = qu[i, 2]
        di = qu[i, 3]

        max_dp = 0.0

        for l in range(s):
            sl0 = sym_ops[l, 0]
            sl1 = sym_ops[l, 1]
            sl2 = sym_ops[l, 2]
            sl3 = sym_ops[l, 3]

            # t = M_i * s_l
            t0 = ai * sl0 - bi * sl1 - ci * sl2 - di * sl3
            t1 = ai * sl1 + bi * sl0 + ci * sl3 - di * sl2
            t2 = ai * sl2 - bi * sl3 + ci * sl0 + di * sl1
            t3 = ai * sl3 + bi * sl2 - ci * sl1 + di * sl0

            # u = t * M_j^{-1} = (M_i * s_l) * conj(M_j)
            u0 = t0 * refw - t1 * refx - t2 * refy - t3 * refz
            u1 = t0 * refx + t1 * refw + t2 * refz - t3 * refy
            u2 = t0 * refy - t1 * refz + t2 * refw + t3 * refx
            u3 = t0 * refz + t1 * refy - t2 * refx + t3 * refw

            # Find max |<u, s_r>| over all r in S
            for r in range(s):
                dp = (
                    u0 * sym_ops[r, 0]
                    + u1 * sym_ops[r, 1]
                    + u2 * sym_ops[r, 2]
                    + u3 * sym_ops[r, 3]
                )
                if dp < 0.0:
                    dp = -dp
                if dp > max_dp:
                    max_dp = dp
                    out[i,0] = u0
                    out[i,1] = u1
                    out[i,2] = u2
                    out[i,3] = u3

    return out
=======
    return distance_matrix

@nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _mori_reduced_distance_and_neighbor(
    qu_ref: np.ndarray,
    qu_query: np.ndarray,
    sym_query_ops: np.ndarray,
) -> [np.ndarray,np.ndarray]:  # pragma: no cover

    qu_ref_size = qu_ref.shape[0]
    qu_query_size = qu_query.shape[0]
    sym_size = sym_query_ops.shape[0]
    distance_matrix = np.empty((qu_ref_size, qu_query), dtype=np.float64)
    sym_matrix = np.empty((qu_ref_size, qu_query,2), dtype=np.int64)

    for i in nb.prange(qu_ref_size):
        ai = qu_ref[i, 0]
        bi = qu_ref[i, 1]
        ci = qu_ref[i, 2]
        di = qu_ref[i, 3]

        for j in range(qu_query_size):
            # M_j^{-1} = conj(M_j) for unit quaternions
            wj = qu_query[j, 0]
            xj = -qu_query[j, 1]
            yj = -qu_query[j, 2]
            zj = -qu_query[j, 3]

            max_dp = 0.0
            l_nearest = 0
            r_nearest = 0

            for l in range(sym_size):
                sla = sym_query_ops[l, 0]
                slb = sym_query_ops[l, 1]
                slc = sym_query_ops[l, 2]
                sld = sym_query_ops[l, 3]

                # t = M_i * s_l
                t0 = ai * sla - bi * slb - ci * slc - di * sld
                t1 = ai * slb + bi * sla + ci * sld - di * slc
                t2 = ai * slc - bi * sld + ci * sla + di * slb
                t3 = ai * sld + bi * slc - ci * slb + di * sla

                # u = t * M_j^{-1} = (M_i * s_l) * conj(M_j)
                u0 = t0 * wj - t1 * xj - t2 * yj - t3 * zj
                u1 = t0 * xj + t1 * wj + t2 * zj - t3 * yj
                u2 = t0 * yj - t1 * zj + t2 * wj + t3 * xj
                u3 = t0 * zj + t1 * yj - t2 * xj + t3 * wj

                # Find max |<u, s_r>| over all r in S
                for r in range(sym_size):
                    dp = (
                        u0 * sym_query_ops[r, 0]
                        + u1 * sym_query_ops[r, 1]
                        + u2 * sym_query_ops[r, 2]
                        + u3 * sym_query_ops[r, 3]
                    )
                    if dp < 0.0:
                        dp = -dp
                    if dp > max_dp:
                        max_dp = dp
                        l_nearest = l
                        r_nearest = r

            if max_dp > 1.0:
                max_dp = 1.0

            max_dp = round(max_dp, 12)
            if max_dp > 1.0:
                max_dp = 1.0

            distance = np.arccos(2.0 * max_dp * max_dp - 1.0)
            distance_matrix[i, j] = distance
            sym_matrix[i,j,0]=l_nearest
            sym_matrix[i,j,1]=r_nearest

    return distance_matrix, sym_matrix
>>>>>>> Stashed changes
