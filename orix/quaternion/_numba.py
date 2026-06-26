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
    n = qu.shape[0]
    s = sym_ops.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    dot_product = 0.0

    for i in nb.prange(n):
        ai = qu[i, 0]
        bi = qu[i, 1]
        ci = qu[i, 2]
        di = qu[i, 3]

        for j in range(i, n):
            # M_j^{-1} = conj(M_j) for unit quaternions
            wj = qu[j, 0]
            xj = -qu[j, 1]
            yj = -qu[j, 2]
            zj = -qu[j, 3]

            max_dp = 0.0

            for l in range(s):
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

            if max_dp > 1.0:
                max_dp = 1.0

            max_dp = round(max_dp, 12)
            if max_dp > 1.0:
                max_dp = 1.0

            dot_product = np.arccos(2.0 * max_dp * max_dp - 1.0)
            out[i, j] = dot_product
            out[j, i] = dot_product

    return out


@nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _mori_nearest_neighbor_symmetries(
    qu_ref: np.ndarray,
    miso_query: np.ndarray,
    starting_sym_ops: np.ndarray,
    ending_sym_ops: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    r"""Return the symmetries that place queried misorientations
    closest to reference quaternions.

    This function runs the computationally expensive calculation of
    comparing all symmetry combinations between a misorientation's two
    crystal groups to find the symmetrically-equivalent version nearest
    to a reference quaternion. Mathematically, given a misorientation
    :math:`M`, reference quaternion :math:`Q`, starting symmetries
    :math:`{S}_k`, and ending-symmetries `{E}_l`, this equation
    returns k and l such that :

    .. math::

         \left(x, y\right) = \max_{x \in [0,k],y \in [0,l]} \left(\left[E_x M S_y\right] \cdot Q^{-1} \right)


    This is done in parallel for every combination of `m` queried
    misorientations and  `n` reference quaternions.

    Parameters
    ----------
    qu_ref
        Array of shape (m, 4) of reference unit quaternions.
    miso_query
        Array of shape (n, 4) of misorientations to be reduced to the
        nearest neighbor for each reference quaternion.
    starting_sym_ops
        Array of shape (k, 4) with unit quaternion components of all
        symmetry elements (proper *and* improper) applied to the
        right side of the misorientations.
    ending_sym_ops
        Array of shape (l, 4) with unit quaternion components of all
        symmetry elements (proper *and* improper) applied to the
        left side of the misorientations.

    Returns
    -------
    sym_matrix
        Array of shape(n, m, 2) of the starting and ending symmetries
        that reduce each mth misorientation to be closest to the
        nth reference quaternion.
    """

    m = qu_ref.shape[0]
    n = miso_query.shape[0]
    k = starting_sym_ops.shape[0]
    l = ending_sym_ops.shape[0]
    sym_matrix = np.empty((n, m, 2), dtype=np.int64)

    for i in nb.prange(n):
        # Q^{-1} = conj(Q) for unit quaternions
        ai = qu_ref[i, 0]
        bi = -qu_ref[i, 1]
        ci = -qu_ref[i, 2]
        di = -qu_ref[i, 3]

        for j in nb.prange(m):
            aj = miso_query[m, 0]
            bj = miso_query[m, 1]
            cj = miso_query[m, 2]
            dj = miso_query[m, 3]

            max_dp = 0.0
            k_nearest = 0
            l_nearest = 0

            for y in range(k):
                ay = starting_sym_ops[y, 0]
                by = starting_sym_ops[y, 1]
                cy = starting_sym_ops[y, 2]
                dy = starting_sym_ops[y, 3]

                # t = S_y * conj(Q_i)
                t0 = ay * ai - by * bi - cy * ci - dy * di
                t1 = ay * bi + by * ai + cy * di - dy * ci
                t2 = ay * ci - by * di + cy * ai + dy * bi
                t3 = ay * di + by * ci - cy * bi + dy * ai

                # u = M_j * t = M_j * S_y * conj(Q_i)
                u0 = aj * t0 - bj * t1 - cj * t2 - dj * t3
                u1 = aj * t1 + bj * t0 + cj * t3 - dj * t2
                u2 = aj * t2 - bj * t3 + cj * t0 + dj * t1
                u3 = aj * t3 + bj * t2 - cj * t1 + dj * t0

                # v = M_j * t = M_j * S_y * conj(Q_i)
                # However, we only need to find the k and l that give
                # the minimum angle, (maxium dot product)
                for x in range(l):
                    dp = (
                        u0 * ending_sym_ops[y, 0]
                        - u1 * ending_sym_ops[y, 1]
                        - u2 * ending_sym_ops[y, 2]
                        - u3 * ending_sym_ops[y, 3]
                    )
                    if dp < 0.0:
                        dp = -dp
                    if dp > max_dp:
                        max_dp = dp
                        k_nearest = y
                        l_nearest = x

                sym_matrix[n, m, 0] = k_nearest
                sym_matrix[n, m, 1] = l_nearest

    return sym_matrix


# @nb.njit(cache=True, fastmath=True, nogil=True, parallel=True)
# def _mori_nearest_neighbor(
#     qu: np.ndarray,
#     ref: np.ndarray,
#     sym_ops: np.ndarray,
# ) -> np.ndarray:  # pragma: no cover
#     r"""Return the symmetry-reduced nearest neighbors to a reference
#     misorientation.

#     This is equivalent to _mori_distance_matrix, but calculated
#     relative to only a single reference point, and returning the
#     misorientations, not their dot products.

#     Memory usage is O(n).

#     Parameters
#     ----------
#     qu
#         Array of shape (n, 4) with unit quaternion components.
#     ref
#         Array of shape (1, 4) of a reference unit quternion that
#         distances are calculated relative to.
#     sym_ops
#         Array of shape (s, 4) with unit quaternion components of all
#         symmetry elements (proper *and* improper).

#     Returns
#     -------
#     qu_out
#         Array of shape(n, 4) of nearest neighbor misorientations in
#         quaternion format
#     """
#     n = qu.shape[0]
#     s = sym_ops.shape[0]
#     qu_out = np.empty((n, 4), dtype=np.float64)

#     # M_j^{-1} = conj(M_j) for unit quaternions
#     wref = ref[0, 0]
#     xref = -ref[0, 1]
#     yref = -ref[0, 2]
#     zref = -ref[0, 3]

#     for i in nb.prange(n):
#         ai = qu[i, 0]
#         bi = qu[i, 1]
#         ci = qu[i, 2]
#         di = qu[i, 3]

#         max_dp = 0.0

#         for l in range(s):
#             sl0 = sym_ops[l, 0]
#             sl1 = sym_ops[l, 1]
#             sl2 = sym_ops[l, 2]
#             sl3 = sym_ops[l, 3]

#             # t = M_i * s_l
#             t0 = ai * sl0 - bi * sl1 - ci * sl2 - di * sl3
#             t1 = ai * sl1 + bi * sl0 + ci * sl3 - di * sl2
#             t2 = ai * sl2 - bi * sl3 + ci * sl0 + di * sl1
#             t3 = ai * sl3 + bi * sl2 - ci * sl1 + di * sl0

#             # u = t * M_j^{-1} = (M_i * s_l) * conj(M_j)
#             u0 = t0 * wref - t1 * xref - t2 * yref - t3 * zref
#             u1 = t0 * xref + t1 * wref + t2 * zref - t3 * yref
#             u2 = t0 * yref - t1 * zref + t2 * wref + t3 * xref
#             u3 = t0 * zref + t1 * yref - t2 * xref + t3 * wref

#             # Find max |<u, s_r>| over all r in S
#             for r in range(s):
#                 dp = (
#                     u0 * sym_ops[r, 0]
#                     + u1 * sym_ops[r, 1]
#                     + u2 * sym_ops[r, 2]
#                     + u3 * sym_ops[r, 3]
#                 )
#                 if dp < 0.0:
#                     dp = -dp
#                 if dp > max_dp:
#                     max_dp = dp
#                     qu_out[i,0] = u0
#                     qu_out[i,1] = u1
#                     qu_out[i,2] = u2
#                     qu_out[i,3] = u3

#     return qu_out
