#
# Copyright 2018-2025 the orix developers
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

import numpy as np
from scipy.ndimage import gaussian_filter

from orix.projections.stereographic import StereographicProjection
from orix.quaternion.symmetry import Symmetry
from orix.quaternion import Rotation
from orix.vector.vector3d import Vector3d


def pole_density_function(
    *args: np.ndarray | Vector3d,
    resolution: float = 1,
    sigma: float = 5,
    weights: np.ndarray | None = None,
    hemisphere: str = "upper",
    symmetry: Symmetry | None = None,
    log: bool = False,
    mrd: bool = True,
) -> tuple[np.ma.MaskedArray, tuple[np.ndarray, np.ndarray]]:
    """Compute the Pole Density Function (PDF) of vectors in the
    stereographic projection. See :cite:`rohrer2004distribution`.

    If ``symmetry`` is defined then the PDF is folded back into the
    point group fundamental sector and accumulated.

    Parameters
    ----------
    args
        Vector(s), or azimuth and polar angles of the vectors, the
        latter passed as two separate arguments.
    resolution
        The angular resolution of the sampling grid in degrees.
        Default value is 1.
    sigma
        The angular resolution of the applied broadening in degrees.
        Default value is 5.
    weights
        The weights for the individual vectors. Default is ``None``, in
        which case the weight of each vector is 1.
    hemisphere
        Which hemisphere(s) to plot the vectors on, options are
        ``"upper"`` and ``"lower"``. Default is ``"upper"``.
    symmetry
        If provided the PDF is calculated within the fundamental sector
        of the point group symmetry, otherwise the PDF is calculated
        on ``hemisphere``. Default is ``None``.
    log
        If ``True`` the log(PDF) is calculated. Default is ``True``.
    mrd
        If ``True`` the returned PDF is in units of Multiples of Random
        Distribution (MRD), otherwise the units are bin counts. Default
        is ``True``.

    Returns
    -------
    hist
        The computed histogram, shape is (N, M).
    x, y
        Tuple of coordinate grids for the bin edges of ``hist``. The
        units of ``x`` and ``y`` are cartesian coordinates on the
        stereographic projection plane and the shape of both ``x`` and
        ``y`` is (N + 1, M + 1).

    See Also
    --------
    orix.plot.InversePoleFigurePlot.pole_density_function
    orix.plot.StereographicPlot.pole_density_function
    orix.vector.Vector3d.pole_density_function
    """
    from orix.sampling._polyhedral_sampling import _edge_grid_spherified_corner_cube

    hemisphere = hemisphere.lower()

    poles = {"upper": -1, "lower": 1}
    sp = StereographicProjection(poles[hemisphere])

    if len(args) == 1:
        v = args[0]
        if not isinstance(v, Vector3d):
            raise TypeError(
                "If one argument is passed it must be an instance of "
                + "`orix.vector.Vector3d`."
            )
    elif len(args) == 2:
        # azimuth and polar angles
        v = Vector3d.from_polar(*args)
    else:
        raise ValueError(
            "Accepts only one (Vector3d) or two (azimuth, polar) input arguments."
        )

    if weights is None:
        weights = np.ones(v.size)

    if symmetry is not None:
        v = symmetry.outer(v.flatten())
        weights = np.stack([weights.flatten() for i in range(symmetry.size)], axis=1)
    v = v.flatten()
    weights = weights.flatten()

    # Create the 2D histogram bins that are reused for each face.
    center_to_edge_EA = _edge_grid_spherified_corner_cube(resolution / 3)
    bin_edges = np.arctan(np.hstack([center_to_edge_EA, 1]))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    x_ang, y_ang = np.meshgrid(bin_centers, bin_centers)
    tanx_ang = np.tan(x_ang)
    tany_ang = np.tan(y_ang)
    sz = bin_centers.size
    v_grid = Vector3d(np.stack([tanx_ang, tany_ang, tanx_ang * 0 + 1]).T)

    # Define the 6 rotations that rotate an (x,y,1) mesh to each of the 6 faces.
    # Ordering is [100], [-100], [010], [0-10], [001], [00-1].
    face_rotations = Rotation.from_axes_angles(
        [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [90, 270, 90, 270, 0, 180],
        degrees=True,
    )

    # Define the 6 face-centered vectors, use their dot product to assign
    # each observation to a face, then bin observations per-face.
    hist = np.zeros([6, bin_centers.size, bin_centers.size])
    face_center_vecs = face_rotations.inv() * (Vector3d.zvector())
    face_index = np.argmax(face_center_vecs.dot_outer(v), 0)
    for idx, face_rot in enumerate(face_rotations):
        if np.isin(idx, face_index):
            mask = face_index == idx
            x, y, z = (face_rot * (v[mask])).xyz
            cube_xy = np.stack([np.arctan(x / z), np.arctan(y / z)])
            w = weights[mask]
            hist[idx] = np.histogram2d(*cube_xy, [bin_edges, bin_edges], weights=w)[0]

    if mrd:
        SA_relative_weight = (
            1
            / (tanx_ang**2 + tany_ang**2 + 1)
            / (np.cos(x_ang) * np.cos(y_ang))
            / (1 - 0.5 * (np.sin(x_ang) * np.sin(y_ang)) ** 2)
        ) / 6
        hist = hist / SA_relative_weight[np.newaxis, :, :]
        hist = hist / np.mean(hist)

    stdev_3_in_px = 3 * sigma * 2 / resolution
    if stdev_3_in_px > 1:
        t = 1 / 3
        N = int(np.ceil(((sigma * 3 / resolution) ** 2) / t))
        hist = _smooth_gnom_cube_histograms(hist, t, N)
        print(N)

    # reinterpolate onto  polar-azimuth grid
    v_geom_coords = (face_rotations.inv()).outer(v_grid)
    p_bins = np.linspace(0, 360, int(np.ceil(360 / resolution)) + 1) * np.pi / 180
    az_bins = np.linspace(0, 90, int(np.ceil(90 / resolution)) + 1) * np.pi / 180
    p_centers = (p_bins[1:] + p_bins[:-1]) / 2
    az_centers = (az_bins[1:] + az_bins[:-1]) / 2
    v_regridded = Vector3d.from_polar(
        *np.meshgrid(p_centers, az_centers, indexing="ij")
    )

    pol_crd, az_crd = [x.flatten() for x in v_geom_coords.to_polar()[:2]]
    regridded_hist = np.histogram2d(
        pol_crd, az_crd, [p_bins, az_bins], weights=hist.flatten()
    )[0]

    mask = ~(v_regridded <= symmetry.fundamental_sector)
    regridded_hist = np.ma.array(regridded_hist / np.sin(az_centers), mask=mask)
    x, y = sp.vector2xy(v_regridded)
    x, y = x.reshape(v_regridded.shape), y.reshape(v_regridded.shape)

    if mrd:
        regridded_hist = regridded_hist / regridded_hist.mean()

    if log:
        # +1 to avoid taking the log of 0
        regridded_hist = np.log(regridded_hist + 1)

    return regridded_hist, (x, y)


# %%
def _smooth_gnom_cube_histograms(
    histograms: np.ndarray[float],
    step_parameter: float,
    iterations: int = 1,
) -> np.ndarray[float]:
    """Histograms shape is (6, n_nbins, n_bins) and edge connectivity
    is as according to the rest of this file.
    """
    output_histogram = np.copy(histograms)
    diffused_weight = np.zeros(histograms.shape)

    for n in range(iterations):

        diffused_weight[...] = 0

        # Diffuse on faces
        for fi in range(6):
            diffused_weight[fi, 1:, :] += output_histogram[fi, :-1, :]
            diffused_weight[fi, :-1, :] += output_histogram[fi, 1:, :]
            diffused_weight[fi, :, 1:] += output_histogram[fi, :, :-1]
            diffused_weight[fi, :, :-1] += output_histogram[fi, :, 1:]

        aligned_edge_pairs = (
            ((0, slice(None), 0), (3, 0, slice(None))),  # +x+y
            ((1, slice(None), -1), (2, -1, slice(None))),  # +x+y
            ((0, -1, slice(None)), (4, 0, slice(None))),  # +x+y
            ((1, 0, slice(None)), (4, -1, slice(None))),  # +x+y
            ((2, slice(None), 0), (4, slice(None), -1)),  # +x+y
            ((3, slice(None), 0), (5, slice(None), -1)),  # +x+y
        )
        reversed_edge_pairs = (
            ((0, slice(None), -1), (2, 0, slice(None))),  # +x+y
            ((1, slice(None), 0), (3, -1, slice(None))),  # +x+y
            ((1, -1, slice(None)), (5, -1, slice(None))),  # +x+y
            ((0, 0, slice(None)), (5, 0, slice(None))),  # +x+y
            ((2, slice(None), -1), (4, slice(None), 0)),  # +x+y
            ((3, slice(None), -1), (5, slice(None), 0)),  # +x+y
        )

        for edge_1, edge_2 in aligned_edge_pairs:
            diffused_weight[edge_1] += output_histogram[edge_2]
            diffused_weight[edge_2] += output_histogram[edge_1]
        for edge_1, edge_2 in reversed_edge_pairs:
            diffused_weight[edge_1] += output_histogram[edge_2][::-1]
            diffused_weight[edge_2] += output_histogram[edge_1][::-1]

        # Add to output
        output_histogram = (
            1 - step_parameter
        ) * output_histogram + diffused_weight / 4 * step_parameter

    return output_histogram
