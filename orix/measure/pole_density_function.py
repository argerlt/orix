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
from scipy.sparse import coo_array

from orix.projections.stereographic import StereographicProjection
from orix.quaternion.symmetry import Symmetry, Rotation
from orix.vector.vector3d import Vector3d
from scipy.spatial import Voronoi


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
    from orix.sampling.S2_sampling import _sample_S2_equal_area_coordinates
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

    if symmetry is not None:
        v = v.in_fundamental_sector(symmetry)

    v = v.flatten().unit
    # proceedurally generate grids on each face of the geonomic cube
    face_rotations = Rotation.from_axes_angles(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
        [0, 90, 180, 270, 90, 270],
        degrees=True,
    )
    face_center_vecs = face_rotations * (Vector3d.zvector())
    center_to_edge_EA = np.arctan(_edge_grid_spherified_corner_cube(resolution))
    bin_edges = np.hstack([center_to_edge_EA, np.arctan(1)])
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    x_ang, y_ang = np.meshgrid(bin_centers, bin_centers)
    v_grid = Vector3d(np.stack([np.tan(x_ang), np.tan(y_ang), x_ang * 0 + 1]).T)
    all_v = ((face_rotations.inv()).outer(v_grid)).reshape(6 * v_grid.size)

    # find the relative solid-angle weighting of each bin
    SA_relative_weight = (
        1
        / (np.tan(x_ang) ** 2 + np.tan(y_ang) ** 2 + 1)
        / (np.cos(x_ang) * np.cos(y_ang))
        / (1 - 0.5 * (np.sin(x_ang) * np.sin(y_ang)) ** 2)
    ) / 6

    # identify the face to bin on and perform the binning
    face_index = np.argmax(face_center_vecs.dot_outer(v), 0)
    w = np.zeros([6, bin_centers.size, bin_centers.size])
    for idx, face_rot in enumerate(face_rotations):
        if np.isin(idx, face_index):
            mask = face_index == idx
            x, y, z = (face_rot * (v[mask])).xyz
            cube_xy = np.stack([np.arctan(x / z), np.arctan(y / z)])
            bin_count = np.histogram2d(*cube_xy, [bin_edges, bin_edges])[0]
            w[idx] = bin_count / SA_relative_weight

    # create adjacency matrix via a voronoi tesselation to perform blurring
    def vector3d2schmidt(v):
        r = 2 * np.sin(v.polar / 2)
        azimuth = v.azimuth
        x = r * np.cos(azimuth)
        y = r * np.sin(azimuth)
        return np.stack([x, y]).T

    vor = Voronoi(vector3d2schmidt(v))

    if poles[hemisphere] > 0:
        w = w[1:]
    else:
        w = w[()]

    # face_center_vecs = face_rotations * (Vector3d.zvector())
    # center_to_edge_EA = np.arctan(_edge_grid_spherified_corner_cube(resolution))
    # bin_edges = np.hstack([center_to_edge_EA, np.arctan(1)])
    # bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # x_ang, y_ang = np.meshgrid(bin_centers, bin_centers)
    # v_grid = Vector3d(np.stack([np.tan(x_ang), np.tan(y_ang), x_ang * 0 + 1]).T)
    # all_v = (face_rotations.inv()).outer(v_grid)
    # all_v.scatter(s=2,c = cm(w))
    # all_v = ((face_rotations.inv()).outer(v_grid)).reshape(6*v_grid.size())
    # all_v = ((face_rotations.inv()).outer(v_grid)).reshape(6*v_grid.size)
    # all_v
    # all_v.scatter(s=2,c = cm(w))
    # all_v
    # all_v.scatter(s=2,c = cm.viridis(w))
    # all_v.scatter(s=2,c = cm.viridis(w.flatten()))
    # all_v.scatter(s=5,c = cm.viridis(w.flatten()))
    # (face_rotations.inv()).outer(v_grid)
    # ((face_rotations.inv()).outer(v_grid)).shape
    # all_v = ((face_rotations.inv()).outer(v_grid.T)).reshape(6*v_grid.size)
    # v_grid = Vector3d(np.stack([np.tan(y_ang), np.tan(x_ang), x_ang * 0 + 1]).T)
    # all_v = ((face_rotations.inv()).outer(v_grid)).reshape(6*v_grid.size)
    # all_v.scatter(s=5,c = cm.viridis(w.flatten()))
    # v = Vector3d.random(20).get_circle(1000)
    # v = v.flatten().unit
    # # proceedurally generate grids on each face of the geonomic cube
    # face_rotations = Rotation.from_axes_angles(
    #     [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
    #     [0, 90, 180, 270, 90, 270],
    #     degrees=True,
    # )
    # face_center_vecs = face_rotations * (Vector3d.zvector())
    # center_to_edge_EA = np.arctan(_edge_grid_spherified_corner_cube(resolution))
    # bin_edges = np.hstack([center_to_edge_EA, np.arctan(1)])
    # bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # x_ang, y_ang = np.meshgrid(bin_centers, bin_centers)
    # v_grid = Vector3d(np.stack([np.tan(x_ang), np.tan(y_ang), x_ang * 0 + 1]).T)
    # all_v = ((face_rotations.inv()).outer(v_grid)).reshape(6*v_grid.size)

    # # find the relative solid-angle weighting of each bin
    # SA_relative_weight = (
    #     1
    #     / (np.tan(x_ang) ** 2 + np.tan(y_ang) ** 2 + 1)
    #     / (np.cos(x_ang) * np.cos(y_ang))
    #     / (1 - 0.5 * (np.sin(x_ang) * np.sin(y_ang)) ** 2)
    # ) / 6

    # # identify the face to bin on and perform the binning
    # face_index = np.argmax(face_center_vecs.dot_outer(v), 0)
    # w = np.zeros([6, bin_centers.size, bin_centers.size])
    # for idx, face_rot in enumerate(face_rotations):
    #     if np.isin(idx, face_index):
    #         mask = face_index == idx
    #         x, y, z = (face_rot * (v[mask])).xyz
    #         cube_xy = np.stack([np.arctan(x / z), np.arctan(y / z)])
    #         bin_count = np.histogram2d(*cube_xy, [bin_edges, bin_edges])[0]
    #         w[idx] = bin_count / SA_relative_weight

    # fig,ax = plt.subplots(2,1,subplot_kw={'projection':'stereographic'})
    # fig,ax = plt.subplots(1,2,subplot_kw={'projection':'stereographic'})
    # ax[0].scatter(all_v,cm.viridis(w))
    # w.max()
    # ax[0].scatter(all_v,cm.viridis(w/20))
    # fig,ax = plt.subplots(1,2,subplot_kw={'projection':'stereographic'})
    # ax[0].scatter(all_v)
    # fig,ax = plt.subplots(1,2,subplot_kw={'projection':'stereographic'})
    # ax[0].scatter(all_v,c = cm.viridis(w/20))
    # ax[0].scatter(all_v,c = cm.viridis(w.flatten()/20))
    # v = Vector3d.random(20).get_circle(steps = 1000)
    # v = v.flatten().unit
    # # proceedurally generate grids on each face of the geonomic cube
    # face_rotations = Rotation.from_axes_angles(
    #     [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
    #     [0, 90, 180, 270, 90, 270],
    #     degrees=True,
    # )
    # face_center_vecs = face_rotations * (Vector3d.zvector())
    # center_to_edge_EA = np.arctan(_edge_grid_spherified_corner_cube(resolution))
    # bin_edges = np.hstack([center_to_edge_EA, np.arctan(1)])
    # bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # x_ang, y_ang = np.meshgrid(bin_centers, bin_centers)
    # v_grid = Vector3d(np.stack([np.tan(x_ang), np.tan(y_ang), x_ang * 0 + 1]).T)
    # all_v = ((face_rotations.inv()).outer(v_grid)).reshape(6*v_grid.size)

    # # find the relative solid-angle weighting of each bin
    # SA_relative_weight = (
    #     1
    #     / (np.tan(x_ang) ** 2 + np.tan(y_ang) ** 2 + 1)
    #     / (np.cos(x_ang) * np.cos(y_ang))
    #     / (1 - 0.5 * (np.sin(x_ang) * np.sin(y_ang)) ** 2)
    # ) / 6

    # # identify the face to bin on and perform the binning
    # face_index = np.argmax(face_center_vecs.dot_outer(v), 0)
    # w = np.zeros([6, bin_centers.size, bin_centers.size])
    # for idx, face_rot in enumerate(face_rotations):
    #     if np.isin(idx, face_index):
    #         mask = face_index == idx
    #         x, y, z = (face_rot * (v[mask])).xyz
    #         cube_xy = np.stack([np.arctan(x / z), np.arctan(y / z)])
    #         bin_count = np.histogram2d(*cube_xy, [bin_edges, bin_edges])[0]
    #         w[idx] = bin_count / SA_relative_weight

    # fig,ax = plt.subplots(1,2,subplot_kw={'projection':'stereographic'})
    # ax.scatter(v)
    # ax[0].scatter(all_v,c = cm.viridis(w.flatten()/20))

    #     # apply gaussin blur
    #     pad = int(np.ceil(3 * sigma / resolution))
    #     padded_w = np.pad(w,
    #                       pad_width=[(0, 0), (pad, pad), (pad, pad)],
    #                       constant_values=0)
    #     gauss_padded_w = gaussian_filter(
    #         padded_w, sigma=sigma, truncate=3, mode='constant')
    #     # blur over the edges
    #     gauss_w = gauss_padded_w[:, pad:-pad, pad:-pad]
    #     gauss_w[0, :, :pad] += gauss_padded_w[1, :, -pad:]

    #     connected_edge_pairs = (
    #         ([0,],[]),
    #         ([],[]),
    #         ([],[]),
    #         ([],[]),
    #         )

    #         connected_edge_pairs = (
    #             ((2, slice(None), -1), (4, slice(None), -1)),  # +y+z
    #             ((3, slice(None), -1), (4, slice(None), 0)),  # -y+z
    #             ((2, slice(None), 0), (5, slice(None), -1)),  # +y-z
    #             ((3, slice(None), 0), (5, slice(None), 0)),  # -y-z
    #             ((0, slice(None), -1), (4, -1, slice(None))),  # +x+z
    #             ((1, slice(None), -1), (4, 0, slice(None))),  # -x+z
    #             ((0, slice(None), 0), (5, -1, slice(None))),  # +x-z
    #             ((1, slice(None), 0), (5, 0, slice(None))),  # -x-z
    #             ((0, -1, slice(None)), (2, -1, slice(None))),  # +x+y
    #             ((1, -1, slice(None)), (2, 0, slice(None))),  # -x+y
    #             ((0, 0, slice(None)), (3, -1, slice(None))),  # +x-y
    #             ((1, 0, slice(None)), (3, 0, slice(None))),  # -x-y
    #         )

    #     # Wrap the blurring around the edges
    #     # NOTE: there should be a more proceedural way to do this...
    #     for i in range(6):
    #         neighbor_mask = np.abs(face_center_vecs[i].angle_with(face_center_vecs)-np.pi/2)<1e-4
    #         relative_rots = ((face_rotations[i]).inv())*face_rotations[neighbor_mask]
    #         ud_mask = np.abs(relative_rots.axis.x) > 0.9
    #         lr_mask = relative_rots.axis.data.max(axis=1) > 0.9

    #     w_0 = np.zeros([])

    #     x_wrap = np.zeros(4*w.shape[1],w.shape[2])

    #     w[0] = gauss_padded_w[pad:-pad,pad,-pad]
    #     w[0,:pad] = gauss_padded_w[pad:-pad,pad,-pad]
    #     w[0,-pad:] = gauss_padded_w[pad:-pad,pad,-pad]
    #     w[0,:pad] = gauss_padded_w[pad:-pad,pad,-pad]
    #     w[0,-pad:] = gauss_padded_w[pad:-pad,pad,-pad]

    #     padded_dims = bin_centers.size + (2 * padding)
    #     if padding > 0:
    #         # blur around x
    #         blur_x = gaussian_filter1d(np.hstack([w[0], w[1], w[2], w[3]]), sigma=sigma, axis=1, mode='wrap').T.reshape((4,) + w.shape[1:])
    #         w[0] = blur_x[0].T
    #         w[1] = blur_x[1].T
    #         w[2] = blur_x[2].T
    #         w[3] = blur_x[3].T
    #         # blur around y
    #         blur_y = gaussian_filter1d(np.vstack([w[5],w[2][::-1,::-1],w[4],w[0]]),axis=0,sigma=sigma,mode='wrap').reshape((4,)+w.shape[1:])
    #         w[5] = blur_y[0]
    #         w[2] = blur_y[1][::-1,::-1]
    #         w[4] = blur_y[2]
    #         w[0] = blur_y[3]
    #         # blur around z
    #         blur_z = gaussian_filter1d(np.hstack([w[1].T,w[4], w[3].T, w[5]]),sigma=sigma,axis=1,mode='wrap').T.reshape((4,) + w.shape[1:])
    #         w[1] = blur_z[0]
    #         w[4] = blur_z[1].T
    #         w[3] = blur_z[2]
    #         w[5] = blur_z[3].T

    #         blur_y = gaussian_filter1d(np.vstack([w[2],w[5][::-1,::-1],w[4],w[0]]),axis=0,sigma=sigma,mode='wrap')
    # #        w[0:4] = blur_x
    #         blur_y = gaussian_filter1d(
    #             np.hstack([w[0], w[4], w[2][::-1,::-1], w[5]]), sigma=sigma, axis=1, mode="wrap"
    #         ).reshape((4,) + w.shape[1:])
    #         w[0] = blur_y[0]
    #         w[4] = blur_y[1]
    #         w[0] = blur_y[2][::-1,::-1]
    #         w[0] = blur_y[0]
    #         # blur around z
    #         blur_z = gaussian_filter1d(
    #             np.hstack([w[1], w[4], w[3][::-1,::-1], w[5]]), sigma=sigma, axis=1, mode="wrap"
    #         ).reshape((4,) + w.shape[1:])

    #         w[0:4] = blur_x.reshape((4,) + w.shape[1:])

    # np.hstack([w[0],w[4],w[2],w[5]])
    azimuth, polar, _ = v.to_polar()
    # np.histogram2d expects 1d arrays
    azimuth, polar = np.ravel(azimuth), np.ravel(polar)
    if not azimuth.size:
        raise ValueError("`azimuth` and `polar` angles have 0 size.")

    # Generate equal area mesh on S2
    azimuth_coords, polar_coords = _sample_S2_equal_area_coordinates(
        resolution,
        hemisphere=hemisphere,
        azimuth_endpoint=True,
    )
    azimuth_grid, polar_grid = np.meshgrid(azimuth_coords, polar_coords, indexing="ij")
    # generate histogram in angular space
    hist, *_ = np.histogram2d(
        azimuth,
        polar,
        bins=(azimuth_coords, polar_coords),
        density=False,
        weights=weights,
    )

    # "wrap" along azimuthal axis, "reflect" along polar axis
    mode = ("wrap", "reflect")
    # apply broadening in angular space
    hist = gaussian_filter(hist, sigma / resolution, mode=mode)

    # In the case of inverse pole figure, accumulate all values outside
    # of the point group fundamental sector back into correct bin within
    # fundamental sector
    if symmetry is not None:
        # compute histogram bin centers in azimuth and polar coords
        azimuth_center_grid, polar_center_grid = np.meshgrid(
            azimuth_coords[:-1] + np.diff(azimuth_coords) / 2,
            polar_coords[:-1] + np.diff(polar_coords) / 2,
            indexing="ij",
        )
        v_center_grid = Vector3d.from_polar(
            azimuth=azimuth_center_grid, polar=polar_center_grid
        ).unit
        # fold back in into fundamental sector
        v_center_grid_fs = v_center_grid.in_fundamental_sector(symmetry)
        azimuth_center_fs, polar_center_fs, _ = v_center_grid_fs.to_polar()
        azimuth_center_fs = azimuth_center_fs.ravel()
        polar_center_fs = polar_center_fs.ravel()

        # Generate coorinates with user-defined resolution.
        # When `symmetry` is defined, the initial grid was calculated
        # with `resolution = resolution / 2`
        azimuth_coords_res2, polar_coords_res2 = _sample_S2_equal_area_coordinates(
            2 * resolution,
            hemisphere=hemisphere,
            azimuth_endpoint=True,
        )
        azimuth_res2_grid, polar_res2_grid = np.meshgrid(
            azimuth_coords_res2, polar_coords_res2, indexing="ij"
        )
        v_res2_grid = Vector3d.from_polar(
            azimuth=azimuth_res2_grid, polar=polar_res2_grid
        )

        # calculate histogram values for vectors folded back into
        # fundamental sector
        i = np.digitize(azimuth_center_fs, azimuth_coords_res2[1:-1])
        j = np.digitize(polar_center_fs, polar_coords_res2[1:-1])
        # recompute histogram
        temp = np.zeros((azimuth_coords_res2.size - 1, polar_coords_res2.size - 1))
        # add hist data to new histogram without buffering
        np.add.at(temp, (i, j), hist.ravel())

        # get new histogram bins centers to compute histogram mask
        azimuth_center_res2_grid, polar_center_res2_grid = np.meshgrid(
            azimuth_coords_res2[:-1] + np.ediff1d(azimuth_coords_res2) / 2,
            polar_coords_res2[:-1] + np.ediff1d(polar_coords_res2) / 2,
            indexing="ij",
        )
        v_center_res2_grid = Vector3d.from_polar(
            azimuth=azimuth_center_res2_grid, polar=polar_center_res2_grid
        ).unit

        # compute histogram data array as masked array
        hist = np.ma.array(
            temp, mask=~(v_center_res2_grid <= symmetry.fundamental_sector)
        )
        # calculate bin vertices
        x, y = sp.vector2xy(v_res2_grid)
        x, y = x.reshape(v_res2_grid.shape), y.reshape(v_res2_grid.shape)
    else:
        # all points valid in stereographic projection
        hist = np.ma.array(hist, mask=np.zeros_like(hist, dtype=bool))
        # calculate bin vertices
        v_grid = Vector3d.from_polar(azimuth=azimuth_grid, polar=polar_grid).unit
        x, y = sp.vector2xy(v_grid)
        x, y = x.reshape(v_grid.shape), y.reshape(v_grid.shape)

    # Normalize by the average number of counts per cell on the
    # unit sphere to calculate in terms of Multiples of Random
    # Distribution (MRD). See :cite:`rohrer2004distribution`.
    # as `hist` is a masked array, only valid (unmasked) values are
    # used in this computation
    if mrd:
        hist = hist / hist.mean()

    if log:
        # +1 to avoid taking the log of 0
        hist = np.log(hist + 1)

    return hist, (x, y)
