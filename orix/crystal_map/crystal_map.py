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

from __future__ import annotations

import copy
from typing import Any

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map._phase import Phase
from orix.crystal_map._phase_list import PhaseList
from orix.crystal_map.crystal_map_properties import CrystalMapProperties
from orix.plot._util.color import get_named_matplotlib_colors
from orix.quaternion.orientation import Orientation
from orix.quaternion.rotation import Rotation


class CrystalMap:
    """Crystallographic map of orientations, crystal phases and key
    properties associated with every spatial coordinate in a 1D or 2D.

    Parameters
    ----------
    rotations
        Rotation in each point. Must be passed with all spatial
        dimensions in the first array axis (flattened). May contain
        multiple rotations per point, included in the second array
        axes. Crystal map data size is set equal to the first array
        axis' size.
    phase_id
        Phase ID of each pixel. IDs equal to ``-1`` are considered not
        indexed. If not given, all points are considered to belong to
        one phase with ID ``0``.
    x
        Map x coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to an array of increasing
        integers from 0 to the length of the ``phase_id`` array.
    y
        Map y coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to ``None``.
    z
        Map z coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to ``None``.
    phase_list
        A PhaseList containing the name, crystallographic information,
        and plotting color of one or more phases. This is compared to
        ``phase_id``, and empty phases are added for undeined phases.
        If not given, empty phases are added for every unique phase ID.
    prop
        Dictionary of properties of each data point.
    scan_unit
        Length unit of the data. Default is ``"px"``.

    See Also
    --------
    create_coordinate_arrays
    :mod:`~orix.data`
    :func:`~orix.io.load`

    Notes
    -----
    Data is stored as 1D arrays and reshaped when necessary.

    Examples
    --------
    Constructing a crystal map from scratch, with two rows and three
    columns and containing Austenite and Ferrite orientations

    >>> from diffpy.structure import Atom, Lattice, Structure
    >>> from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
    >>> from orix.quaternion import Rotation
    >>> coords, n = create_coordinate_arrays(shape=(2, 3))
    >>> structures = [
    ...     Structure(
    ...         title="austenite",
    ...         atoms=[Atom("fe", [0] * 3)],
    ...         lattice=Lattice(0.360, 0.360, 0.360, 90, 90, 90)
    ...     ),
    ...     Structure(
    ...         title="ferrite",
    ...         atoms=[Atom("fe", [0] * 3)],
    ...         lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90)
    ...     )
    ... ]
    >>> xmap = CrystalMap(
    ...     rotations=Rotation.from_axes_angles([0, 0, 1], np.linspace(0, np.pi, n)),
    ...     phase_id=np.array([0, 0, 1, 1, 0, 1]),
    ...     x=coords["x"],
    ...     y=coords["y"],
    ...     phase_list=PhaseList(space_groups=[225, 229], structures=structures),
    ...     prop={"score": np.random.random(n)},
    ...     scan_unit="nm",
    ... )
    >>> xmap
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        0     3 (50.0%)  austenite        Fm-3m         m-3m                 432    tab:blue
        1     3 (50.0%)    ferrite        Im-3m         m-3m                 432  tab:orange
    Properties: score
    Scan unit: nm

    Data in a crystal map can be selected in multiple ways. Let's
    demonstrate this on a dual phase dataset available in the
    :mod:`~orix.data` module

    >>> from orix import data
    >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
    >>> xmap
    Phase   Orientations       Name  Space group  Point group  Proper point group       Color
        1   5657 (48.4%)  austenite         None          432                 432    tab:blue
        2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap.shape
    (100, 117)

    Selecting based on coordinates, passing ranges (slices), integers or
    both

    >>> xmap2 = xmap[20:40, 50:60]
    >>> xmap2
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1   148 (74.0%)  austenite         None          432                 432    tab:blue
        2    52 (26.0%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (20, 10)
    >>> xmap2 = xmap[20:40, 3]
    >>> xmap2
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1    16 (80.0%)  austenite         None          432                 432    tab:blue
        2     4 (20.0%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (20, 1)

    Note that 1-dimensions are NOT removed

    >>> xmap2 = xmap[10, 10]
    >>> xmap2
    Phase  Orientations     Name  Space group  Point group  Proper point group       Color
        2    1 (100.0%)  ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (1, 1)

    Select by phase name(s)

    >>> xmap2 = xmap["austenite"]
    >>> xmap2
    Phase  Orientations       Name  Space group  Point group  Proper point group     Color
        1  5657 (100.0%)  austenite         None          432                 432  tab:blue
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (100, 117)
    >>> xmap["austenite", "ferrite"]
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1  5657 (48.4%)  austenite         None          432                 432    tab:blue
        2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um

    Select by indexed and not indexed data

    >>> xmap["indexed"]
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1  5657 (48.4%)  austenite         None          432                 432    tab:blue
        2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap["not_indexed"]
    No data.

    Select with a boolean array (possibly chained)

    >>> xmap[xmap.dp > 0.81]
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1  4092 (44.8%)  austenite         None          432                 432    tab:blue
        2  5035 (55.2%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap[(xmap.iq > np.mean(xmap.iq)) & (xmap.phase_id == 1)]
    Phase  Orientations       Name  Space group  Point group  Proper point group     Color
        1  1890 (100.0%)  austenite         None          432                 432  tab:blue
    Properties: iq, dp
    Scan unit: um
    """

    def __init__(
        self,
        rotations: "Rotation | CrystalMap",
        phase_id: np.ndarray | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        phase_list: PhaseList | None = None,
        prop: dict | None = None,
        scan_unit: str | None = "px",
        indices: np.ndarray | None = None,
        spacing: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        indexing_order: str = "zyx",
    ) -> None:
        if isinstance(rotations, CrystalMap):
            return CrystalMap.__init__(
                self,
                rotations.rotations,
                rotations.phase_id,
                rotations.x,
                rotations.y,
                rotations.z,
                rotations.phases,
                rotations.prop,
                rotations.scan_unit,
            )

        # Set data size based on either 'x', 'indices', or 'rotations'.
        if x is not None:
            data_size = x.size
        elif indices is not None:
            data_size = np.atleast_2d(indices).shape[-1]
        else:
            data_size = rotations.shape[0]

        # set indexing convention, which defines how "flatten", "meshgrid",
        # and "__getitem__" operate on the data.
        if indexing_order in ["xyz", "zyx"]:
            self.indexing_order = indexing_order
        else:
            raise ValueError(
                f"indexing_order must be 'xyz' or 'zyz'm not {indexing_order}"
            )

        # Set rotations.
        if not isinstance(rotations, Rotation):
            raise ValueError(
                f"rotations must be of type {Rotation}, not {type(rotations)}."
            )
        if rotations.size == data_size:
            # TODO: after updating object3d.flatten, this should include an "F"
            # flag when indexing_order="xyz".
            self._rotations = rotations.flatten()
        elif rotations.shape[0] == data_size:
            self._rotations = rotations
        else:
            raise ValueError(
                "'rotations' has a shape of {}. Either the ".format(rotations.shape)
                + "total size or the size of the first indicies of 'rotations' must"
                + "match the size of the CrystalMap, {}".format(data_size)
            )
        # past this point, the size of the data should be queried via `self.size`

        # Set unique ingeters identifing each voxel.
        self._id = np.arange(self.size)

        # Set phase IDs
        if phase_id is None:  # Assume single phase data
            phase_id = np.zeros(self.size)
        self.phase_id = phase_id

        # Set spatial coordinates
        if indices is not None:
            self._set_grid_from_indices(indices, spacing, origin)
        else:
            self._set_grid_from_coords(x, y, z)

        # set the phase information
        if phase_list is None:
            phase_list = PhaseList(ids=np.unique(self.phase_id))
        self.phases = phase_list

        # Set scan unit
        self.scan_unit = scan_unit

        # Set properties
        if prop is None:
            prop = {}
        self._prop = CrystalMapProperties(prop, id=self.id)

    def _set_grid_from_indices(
        self, indices: np.ndarray, spacing: np.ndarray, origin: np.ndarray
    ):
        """Sets the values for _layer, _row, _column, _dx, _dy, and _dz based on
        integer indices values"""
        indices = np.atleast_2d(indices)
        if len(indices.shape) != 2:
            ValueError("indices must be interpretable as a two-dimensional array")
        if not np.issubdtype(indices.dtype, np.integer):
            ValueError("indices must be an array of integers")
        dims = indices.shape[0]
        if not np.isin(dims, (1, 2, 3)):
            ValueError("indices must have a shape of (d, N), where 'd' is 1, 2, or 3")
        size = indices.shape[1]
        if size != self._rotations.shape[0]:
            ValueError("There must be the same number of indices as there are pixels")

        if spacing is None:
            spacing = np.ones(dims, dtype=np.float32)
        spacing = np.atleast_1d(spacing).flatten()
        if spacing.size != dims:
            ValueError(
                "Spacing should have {} values, not {}".format(dims, spacing.size)
            )

        if origin is None:
            spacing = np.zeros(dims, dtype=np.float32)
        spacing = np.atleast_1d(spacing).flatten()
        if spacing.size != dims:
            ValueError(
                "origin should have {} values, not {}".format(dims, spacing.size)
            )

        if self._indexing_order == "xyz":
            indices = indices[::-1, :]
            origin = origin[::-1]

        # Assign data AFTER all checks and calculations are completed.
        self._layer = indices[-2] if dims > 2 else None
        self._row = indices[-1] if dims > 1 else None
        self._column = indices[0]

        self._dz = spacing[-2] if dims > 2 else 0
        self._dy = spacing[-1] if dims > 1 else 0
        self._dx = spacing[0]

        self._zmin = origin[-2] if dims > 2 else 0
        self._ymin = origin[-1] if dims > 1 else 0
        self._xmin = origin[0]

        return

    def _set_grid_from_coords(self, x, y, z):
        """Sets the values for _layer, _row, _column, _dx, _dy, and _dz based on
        xyz spatial coordinates"""

        # Reminder: Default numpy convention implies zyx (layer/row/column) ordering.
        if y is None and z is not None:
            ValueError("y cannot be None if z is not None")
        if x is None and y is not None:
            ValueError("x cannot be None if y is not None")
        if x is None and z is not None:
            ValueError("x cannot be None if z is not None")

        if z is None:
            dz = 0
            zmin = 0
            layer = None
        elif not np.issubdtype(z.dtype, np.number):
            ValueError("z must be interpretable as a 1d array of floats or ints")
        else:
            z = np.atleast_1d(z).flatten()
            dz = _step_size_from_coordinates(z)
            zmin = np.min(z)
            layer = np.around((z - zmin) / dz, 0).astype(int)

        if y is None:
            dy = 0
            ymin = 0
            row = None
        elif not np.issubdtype(y.dtype, np.number):
            ValueError("y must be interpretable as a 1d array of floats or ints")
        else:
            y = np.atleast_1d(y).flatten()
            dy = _step_size_from_coordinates(y)
            ymin = np.min(y)
            row = np.around((y - ymin) / dy, 0).astype(int)

        if x is None:
            x = np.arange(self._rotations.shape[0], dtype=int)
        elif not np.issubdtype(x.dtype, np.number):
            ValueError("x must be interpretable as a 1d array of floats or ints")
        x = np.atleast_1d(x).flatten()
        dx = _step_size_from_coordinates(x)
        xmin = np.min(x)
        column = np.around((x - xmin) / dx, 0).astype(int)

        # Assign data AFTER all checks and calculations are completed.
        self._layer = layer
        self._row = row
        self._column = column

        self._dz = dz
        self._dy = dy
        self._dx = dx

        self._zmin = zmin
        self._ymin = ymin
        self._xmin = xmin

        return

    @property
    def id(self) -> np.ndarray:
        """Return the unique integer ID for each voxel in the CrystalMap."""
        return self._id

    @id.setter
    def id(self, id_array: np.ndarray | str):
        """set a unique integer ID for each voxel in the CrstalMap."""
        id_array = np.atleast_1d(id_array).flatten().astype(int)
        if id_array.size != self.size:
            ValueError("IDs must be same the same size as the CrystalMap")
        if np.max(np.unique(id_array, return_counts=True)[1]) != 1:
            ValueError(
                "IDs must be an array of unique of integers. If attempting to assign "
                + "feature ids or property data, consider using CrystalMap.prop."
            )
        self._id = id_array

    @property
    def size(self) -> int:
        """Return the total number of voxels in the CrystalMap."""
        return self.rotations.shape[0]

    @property
    def shape(self) -> tuple:
        """Return the shape of the minimum rectilinear grid within which all
        voxels exist.

        Notes
        -----
        np.prod(xmap.shape) is NOT necessarily the same as xmap.size, as
        CrystalMaps can be subsets of rectilinear maps. for example, a grain
        sectioned from a larger dataset.
        """
        nx = None if self.column is None else np.max(self.column) - np.min(self.column)
        ny = None if self.row is None else np.max(self.row) - np.min(self.row)
        nz = None if self.layer is None else np.max(self.layer) - np.min(self.layer)
        if self._indexing_order == "xyz":
            all_n = [nx, ny, nz]
        else:
            all_n = [nz, ny, nx]
        return tuple(int(n + 1) for n in all_n if n is not None)

    @property
    def ndim(self) -> int:
        """Return the dimensionality of the voxel grid."""
        if self._column is None:
            return 0
        elif self._row is None:
            return 1
        elif self._layer is None:
            return 2
        else:
            return 3

    @property
    def x(self) -> np.ndarray | None:
        """Return the x coordinates of voxels in the CrystalMap.

        Note
        ----
        This returns the spatial coordinates, ie the 'x' component of it's xyz
        location on a scatter plot. for the x indices of the underlying grid,
        use `CrystalMap.col`

        Additionally, these values are calculated from the grid indices, and
        cannot be directly changed. To change the x-axis spacing or location,
        modify `Crystalmap.dx` or `CrystalMap.xmin`, respectively"""
        if self._column is None:
            return
        else:
            return (self._column * self._dx) + self._xmin

    @property
    def y(self) -> np.ndarray | None:
        """Return the y coordinates of voxels in the CrystalMap.

        Note
        ----
        This returns the spatial coordinates, ie the 'y' component of it's xyz
        location on a scatter plot. for the y indices of the underlying grid,
        use `CrystalMap.row`

        Additionally, these values are calculated from the grid indices, and
        cannot be directly changed. To change the y-axis spacing or location,
        modify `Crystalmap.dy` or `CrystalMap.ymin`, respectively"""
        if self._row is None:
            return
        else:
            return (self._row * self._dy) + self._ymin

    @property
    def z(self) -> np.ndarray | None:
        """Return the z coordinates of voxels in the CrystalMap.

        Note
        ----
        This returns the spatial coordinates, ie the 'z' component of it's xyz
        location on a scatter plot. for the z indices of the underlying grid,
        use `CrystalMap.layer`

        Additionally, these values are calculated from the grid indices, and
        cannot be directly changed. To change the z-axis spacing or location,
        modify `Crystalmap.dz` or `CrystalMap.zmin`, respectively"""
        if self._layer is None:
            return
        else:
            return (self._layer * self._dz) + self._zmin

    @property
    def crds(self):
        """Returns all defined spatial coordinates as a dictionary"""
        keys = ["z", "y", "x"]
        vals = self.z, self.y, self.x
        return {aa: bb for aa, bb in zip(keys, vals) if bb is not None}

    @property
    def dx(self) -> float:
        """Either get or set the x coordinate step size."""
        return self._dx

    @dx.setter
    def dx(self, dx: float | int):
        if self.column is None:
            ValueError("dx cannot be set when column is None")
        dx = np.asanyarray(dx).flatten()[0]
        if not np.isin(type(dx), np.number):
            ValueError("dx must be interpretable as an int or float")
        self._dx = dx

    @property
    def dy(self) -> float:
        """Either get or set the y coordinate step size."""
        return self._dy

    @dy.setter
    def dy(self, dy: float | int):
        if self.row is None:
            ValueError("dy cannot be set when row is None")
        dy = np.asanyarray(dy).flatten()[0]
        if not np.isin(type(dy), np.number):
            ValueError("dy must be interpretable as an int or float")
        self._dy = dy

    @property
    def dz(self) -> float:
        """Either get or set the z coordinate step size."""
        return self._dz

    @dz.setter
    def dz(self, dz: float | int):
        if self.layer is None:
            ValueError("dz cannot be set when row is None")
        dz = np.asanyarray(dz).flatten()[0]
        if not np.isin(type(dz), np.number):
            ValueError("dz must be interpretable as an int or float")
        self._dz = dz

    @property
    def steps(self):
        """Returns the all defined step sizes as a dictionary"""
        keys = ["dz", "dy", "dx"]
        vals = self.dz, self.dy, self.dx
        return {aa: bb for aa, bb in zip(keys, vals) if bb is not None}

    @property
    def column(self) -> np.ndarray | None:
        """Either get or set the column (x-axis) indices for each voxel
        in the CrystalMap.

        This is identical to self.col"""
        return self.col

    @column.setter
    def column(self, value: np.ndarray):
        self.col(value)

    @property
    def col(self) -> np.ndarray | None:
        """Either get or set the column (x-axis) indices for each voxel
        in the CrystalMap."""
        # TODO: re-add example
        return self._column

    @col.setter
    def col(self, value: np.ndarray):
        value = np.asanyarray(value).flatten().astype(int)
        if value.size != self.size:
            ValueError("input must be the same size as the CrystalMap.")
        self._column = value

    @property
    def row(self) -> np.ndarray | None:
        """Either get or set the row (y-axis) indices for each voxel
        in the CrystalMap."""
        # TODO: re-add example
        return self._row

    @row.setter
    def row(self, value: np.ndarray):
        value = np.asanyarray(value).flatten().astype(int)
        if value.size != self.size:
            ValueError("input must be the same size as the CrystalMap.")
        self._row = value

    @property
    def layer(self) -> np.ndarray | None:
        """Either get or set the layer (z-axis) indices for each voxel
        in the CrystalMap."""
        # TODO: re-add example
        return self._layer

    @layer.setter
    def layer(self, value: np.ndarray):
        value = np.asanyarray(value).flatten().astype(int)
        if value.size != self.size:
            ValueError("input must be the same size as the CrystalMap.")
        self._layer = value

    @property
    def indices(self):
        """Returns all defined spatial coordinates as n N-by-d numpy array.

        For example, a two-dimensional array would be an N-by-2, and a
        three-dimensional an N-by-3, where N is the number of voxels in the
        CrystalMap."""
        vals = [self.layer, self.row, self.col]
        if all(x is None for x in vals):
            return
        if self.indexing_order == "xyz":
            vals = [self.col, self.row, self.layer]
        vals = [x for x in vals if x is not None]
        return np.stack(vals).T

    @property
    def phase_id(self) -> np.ndarray:
        """Return or set the phase IDs of points in data.

        Parameters
        ----------
        value : numpy.ndarray or int
            Phase ID of points in data.
        """
        return self._phase_id

    @phase_id.setter
    def phase_id(self, values: np.ndarray | int) -> None:
        values = np.atleast_1d(values).flatten().astype(int)
        if values.size != self.size:
            ValueError("phase_id array must be same the same size as the CrystalMap")
        if np.min(values) < -1:
            ValueError(
                "All values in phase_ids must be zero or higher for"
                + "indexed voxels, or -1 for unindexed voxels."
            )
        if hasattr(self, "phases"):
            if np.min(values) == 0 and "not_indexed" not in self.phases.names:
                self.phases.add_not_indexed()
        self._phase_id = values

    @property
    def phases(self) -> PhaseList:
        """Return or set the list of phases.

        Parameters
        ----------
        value : PhaseList
            Phase list with at least as many phases as unique phase IDs
            in :attr:`phase_id`.

        Raises
        ------
        ValueError
            If there are fewer phases in the list than unique phase IDs.

        Note
        ----
        Because CrystalMap includes methods for altering phase information,
        imported PhaseLists are copied before being added to the CrystalMap.
        """
        return self._phases

    @phases.setter
    def phases(self, value: PhaseList) -> None:

        phase_list = value.deepcopy()
        unique_phase_ids_in_xmap = np.unique(self.phase_id)

        # Create a new dictionary of phases, so missing inormation necessary for
        # plotting can be filled in where needed. Also add defined but unused phases.
        phase_dict = {}
        mpl_colors = list(get_named_matplotlib_colors()[0].keys())
        new_colors = np.delete(mpl_colors, np.isin(mpl_colors, phase_list.colors))
        ci = 0
        for i in np.arange(np.max(unique_phase_ids_in_xmap) + 1):
            if i in phase_list.ids:
                phase_dict[i] = phase_list[i]
            elif i in unique_phase_ids_in_xmap:  # but NOT in pase_list (ie, missing).
                raise Warning(
                    f"The phase ID {i} exists in the CrystalMap but not in the "
                    + "PhaseMap. A new empty Phase has been appended to "
                    + "CrystalMap.phases."
                )
                name = "New_Phase_{}".format(i)
                phase_dict[i] = Phase(name=name, color=new_colors[ci])
                ci += 1

        ordered_phase_list = PhaseList(phase_dict)
        if np.isin(-1, unique_phase_ids_in_xmap):
            ordered_phase_list.add_not_indexed()
        self._phases = ordered_phase_list

    @property
    def phases_in_data(self) -> PhaseList:
        """Return the list of phases in data.

        See Also
        --------
        phases

        Notes
        -----
        Can be useful when there are phases in :attr:`phases` which are
        not in the data.
        """
        unique_ids = np.unique(self.phase_id)
        phase_list = self.phases[np.intersect1d(unique_ids, self.phases.ids)]
        if isinstance(phase_list, Phase):  # One phase in data
            # Get phase ID so it carries over to the new `PhaseList`
            # instance
            phase = phase_list  # Since it's actually a single phase
            phase_id = self.phases.id_from_name(phase.name)
            return PhaseList(phases=phase, ids=phase_id)
        else:  # Multiple phases in data
            return phase_list

    def remove_unused_phases(self):
        """removes any phases not associated with indexed voxels"""
        self.phases = self.phases_in_data

    @property
    def rotations(self) -> Rotation:
        """Get or set the rotations in the CrystalMap."""
        return self._rotations

    @rotations.setter
    def rotations(self, rots):
        if not isinstance(rots, Rotation):
            ValueError("rotations must be a orix.quaternion.Rotation object.")
        if rots.shape != self.rotations.shape:
            ValueError(f"rotations shape {rots.shape} did not match existing shape.")
        self._rotations.data = rots.data  # auto_normalizes quaternions.

    @property
    def rotations_per_point(self) -> int:
        """Return the number of rotations per data point in data."""
        return self.rotations.size // self.is_indexed.size

    @property
    def rotations_shape(self) -> tuple:
        """Return the shape of :attr:`rotations`.

        Notes
        -----
        Map shape and possible multiple rotations per point are
        accounted for. 1-dimensions are squeezed out.
        """
        return tuple(i for i in self.shape + (self.rotations_per_point,) if i != 1)

    @property
    def orientations(self) -> Orientation:
        """Return orientations (rotations respecting symmetry), in data.

        Raises
        ------
        ValueError
            When the (potentially sliced) map has more than one phase in
            the data.

        Examples
        --------
        >>> from orix import data
        >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
        >>> xmap
        Phase   Orientations       Name  Space group  Point group  Proper point group       Color
            1   5657 (48.4%)  austenite         None          432                 432    tab:blue
            2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> xmap["austenite"].orientations
        Orientation (5657,) 432
        [[ 0.8686  0.3569 -0.2749 -0.2064]
         [ 0.8681  0.3581 -0.2744 -0.2068]
         [ 0.8684  0.3578 -0.2751 -0.2052]
         ...
         [ 0.9639  0.022   0.0754 -0.2545]
         [ 0.8854  0.3337 -0.2385  0.2187]
         [ 0.885   0.3341 -0.2391  0.2193]]
        """
        phases = self.phases_in_data
        if phases.size == 1:
            # Extract top matching rotations per point, if more than one
            if self.rotations_per_point > 1:
                rotations = self.rotations[:, 0]
            else:
                rotations = self.rotations
            # Point group can be None, so it cannot be passed upon
            # initialization to Orientation but has to be set afterwards
            # to trigger the checks
            orientations = Orientation(rotations)
            orientations.symmetry = phases[:].point_group
            return orientations
        else:
            raise ValueError(
                f"Data has the phases {phases.names}, however, you are executing a "
                "command that only permits one phase."
            )

    @property
    def is_indexed(self) -> np.ndarray:
        """Return whether points in data are indexed."""
        return self.phase_id != -1

    @property
    def all_indexed(self) -> np.ndarray:
        """Return whether all points in data are indexed."""
        return np.count_nonzero(self.is_indexed) == self.is_indexed.size

    @property
    def prop(self) -> CrystalMapProperties:
        """Return the data properties in each data point."""
        self._prop.id = self.id
        return self._prop

    @property
    def _coordinate_axes(self) -> dict:
        """Return which data axis corresponds to which coordinate."""
        keys = np.arange(self.ndim)
        if self.indexing_order == "xyz":
            vals = self.indexing_order[: self.ndim]
        else:
            vals = self.indexing_order[-self.ndim :]
        return dict(zip(keys, vals))

    def __getattr__(self, item) -> Any:
        """Return an attribute in the :attr:`prop` dictionary directly
        from the ``CrystalMap`` instance.

        Called when the default attribute access fails with an
        ``AttributeError``.
        """
        if item in self.__getattribute__("_prop"):
            # Calls CrystalMapProperties.__getitem__()
            return self.prop[item]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a class instance attribute."""
        if hasattr(self, "_prop") and name in self._prop:
            # Calls CrystalMapProperties.__setitem__()
            self.prop[name] = value
        else:
            return object.__setattr__(self, name, value)

    def __getitem__(self, key: str | slice | tuple | int | np.ndarray) -> CrystalMap:
        """return a subset of the CrystalMap instance. See the docstring of
        ``__init__()`` for examples.

        If `key` is a string or tuple of strings, the input is intepreted as a
        phase name (or names), and the subset of voxels with matching phase(s)
        is returned.

        If `key` is a numpy array of booleans, the input is interpreted as a mask.
        If the mask is one-dimensional, the masking is done directly on the IDs.
        Otherwise, it is done relative to the layer/row/column values.

        if 'key' is a numpy integer array of size (self.size,self.ndims), or
        a tuple of length (self.ndims) containing 1d numpy arrays of integers,
        they are interpreted as indicies referring to the values of layer, row,
        and column, and masking is performed accordingly.

        If none of the above, 'key' is assumed to be some form of numpy-like slicing,
        and slicing is performed based on the layer/row/column grid.
        """
        # Create a placeholder varaible for the boolean mask.
        data_to_keep = None

        # make non-iterable inputs iterable.
        if isinstance(key, (str, slice, int)):
            key = (key,)

        # Case 1: these are phase names.
        if all(type(x) is str for x in key):
            data_to_keep = np.zeros(self.size, dtype=bool)
            for k in key:
                for phase_id, phase in self.phases:
                    if k == phase.name:
                        data_to_keep[self.phase_id == phase_id] = True
                    elif k.lower() == "indexed":
                        # Add all indexed phases to data
                        data_to_keep[self.phase_id != -1] = True
                    elif k.lower() == "not_indexed":
                        data_to_keep[self.phase_id == -1] = True
                    else:
                        raise Warning("phase {} was not found in self.phases".format(k))

        # Case 2: it's a boolean array for masking.
        if isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # Case 2a: it's a boolean array operating directly on the rotations
            if len(key.shape) == 1:
                data_to_keep = key
            # Case 2b: it's a boolean array operating on the grid coordinates
            else:
                key = np.where(key)  # convert to tuple of (zyx) indices.

        # Case 3a: it's grid indices as an array.
        if isinstance(key, np.ndarray) and key.dtype == np.integer:
            if key.shape[0] == self.ndim and key.shape[1] == self.size:
                key = tuple(x for x in key)

        # Case 3b: it a tuple of arrays representing grid indices.
        if isinstance(key, tuple) and all(type(x) is np.ndarray for x in key):
            # NOTE: this also handles cases 2b and 3a above; hence 'if', not 'elif'
            data_to_keep = np.zeros(self.size, dtype=bool)
            if self.indexing_order == "xyz":
                key = key[::-1]
            data_to_keep = np.ones(self.shape, dtype=bool)
            if len(key) == 3:
                data_to_keep[~np.isin(self.layer, key[-3])] = False
            if len(key) >= 2:
                data_to_keep[~np.isin(self.row, key[-2])] = False
            data_to_keep[~np.isin(self.row, key[-1])] = False

        # Case 4: key is interpretable as a form of numpy-like slicing.
        if np.all(isinstance(x, (slice, int)) for x in key):
            data_to_keep = np.ones(self.size, dtype=bool)
            slices = [slice(None, None, None)] * self.ndim
            for i, k in enumerate(key):
                slices[i] = k
            if self._indexing_order == "xyz":
                slices = slices[::-1]
            for axis, choice in zip((self.layer, self.row, self.col), slices):
                if isinstance(choice, int):
                    data_to_keep[axis != choice] = False
                else:
                    if choice.stop is not None:
                        data_to_keep[axis > choice.stop] = False
                    if choice.start is not None:
                        data_to_keep[axis <= choice.start] = False
                        axis = axis - choice.start
                    if choice.step is not None:
                        data_to_keep[axis % choice.start != 0] = False

        if data_to_keep is None:
            ValueError(
                "`key` could not be interpreted as a phase name, boolean"
                + " mask, indices, and/or slices."
            )
        else:
            # create masked values one at a time to save on memory.
            new_layer = self.layer[data_to_keep] if self.layer is not None else None
            new_row = self.row[data_to_keep] if self.row is not None else None
            new_col = self.col[data_to_keep] if self.col is not None else None
            if self.indexing_order == "xyz":
                ind_list = [new_col, new_row, new_layer]
            else:
                ind_list = [new_layer, new_row, new_col]
            del new_layer, new_row, new_col
            new_indices = np.stack([x for x in ind_list if x is not None], axis=0)
            ind_min = np.min(new_indices, axis=0)
            new_indices -= ind_min

            new_origin = self.origin - (ind_min * self.steps)
            new_r = self.rotations[data_to_keep]
            new_phase_id = self.phase_id[data_to_keep]
            new_prop = self.prop  # TODO: fix this

            new_xmap = CrystalMap(
                rotations=new_r,
                phase_id=new_phase_id,
                phase_list=self.phase_list,
                prop=new_prop,
                scan_unit=self.scan_unit,
                indices=new_indices,
                spacing=self.steps,
                origin=new_origin,
                indexing_order=self.indexing_order,
            )
            return new_xmap

    def __repr__(self) -> str:
        """Return a nice representation of the data."""
        if self.size == 0:
            return "No data."

        phases = self.phases_in_data
        phase_ids = self.phase_id

        # Ensure attributes set to None are treated OK
        names = ["None" if not name else name for name in phases.names]
        sg_names = ["None" if not i else i.short_name for i in phases.space_groups]
        pg_names = ["None" if not i else i.name for i in phases.point_groups]
        ppg_names = [
            "None" if not i else i.proper_subgroup.name for i in phases.point_groups
        ]

        # Determine column widths
        unique_phases, phase_counts = np.unique(phase_ids, return_counts=True)
        p_sizes = [np.where(phase_ids == i)[0].size for i in unique_phases]
        id_len = 5
        ori_len = max(max([len(str(p_size)) for p_size in p_sizes]) + 9, 12)
        name_len = max(max([len(n) for n in names]), 4)
        sg_len = max(max([len(i) for i in sg_names]), 11)
        pg_len = max(max([len(i) for i in pg_names]), 11)
        ppg_len = max(max([len(i) for i in ppg_names]), 18)
        col_len = max(max([len(i) for i in phases.colors]), 5)

        # Column alignment
        align = ">"  # right ">" or left "<"

        # Header (note the two-space spacing)
        representation = (
            "{:{align}{width}}  ".format("Phase", width=id_len, align=align)
            + "{:{align}{width}}  ".format("Orientations", width=ori_len, align=align)
            + "{:{align}{width}}  ".format("Name", width=name_len, align=align)
            + "{:{align}{width}}  ".format("Space group", width=sg_len, align=align)
            + "{:{align}{width}}  ".format("Point group", width=pg_len, align=align)
            + "{:{align}{width}}  ".format(
                "Proper point group", width=ppg_len, align=align
            )
            + "{:{align}{width}}\n".format("Color", width=col_len, align=align)
        )

        # Overview of data for each phase
        for i, phase_id in enumerate(unique_phases):
            p_size = phase_counts[np.where(unique_phases == phase_id)][0]
            p_fraction = 100 * p_size / self.size
            ori_str = f"{p_size} ({p_fraction:.1f}%)"
            representation += (
                f"{phase_id:{align}{id_len}}  "
                + f"{ori_str:{align}{ori_len}}  "
                + f"{names[i]:{align}{name_len}}  "
                + f"{sg_names[i]:{align}{sg_len}}  "
                + f"{pg_names[i]:{align}{pg_len}}  "
                + f"{ppg_names[i]:{align}{ppg_len}}  "
                + f"{phases.colors[i]:{align}{col_len}}\n"
            )

        # Properties and spatial coordinates
        props = []
        for k in self.prop.keys():
            props.append(k)
        representation += "Properties: " + ", ".join(props) + "\n"

        # Scan unit
        representation += f"Scan unit: {self.scan_unit}"

        return representation

    def deepcopy(self) -> CrystalMap:
        """Return a deep copy using :func:`copy.deepcopy` function."""
        return copy.deepcopy(self)

    @classmethod
    def empty(
        cls,
        shape: int | tuple[int, int] | tuple[int] | None = None,
        step_sizes: float | tuple[float] | tuple[float, float] | None = None,
    ) -> CrystalMap:
        """Return a crystal map of a given 2D shape and step sizes with
        identity rotations.

        Parameters
        ----------
        shape
            Map shape. Default is a 2D map of shape (5, 10), i.e. with
            five rows and ten columns.
        step_sizes
            Map step sizes. If not given, it is set to 1 px in each map
            direction given by ``shape``.

        Returns
        -------
        xmap
            Crystal map.
        """
        d, n = create_coordinate_arrays(shape, step_sizes)
        d["rotations"] = Rotation.identity((n,))
        return cls(**d)

    def get_map_data(
        self,
        item: str | np.ndarray,
        decimals: int | None = None,
        fill_value: int | float | None = np.nan,
    ) -> np.ndarray:
        """Return an array of a class instance attribute, with values
        equal to ``False`` in :attr:`self.is_in_data` set to
        ``fill_value``, of map data shape.

        Parameters
        ----------
        item
            Name of the class instance attribute or a
            :class:`numpy.ndarray`.
        decimals
            Number of decimals to round data point values to. If not
            given, no rounding is done.
        fill_value
            Value to fill points not in the data with. Default is
            :class:`numpy.nan`.

        Returns
        -------
        output_array
            Array of the class instance attribute with points not in
            data set to ``fill_value``, of float data type.

        Notes
        -----
        Rotations and orientations should be accessed via
        :attr:`rotations` and :attr:`orientations`.

        If ``item`` is ``"orientations"`` or ``"rotations"`` and there
        are multiple rotations per point, only the first rotation is
        used. Rotations are returned as Euler angles.

        Examples
        --------
        >>> from orix import data
        >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
        >>> xmap
        Phase   Orientations       Name  Space group  Point group  Proper point group       Color
            1   5657 (48.4%)  austenite         None          432                 432    tab:blue
            2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> xmap.shape
        (100, 117)

        Get a 2D map in the correct shape of any attribute, ready for
        plotting

        >>> xmap.iq.shape
        (11700,)
        >>> iq = xmap.get_map_data("iq")
        >>> iq.shape
        (100, 117)
        """
        # Get full map shape
        map_shape = self._original_shape

        # Declare array of correct shape, accounting for RGB
        # TODO: Better account for `item.shape`, e.g. quaternions
        #  (item.shape[-1] == 4) in a more general way than here (not
        #  more if/else)!
        map_size = np.prod(map_shape)
        if isinstance(item, np.ndarray):
            array = np.empty(map_size, dtype=item.dtype)
            if item.shape[-1] == 3 and map_size > 3:  # Assume RGB
                map_shape += (3,)
                array = np.column_stack((array,) * 3)
        elif item in ["orientations", "rotations"]:  # Definitely RGB
            array = np.empty(map_size, dtype=np.float64)
            map_shape += (3,)
            array = np.column_stack((array,) * 3)
        else:
            array = np.empty(map_size, dtype=np.float64)

        # Enter non-masked values into array
        if isinstance(item, np.ndarray):
            # TODO: Account for 2D map with more than one value per point
            array[self.is_in_data] = item
        elif item in ["orientations", "rotations"]:
            if item == "rotations":
                # Use only the top matching rotation per point
                if self.rotations_per_point > 1:
                    rotations = self.rotations[:, 0]
                else:
                    rotations = self.rotations
                array[self.is_in_data] = rotations.to_euler()
            else:  # item == "orientations"
                # Fill in orientations per phase
                for i, _ in self.phases_in_data:
                    phase_mask = (self._phase_id == i) * self.is_in_data
                    phase_mask_in_data = self.phase_id == i
                    array[phase_mask] = self[phase_mask_in_data].orientations.to_euler()
        else:  # String
            data = self.__getattr__(item)
            if data is None:
                raise ValueError(f"{item} is {data}.")
            else:
                # TODO: Account for 2D map with more than one value per point
                array[self.is_in_data] = data
                array = array.astype(data.dtype)

        # Slice and reshape array
        slices = self._data_slices_from_coordinates()
        reshaped_array = array.reshape(map_shape)
        sliced_array = reshaped_array[slices]

        # Reshape and slice mask with points not in data
        if array.shape[-1] == 3 and map_size > 3:  # RGB
            not_in_data = np.dstack((~self.is_in_data,) * 3)
        else:  # Scalar
            not_in_data = ~self.is_in_data
        not_in_data = not_in_data.reshape(map_shape)[slices]

        # Fill points not in data with the fill value
        if not_in_data.any():
            if fill_value is None or fill_value is np.nan:
                sliced_array = sliced_array.astype(np.float64)
            sliced_array[not_in_data] = fill_value

        # Round values
        if decimals is not None:
            output_array = np.round(sliced_array, decimals=decimals)
        else:  # np.issubdtype(array.dtype, np.bool_):
            output_array = sliced_array

        return output_array

    def plot(
        self,
        value: np.ndarray | str | None = None,
        overlay: str | np.ndarray | None = None,
        scalebar: bool | None = None,
        scalebar_properties: dict | None = None,
        legend: bool = True,
        legend_properties: dict | None = None,
        colorbar: bool = False,
        colorbar_label: str | None = None,
        colorbar_properties: dict | None = None,
        remove_padding: bool = False,
        return_figure: bool = False,
        axis: int | None = None,
        layer: int | None = None,
        figure_kwargs: dict | None = None,
        **kwargs,
    ) -> mfigure.Figure | None:
        r"""Plot a 2D map with any crystallographic map property as map
        values.

        Wraps :meth:`matplotlib.axes.Axes.imshow`: see that method for
        relevant keyword arguments.

        Parameters
        ----------
        value
            An array or an attribute string to plot. If not given, a
            phase map is plotted.
        overlay
            Name of map property or a property array to use in the
            alpha (RGBA) channel. The property range is adjusted for
            maximum contrast. Not used if not given.
        scalebar
            Whether to add a scalebar along the horizontal map
            dimension. If not given, a scalebar is added if
            :mod:`matplotlib-scalebar` is installed.
        scalebar_properties
            Keyword arguments passed to
            :class:`matplotlib_scalebar.scalebar.ScaleBar`.
        legend
            Whether to add a legend to the plot. This is only
            implemented for a phase plot (in which case default is
            ``True``).
        legend_properties
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.legend`.
        colorbar
            Whether to add an opinionated colorbar (default is
            ``False``).
        colorbar_label
            Label of colorbar.
        colorbar_properties
            Keyword arguments passed to
            :meth:`orix.plot.CrystalMapPlot.add_colorbar`.
        remove_padding
            Whether to remove white padding around figure (default is
            ``False``).
        return_figure
            Whether to return the figure (default is ``False``).
        axis
            For 3D xmap, axis on which to plot 2D slice.
        layer
            For 3D xmap, layer on defined axis to plot 2D slice.
        figure_kwargs
            Keyword arguments passed to
            :func:`matplotlib.pyplot.subplots`.
        **kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig
            The created figure, returned if ``return_figure=True``.

        See Also
        --------
        matplotlib.axes.Axes.imshow
        orix.plot.CrystalMapPlot.plot_map
        orix.plot.CrystalMapPlot.add_scalebar
        orix.plot.CrystalMapPlot.add_overlay
        orix.plot.CrystalMapPlot.add_colorbar

        Examples
        --------
        >>> from orix import data
        >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
        >>> xmap
        Phase   Orientations       Name  Space group  Point group  Proper point group       Color
            1   5657 (48.4%)  austenite         None          432                 432    tab:blue
            2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um

        Plot phase map

        >>> xmap.plot()

        Remove padding and return the figure (e.g. to be saved)

        >>> fig = xmap.plot(remove_padding=True, return_figure=True)

        Plot a dot product (similarity score) map

        >>> xmap.plot("dp", colorbar=True, colorbar_label="Dot product", cmap="gray")
        """
        # Register "plot_map" projection with Matplotlib
        import orix.plot.crystal_map_plot

        if figure_kwargs is None:
            figure_kwargs = {}

        fig, ax = plt.subplots(subplot_kw={"projection": "plot_map"}, **figure_kwargs)
        ax.plot_map(
            self,
            value=value,
            scalebar=scalebar,
            scalebar_properties=scalebar_properties,
            legend=legend,
            legend_properties=legend_properties,
            axis=axis,
            layer=layer,
            **kwargs,
        )

        if overlay is not None:
            ax.add_overlay(self, overlay)

        if remove_padding:
            ax.remove_padding()

        if colorbar:
            if colorbar_properties is None:
                colorbar_properties = dict()
            ax.add_colorbar(label=colorbar_label, **colorbar_properties)

        if return_figure:
            return fig

    def _xmap_slice_from_axis(self, axis: int, layer: int) -> "CrystalMap":
        """Returns a 2D slice of a CrystalMap object along a given axis.

        Parameters
        ----------
        axis
            For 3D xmap, axis on which to plot 2D slice.
        layer
            For 3D xmap, layer on defined axis to plot 2D slice.

        Returns
        -------
        CrystalMap
            2D CrystalMap slice.
        """
        return self[(slice(None),) * (axis % self.ndim) + (slice(layer, layer + 1),)]

    def _data_slices_from_coordinates(self, only_is_in_data: bool = True) -> tuple:
        """Return a slices defining the current data extent in all
        directions.

        Parameters
        ----------
        only_is_in_data
            Whether to determine slices of points in data or all points.
            Default is ``True``.

        Returns
        -------
        slices
            Data slice in each existing direction in (y, x) order.
        """
        if only_is_in_data:
            coordinates = self._coordinates
        else:
            coordinates = self._all_coordinates
        slices = _data_slices_from_coordinates(coordinates, self._step_sizes)
        return slices

    def _data_shape_from_coordinates(self, only_is_in_data: bool = True) -> tuple:
        """Return data shape based upon coordinate arrays.

        Parameters
        ----------
        only_is_in_data
            Whether to determine shape of points in data or all points.
            Default is ``True``.

        Returns
        -------
        data_shape
            Shape of data in each existing direction in (y, x) order.
        """
        data_shape = []
        for dim_slice in self._data_slices_from_coordinates(only_is_in_data):
            data_shape.append(dim_slice.stop - dim_slice.start)
        return tuple(data_shape)


def _data_slices_from_coordinates(
    coords: dict[str, np.ndarray], steps: dict[str, float] | None = None
) -> tuple[slice]:
    """Return a list of slices defining the current data extent in all
    directions.

    Parameters
    ----------
    coords
        Dictionary with coordinate arrays.
    steps
        Dictionary with step sizes in each direction. If not given, they
        are computed from *coords*.

    Returns
    -------
    slices
        Data slice in each direction.
    """
    if steps is None:
        steps = {
            "x": _step_size_from_coordinates(coords["x"]),
            "y": _step_size_from_coordinates(coords["y"]),
            "z": _step_size_from_coordinates(coords["z"]),
        }
    slices = []
    for coords, step in zip(coords.values(), steps.values()):
        if coords is not None and step != 0:
            c_min, c_max = np.min(coords), np.max(coords)
            i_min = int(np.around(c_min / step))
            i_max = int(np.around((c_max / step) + 1))
            slices.append(slice(i_min, i_max))
    slices = tuple(slices)
    return slices


def _step_size_from_coordinates(coordinates: np.ndarray) -> float:
    """Return step size in input *coordinates* array.

    Parameters
    ----------
    coordinates
        Linear coordinate array.

    Returns
    -------
    step_size
        Step size in *coordinates* array.
    """
    unique = np.sort(np.unique(coordinates))
    if unique.size != 1:
        deltas, counts = np.unique(unique[1:] - unique[:-1], return_counts=True)
        step_size = deltas[np.argmax(counts)]
    else:
        step_size = 0
    return step_size


def create_coordinate_arrays(
    shape: tuple[int] | tuple[int, int] | None = None,
    step_sizes: tuple[float] | tuple[float, float] | None = None,
) -> tuple[dict, int]:
    """Return flattened coordinate arrays from a given map shape and
    step sizes, suitable for initializing a
    :class:`~orix.crystal_map.CrystalMap`.

    Arrays for 1D or 2D maps can be returned.

    Parameters
    ----------
    shape
        Map shape. Default is a 2D map of shape (5, 10) with five rows
        and ten columns.
    step_sizes
        Map step sizes. If not given, it is set to 1 px in each map
        direction given by *shape*.

    Returns
    -------
    d
        Dictionary with keys ``"x"`` and ``"y"``, depending on the
        length of *shape*, with coordinate arrays.
    map_size
        Number of map points.

    Examples
    --------
    >>> from orix.crystal_map import create_coordinate_arrays
    >>> create_coordinate_arrays((2, 3))
    ({'x': array([0, 1, 2, 0, 1, 2]), 'y': array([0, 0, 0, 1, 1, 1])}, 6)
    >>> create_coordinate_arrays((3, 2))
    ({'x': array([0, 1, 0, 1, 0, 1]), 'y': array([0, 0, 1, 1, 2, 2])}, 6)
    >>> create_coordinate_arrays((2, 3), (1.5, 1.5))
    ({'x': array([0. , 1.5, 3. , 0. , 1.5, 3. ]), 'y': array([0. , 0. , 0. , 1.5, 1.5, 1.5])}, 6)
    """
    if not shape:
        shape = (5, 10)
    ndim = len(shape)
    if not step_sizes:
        step_sizes = (1,) * ndim

    if ndim == 3 or len(step_sizes) > 2:
        raise ValueError("Can only create coordinate arrays for 2D maps")

    # Set up as if a 2D map is to be returned
    dy, dx = (1,) * (2 - ndim) + step_sizes
    ny, nx = (1,) * (2 - ndim) + shape
    d = dict()

    # Add coordinate arrays depending on the number of map dimensions
    d["x"] = np.tile(np.arange(nx) * dx, ny).flatten()
    map_size = nx
    if ndim > 1:
        d["y"] = np.sort(np.tile(np.arange(ny) * dy, nx)).flatten()
        map_size *= ny

    return d, map_size
