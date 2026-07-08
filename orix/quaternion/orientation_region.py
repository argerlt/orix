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

from __future__ import annotations

import itertools

import numpy as np

from orix._utils import constants
from orix.quaternion.quaternion import Quaternion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, get_distinguished_points
from orix.vector.neo_euler import Rodrigues


def _get_large_cell_normals(dp):

    if dp.size == 0:
        return Rotation.empty()

    normals = Rodrigues.zero(dp.shape + (2,))
    planes1 = dp.axis * np.tan(dp.angle / 4)
    planes2 = -dp.axis * np.tan(dp.angle / 4) ** -1
    planes2.data[np.isnan(planes2.data)] = 0
    normals[:, 0] = planes1
    normals[:, 1] = planes2
    normals = Rotation.from_rodrigues(normals).flatten().unique(antipodal=False)

    _, inv = normals.axis.unique(return_inverse=True)
    axes_unique = []
    angles_unique = []
    for i in np.unique(inv):
        n = normals[inv == i]
        axes_unique.append(n.axis.data[0])
        angles_unique.append(np.max(n.angle))
    normals = Rotation.from_axes_angles(axes_unique, angles_unique)

    return normals


def get_proper_groups(start: Symmetry, end: Symmetry) -> tuple[Symmetry, Symmetry]:
    """Return the appropriate groups for the asymmetric domain
    calculation.

    Parameters
    ----------
    Gl
        First point group.
    Gr
        Second point group.

    Returns
    -------
    Gl
        First proper subgroup(s) or proper inversion subgroup(s), as
        appropriate.
    Gr
        Second proper subgroup(s) or proper inversion subgroup(s), as
        appropriate.

    Raises
    ------
    NotImplementedError
        If both groups are improper and neither contain an inversion,
        special consideration is needed which is not yet implemented in
        orix.
    """
    if start.is_proper and end.is_proper:
        return start, end
    elif start.is_proper and not end.is_proper:
        return start, end.proper_subgroup
    elif not start.is_proper and end.is_proper:
        return start.proper_subgroup, end
    else:
        if start.contains_inversion and end.contains_inversion:
            return start.proper_subgroup, end.proper_subgroup
        elif start.contains_inversion and not end.contains_inversion:
            return start.proper_subgroup, end.laue_proper_subgroup
        elif not start.contains_inversion and end.contains_inversion:
            return start.laue_proper_subgroup, end.proper_subgroup
        else:
            return start.laue_proper_subgroup, end.laue_proper_subgroup


class OrientationRegion(Rotation):
    """Some subset of the complete space of orientations.

    The complete orientation space represents every possible orientation
    of an object. The whole space is not always needed, for example if
    the orientation of an object is constrained or (most commonly) if
    the object is symmetrical. In this case, the space can be segmented
    using sets of Rotations representing boundaries in the space. This
    is clearest in the Rodrigues parametrisation, where the boundaries
    are planes, such as the example here: the asymmetric domain of an
    adjusted 432 symmetry.

    .. image:: /_static/img/orientation-region-Oq.png
       :width: 300px
       :alt: Boundaries of an orientation region in Rodrigues space.
       :align: center

    Rotations or orientations can be inside or outside of an orientation
    region.
    """

    # ------------------------ Dunder methods ------------------------ #

    def __gt__(self, other: OrientationRegion) -> np.ndarray:
        """Overridden greater than method.

        Applying this to an orientation will return only those that lie
        within the region.
        """
        c = Quaternion(self).dot_outer(Quaternion(other))
        inside = np.logical_or(
            np.all(np.greater_equal(c, -constants.eps9), axis=0),
            np.all(np.less_equal(c, constants.eps9), axis=0),
        )
        return inside

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_symmetry(
            cls,
            start: Symmetry = C1,
            end: Symmetry = C1,
            ) -> OrientationRegion:

        Parameters
        ----------
        s1
            First symmetry.
        s2
            Second symmetry. Default is C1 (the identity).

        Returns
        -------
        region
            The orientation region.
        """

        # Step 1: fundamental zones are only defined for the 121 proper
        # symmetries. Convert any improper symmetries to the most sensical
        # proper ones.
        # NOTE: the following logic could be simplified, but keeping
        # it in this format makes the logic for different cases more
        # clear.
        #
        # If one symmetry is proper, any improper rotations from the
        # second symmetry will fall outside the fundamental sector of the
        # disjoint group.
        if start.is_proper or end.is_proper:
            start = start.proper_subgroup
            end =end.proper_subgroup
        # If both are centrosymmetric, all improper operations have identical
        # proper versions, and the proper/improper regions are identical.
        elif start.contains_inversion and end.contains_inversion:
            start = start.proper_subgroup
            end =end.proper_subgroup
        # For all other cases, non-centrosymmetric groups should be converted
        # to laue groups, and then improper operators should be ignored.
        # This is equivalent to converting rotoinversions to rotations, and
        # allows the selection of the correct fundamental sector.
        else:
            if not start.contains_inversion:
                start = start.laue
            if not end.contains_inversion:
                end = end.laue
            start = start.proper_subgroup
            end =end.proper_subgroup

        # Step 2: define the bounding cells using the distinguished points.
        dp = get_distinguished_points(start, end)
        large_cell_normals = _get_large_cell_normals(dp)

        # Step 3: (only for misorientations) restrict the domain to the
        # fundamental sector of the pole figure of the shared symmetries.
        disjoint = start & end
        fz = disjoint.fundamental_zone()
        fz_normals = Rotation.from_axes_angles(fz, np.pi)
        
        # Step 4: combine these restrictions into a single domain, and
        # remove redundant or unused boundares.
        normals = Rotation(np.concatenate([large_cell_normals.data, fz_normals.data]))
        region = cls(normals)
        vertices = region.vertices()
        if vertices.size:
            region = region[np.any(np.isclose(region.dot_outer(vertices), 0), axis=1)]
        return region

    # --------------------- Other public methods --------------------- #

    def vertices(self) -> Rotation:
        """Return the vertices of the orientation region.

        Returns
        -------
        rot
            Orientation region vertices.
        """
        normal_combinations = list(itertools.combinations(self, 3))
        if len(normal_combinations) < 1:
            return Rotation.empty()
        c1, c2, c3 = zip(*normal_combinations)
        c1, c2, c3 = (
            Rotation.stack(c1).flatten(),
            Rotation.stack(c2).flatten(),
            Rotation.stack(c3).flatten(),
        )
        r = Rotation.triple_cross(c1, c2, c3)
        r = r[~np.any(np.isnan(r.data), axis=-1)]
        r = r[r < self].unique()
        surface = np.any(np.isclose(r.dot_outer(self), 0), axis=1)
        return r[surface]

    def faces(self) -> list[Rotation]:
        """Return the faces of the orientation region.

        Returns
        -------
        faces
            List of sets of rotations, each set describing a face of the
            region.
        """
        normals = Rotation(self)
        vertices = self.vertices()
        faces = []
        for n in normals:
            faces.append(vertices[np.isclose(vertices.dot(n), 0)])
        faces = [f for f in faces if f.size > 2]
        return faces

    def get_plot_data(self) -> Rotation:
        """Return suitable rotations for the construction of a wireframe
        delineating the borders of the region.

        Returns
        -------
        rot
            Rotations delineating the borders of the region.
        """
        from orix.vector import Vector3d

        # Get a grid of vector directions
        theta = np.linspace(0, 2 * np.pi - constants.eps9, 361)
        rho = np.linspace(0, np.pi - constants.eps9, 181)
        theta, rho = np.meshgrid(theta, rho)
        g = Vector3d.from_polar(rho, theta)

        # Get the cell vector normal norms
        if self.size == 0:
            return Rotation.from_axes_angles(g, np.pi)
        n = self.to_rodrigues().norm[:, np.newaxis, np.newaxis]

        d = (-self.axis).dot_outer(g.unit)
        x = n * d
        with np.errstate(divide="ignore"):
            omega = 2 * np.arctan(np.where(x != 0, x**-1, np.pi))

        # Keep the smallest allowed angle
        omega[omega < 0] = np.pi
        omega = np.min(omega, axis=0)
        r = Rotation.from_axes_angles(g.unit, omega)

        return r
