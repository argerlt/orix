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
    start
        Initial point group, C1 for orientations.
    end
        Ending point group.

    Returns
    -------
    start
        Initial proper, inversion, or laue subgroup as appropriate.
    end
        Final proper, inversion, or laue subgroup as appropriate.

    Notes
    -----
    ORIX follows the asymmetric domain/fundamental zone definitions
    from in :cite:`krakow2017onthree`. However, for reasons given
    in section 3(b), that paper only defines domains for the 121
    combinations of proper point groups. Orix extends their logic for
    defining domains to the remaining 903 point group combinations.

    This is a trivial terminology issue for all orientations as well
    as any misorientations where both point groups are proper and/or
    centrosymmetric, which together account for 704 of the possible
    1024 symmetry cases. This includes data from EBSD due to the
    artificial centrosymmetry introduced in kikuchi diffraction.For
    the remaining 320 misorientations where one or both point groups
    improper and not centrosymmetric (for example, 6mm-->6mm)
    there is always one and occasionally 2 improper rotations that
    also map to the fundamental zone, as well as a possible
    pseudo-proper rotation only achievable by  combining two
    roto-inversions.(ex., 6mm). There is currently no concencus in
    literature on how to handle these rare edge cases.
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
    """A subset of rotation space.
    
    The complete set of all possible rigid body rotations is called
    SO(3). it can be thought of as half the quaternion unit sphere,
    the entirety of Rodrigues space, the set of all 3x3 matrices
    with a determinate of 1, or various other descriptors based
    on the application.

    Sometimes, this whole space is not need, for example if the
    orientation of an object is constrained or (most commonly) if
    the object is symmetrical. In this case, the space can be
    segmented using set of rotations representing boundaries in the
    space. This can be most easily visualized using Rodrigues
    space, where the boundaries become flat planes normal to the
    rodrigues vectors of those bounding rotations.
    
    .. image:: /_static/img/orientation-region-Oq.png
       :width: 300px
       :alt: Boundaries of an orientation region in Rodrigues space.
       :align: center

    Quaternions can then be quickly defined as inside or outside of
    these regions via a dot product operation.

    Notably, these regions are only defined in SO(3), which means
    they cannot account for improper operations. This is why
    OrientationRegion.from_symmetry() calculates identical regions
    for point groups 432 and m-3m despite m-3m having twice as many
    distinguished points. This ends up being irrelevant for
    Orientations since any improper operations that place a point
    within a fundamental zone always have a paired proper operation
    that returns an identical quaternion, but it can create confusion
    for misorientations with rotoinversions when users assume an
    OrientationRegion can uniquely define a true fundamental zone.
    """

    # ------------------------ Dunder methods ------------------------ #

    def __gt__(self, other: OrientationRegion) -> np.ndarray:
        """Overridden greater than method.

        Applying this to a rotation will return only those that lie
        within the region. This operation does not account for
        inversion.
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
        """ Return an orientation region for a given symmetry.

        These regions are identical to the fundamental zone for all
        orientations and every misorientation where both
        symmetries are proper and/or centrosymmetric. For
        all other cases, it is still garunteed to inlude only one
        unique represenation achievable though proper rotations.See
        Notes for details.

        Parameters
        ----------
        start
            Initial point group, C1 for passive orientations.
        end
            Ending point group.

        Returns
        -------
        region
            The orientation region.

        Notes
        -----
        ORIX follows the asymmetric domain/fundamental zone definitions
        from in :cite:`krakow2017onthree`. However, for reasons given
        in section 3(b), domains are only described for the 121
        combinations of proper point groups. Section 5.3.1 of
        :cite"martineau2020multivariate" gives pseudocode for
        extending this original logic to define domains for all 1024
        possible symmetry cases.

        This ends up being a trivial terminology issue for all
        orientations as well as any misorientations where both point
        groups are proper and/or centrosymmetric. This includes all
        EBSD data as well, since kikuchi diffraction introduces an
        artificial centrosymmetry. For these 704 cases, either the
        region returned bounds a fully unique zone, or it bounds
        all proper representations, and the improper representations
        have identical quaternion representations.
        
        For the remaining 320 misorientations where one or both
        point groups contain rotoinversions but are not
        centrosymmetric (for example, 6mm --> 6mm), there are always
        one and possibly two improper rotations that also map to the
        orientation region but with unique quaternion values, as well
        as a possible unique pseudo-proper rotation only achievable
        through two rotoinversions. There is currently no concensus
        on how to define unique fundamental zones for these edge
        cases.
        """
        # Step 1: fundamental zones are only defined for proper rotations.
        # add inversion centers where necessary to define as unique as
        # possible of a fundamental zone, then remove all improper operators.
        # if either symmetry is proper, any improper symmetries of the second
        # group will fall outside the shared fundamental zone.
        if not start.is_proper and not end.is_proper:
            # If both symmetries contain an inversion, all improper operators
            # will have a paired proper operator. If neither do, the proper
            # and improper rotations will form two identical but inverted
            # fundamental zones. Both cases produce one proper, two improper,
            # and one pseudo-proper fundamental zone, but in the first case 
            # they are aligned, and in the second case they are inverted.
            # The second case is the problematic form that requires
            # consideration when reducing or averaging misorientations.
            if start.contains_inversion != end.contains_inversion:
                # The remaining case is when only one of the two groups
                # contains an inversion. Here, it is necessary to add an
                # inversion to the non-centrosymmetric group to produce
                # a unique fundamental zone. 
                if not start.contains_inversion:
                    start = start.laue
                if not end.contains_inversion:
                    end = end.laue
        start = start.proper_subgroup
        end = end.proper_subgroup

        # Step 2: define the bounding cells using the distinguished points.
        dp = get_distinguished_points(start, end)
        large_cell_normals = _get_large_cell_normals(dp)

        # Step 3: (only for misorientations) restrict the domain to the
        # fundamental sector of the pole figure of the shared symmetries.
        disjoint = start & end
        # if a is True:
        #     disjoint = Symmetry.from_generators(disjoint,Ci)
        fz = disjoint.fundamental_zone()
        fz_normals = Rotation.from_axes_angles(fz, np.pi)
        # if a is True:
        #     fz_normals =Rotation(np.concatenate([fz_normals.data,np.array([[0,1,0,0],
        #                                                                    [0,0,1,0],
        #                                                                    [0,0,0,1]])]))
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
