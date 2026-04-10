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

import matplotlib.figure as mfigure
import numpy as np

from orix.plot.direction_color_keys.direction_color_key_tsl import (
    DirectionColorKeyTSL,
)
from orix.plot.orientation_color_keys.ipf_color_key import IPFColorKey
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import Symmetry
from orix.vector import Miller, Vector3d


class IPFColorKeyTSL(IPFColorKey):
    """Assigns colors to orientations in a symmetry-aware manner.

    IPF color keys work by defining symmetry-specific color maps
    that convert every orientation into a color. Images such as EBSD
    maps can then be colored based on the crystal vector aligned with
    a queried sample direction. For the simple case where the sample
    direction is aligned with the viewing direction, this is akin to
    asking "what color is the crystal vector pointing out of the screen
    at me."

    Parameters
    ----------
    symmetry
        Crystal symmetry used to define the colormap. If symmetry
        is not a Laue Symmetry, the equivalent Laue symmetry will
        be used instead.
    direction
        Sample direction. If not given, sample Z direction (out of
        plane) is used.

    Notes
    -----
    The TSL colormaps map only the 11 Laue classes, as they can all
    be described as weighted combinations of red, green, and blue. For
    details on this map as well as options for less limiting alternative
    mappings, refer to "Orientations, Perfectly Colored"
    https://doi.org/10.1107/S1600576716012942
    """

    def __init__(
        self,
        symmetry: Symmetry,
        direction: Vector3d | list | tuple | None = None,
    ) -> None:
        if direction is not None:
            if isinstance(direction, Miller):
                raise ValueError(
                    "The sample direction must be a sample vector, not a crystal vector"
                )
            if not isinstance(direction, Vector3d):
                try:
                    direction = Vector3d(np.asanyarray(direction))
                except:
                    raise ValueError("'direction' cannot be interpreted as a Vector3d")
            if direction.size != 1:
                raise ValueError("Only one sample direction can be given")
        super().__init__(symmetry.laue, direction=direction)

    @property
    def direction_color_key(self) -> DirectionColorKeyTSL:
        return DirectionColorKeyTSL(self.symmetry)

    def orientation2color(self, orientation: Orientation) -> np.ndarray:
        """Return an RGB color per orientation given a Laue symmetry
        and a sample direction.

        Plot the inverse pole figure color key with :meth:`plot`.

        Parameters
        ----------
        orientation
            Orientations to color.

        Returns
        -------
        rgb
            Color array of shape ``orientation.shape + (3,)``.
        """
        # TODO: Take crystal axes into account, by using Miller instead
        # of Vector3d
        m = orientation * self.direction
        rgb = self.direction_color_key.direction2color(m)
        return rgb

    def plot(self, return_figure: bool = False) -> mfigure.Figure | None:
        """Plot the inverse pole figure color key.

        Parameters
        ----------
        return_figure
            Whether to return the figure. Default is ``False``.

        Returns
        -------
        figure
            Color key figure, returned if ``return_figure=True``.
        """
        return self.direction_color_key.plot(return_figure=return_figure)
