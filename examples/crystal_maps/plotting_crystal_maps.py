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

"""
=====================
Plotting Crystal Maps
=====================

This example gives a basic overview on how to plot CrystalMap objects
in ORIX.

The Orientation data and other properties used in this example were
aquired from a super-duplex stainless steel (SDSS) EBSD dataset provided
courtesy of Prof. Jarle Hjelen from the Norwegian University of Science
and Technology, and carries a CC BY 4.0 License.
"""

import matplotlib.pyplot as plt
import numpy as np

import orix.crystal_map as ocm
import orix.plot as opl
import orix.data as oda
import orix.quaternion as oqu
import orix.vector as ove

opl.register_projections()  # Register our custom Matplotlib projections

###############################################################################
# This example will use one of the default ORIX datasets from the data module.
# For details on how to load in your own datasets or create one from scrach,
# refer to the :ref:`crystal_maps/initializing_crystal_maps.py` example.

xmap = oda.sdss_ferrite_austenite(allow_download=True)
xmap

###############################################################################
# Simple plots can be made using :meth:`~orix.crystal_map.CrystalMap.plot` function.
# By default, plots will give only display the phases.
xmap.plot()

###############################################################################
# If there is either an external array or CrystalMap property representing a
# per-pixel scalar, this can be used to modify the plots as well with the
# `overlay` functionality.

xmap.plot(overlay="iq")  # Use the `Image Quality` saved in xmap.prop.
# Relative distance from the center
dist = (((xmap.x - 87) ** 2) + ((xmap.y - 74) ** 2)) ** 0.5
xmap.plot(overlay=-dist)


def plot_id(
    xmaps: ocm.CrystalMap | list[ocm.CrystalMap], titles: str | list[str]
) -> None:
    """Convenience function to plot at most four crystal maps showing
    rows, columns and IDs of each map point.
    """
    if isinstance(xmaps, ocm.CrystalMap):
        xmaps = [xmaps]
        titles = [titles]
    n_xmaps = len(xmaps)
    if n_xmaps > 2:
        fig_rows, fig_cols = 2, 2
    else:
        fig_rows, fig_cols = 1, len(xmaps)

    fig = plt.figure()
    for i in range(n_xmaps):
        ax = fig.add_subplot(fig_rows, fig_cols, i + 1, projection="plot_map")
        ax.plot_map(xmaps[i], "id", scalebar=False)
        rows, cols = xmaps[i].row, xmaps[i].col
        ax.set_xticks(np.arange(np.max(cols) + 1))
        ax.set_xticklabels(np.arange(np.max(cols) + 1))
        ax.set_yticks(np.arange(np.max(rows) + 1))
        ax.set_yticklabels(np.arange(np.max(rows) + 1))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(titles[i])
        for j, r, c in zip(xmaps[i].id, rows, cols):
            ax.text(c, r, j, va="center", ha="center", c="r")
    fig.tight_layout()


########################################################################################
# We start by creating a crystal map with five rows and ten columns with all points
# having one phase and an identity rotation, and plot the row and column coordinates as
# well as the map ID of each point into the originally created map

xmap = ocm.CrystalMap.empty(shape=(5, 10))
xmap.phases[0].name = "a"
print(xmap)

plot_id(xmap, "Initial map")

########################################################################################
# Slice the map (1) by selecting some rows and columns. We'll plot the IDs again and see
# that these do not update after slicing. We'll also select some values from the sliced
# map (2) by passing one or more indices

xmap2 = xmap[1:4, 5:9]  # First number inclusive, last number exclusive
plot_id(xmap2, "Map slice")

print(xmap2[0, 0].id)
print(xmap2[-1].id)  # Last row
print(xmap2[:, 1].id)

########################################################################################
# Select data based on phase(s) (3) after adding a new phase to the phase list and
# giving some points in the data the new phase ID by modifying the phase IDs inplace

xmap.phases.add(ocm.Phase("b"))

xmap[1, 1].phase_id = 1
xmap[1:4, 5:9].phase_id = 1
print(xmap)

plot_id([xmap["a"], xmap["b"], xmap["a", "b"]], ["a", "b", "a and b"])

########################################################################################
# Set some points to not indexed and select only the indexed data or the not indexed
# data (4)

xmap[3:, 1:4].phase_id = -1
print(xmap)

plot_id([xmap["indexed"], xmap["not_indexed"]], ["Indexed", "Not indexed"])

########################################################################################
# Select data satisfying one or more criteria using boolean arrays (5)

plot_id(
    [
        xmap[xmap.id > 10],
        xmap[(xmap.phase_id == 0) & np.mod(xmap.id, 2).astype(bool)],
        xmap[(xmap.phase_id == 1) | ~xmap.is_indexed],
    ],
    ["Id greater than 10", "a and odd ID", "b or not indexed"],
)

########################################################################################
# When obtaining a new map from part of another map, the new map is a shallow copy of
# the initial map. This means that changes to the new map also changes the initial map.
# When this is undesirable, we make a deep copy by calling
# :meth:`~orix.crystal_map.CrystalMap.deepcopy`.

xmap3_shallow = xmap["b"]
xmap3_deep = xmap["b"].deepcopy()

xmap3_shallow[1, 5].phase_id = -1
xmap3_deep[1, 6].phase_id = -1

plot_id(
    [xmap["indexed"], xmap3_shallow["indexed"], xmap3_deep["indexed"]],
    ["Initial, indexed", "b shallow copy, indexed", "b deep copy, indexed"],
)
