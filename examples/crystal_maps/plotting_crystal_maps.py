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

r"""
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


###############################################################################
# For more complex plotting, it can be more useful to instead define figures and
# axes using Matplotlib in conjunction with ORIX's "plot_map" projection. This
# allow adding additional elements to plots as well as making subfigures.

fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection="plot_map"))
ax[0].plot_map(xmap)
ax[1].plot_map(xmap, xmap.prop["iq"])
ax[1].plot([25, 25, 77, 75, 25], [25, 75, 75, 25, 25], color="r")
plt.tight_layout()

# TODO: add the rest verbatim from example, it's all getting changed this
# year anyway.

###############################################################################
# Orientation-Based Coloring
# --------------------------
#
# people often want to plot crystalmaps with orientation-based coloring. A few
# things to note about this:
#    - Coloring are defined per-Phase.
#    - Direction-based (ie, IPF) colorings include a loss of orientation information
#      perpendicular to the chosen direction.
#
# Here we will use the same IPF colormap used by EDAX. For details on this colormap
# refer to :ref:`examples/inverse_pole_figures`, but to summarize, It rotates the
# 'direction' vector by each orientation, then returns the IPF coloring of the
# rotated vector.
ckey_m3m = opl.IPFColorKeyTSL(
    xmap.phases["austenite"].point_group, direction=ove.Vector3d.zvector()
)
rgb_au = ckey_m3m.orientation2color(xmap["austenite"].orientations)
xmap["austenite"].plot(rgb_au)
