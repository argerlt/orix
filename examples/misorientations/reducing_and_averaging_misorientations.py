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
=======================================================
Reducing and Averaging Misorientations and Orientations
=======================================================

This example introduces the concept of reducing an orientation or
misorientation with respect to symmetry, as well as the related
concept of averaging a misorientation.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from orix.plot.rotation_plot import _setup_rotation_plot
import orix.quaternion as oqu
import orix.quaternion.symmetry as osm

# Reproducible random data.
np.random.seed(2319)


# convenience function for plotting.
def setup_plot(fz):
    fig, ax = _setup_rotation_plot(projection="homochoric")
    fig.set_figwidth(6)
    fig.set_figheight(5)
    ax.axis("off")
    ax._correct_aspect_ratio(fz)
    return (fig, ax)


#########################################################################################
# Basic Example
# -------------
# `Orientation.reduce` and `Misorientation.reduce` will convert a transform
# with symmetry to a unique equivalent representation with the smallest
# possible angle of rotation. This is what is meant by "reducing" a
# transform.

g_random = oqu.Orientation.random(shape=1, symmetry=osm.Oh)
g_reduced = g_random.reduce()

g_fz = oqu.OrientationRegion.from_symmetry(osm.C1, osm.Oh)
g_all = osm.Oh.outer(g_random)
g_equiv = g_all[oqu.Rotation(g_all).dot(g_reduced) < 0.999]

fig_1, ax_1 = setup_plot(g_fz)
ax_1.set_xlim([-1, 1])
ax_1.set_aspect("equal")
ax_1.scatter(g_equiv, color="black")
ax_1.scatter(g_reduced, color="red")
ax_1.plot_wireframe(g_fz, color="grey")
fig_1.suptitle(
    "Reduced (red) and equivalent (black) \nrepresentations in point group Oh (m-3m)"
)

#########################################################################################
# The inclusion of symmetry combined with the periodic nature of rotations
# can make the definition of a mean or average ambiguous (more on this below),
# so the first step when an average is calculated is to reduce the transforms
# and then  calculate the average. A side effect of this is the mean
# (mis)orientation returned by ORIX will also be a reduced representation.

qu_data = np.stack([norm.rvs(i, 0.06, 20) for i in [0.39, 0.28, -0.39, 0.78]]).T
m_sym = [osm.C3, osm.D2]

m_clustered = oqu.Misorientation(qu_data, symmetry=m_sym)
m_reduced = m_clustered.reduce()
m_mean, m_neighbors = m_clustered.mean(return_neighbors=True)

fz_cluster = oqu.OrientationRegion.from_symmetry(*m_sym)

fig2, ax2 = setup_plot(fz_cluster)
ax2.scatter(oqu.Rotation(m_clustered), c="k")
ax2.scatter(oqu.Rotation(m_reduced), c="r")
ax2.scatter(oqu.Rotation(m_mean), color="blue", marker="X", s=100)
ax2.plot_wireframe(fz_cluster, color="grey")
fig2.suptitle(
    "Reduced (red), Original (black), and symmetry-aware \nMean (blue) for {C3(3)-> D2(222)} system"
)

#########################################################################################
# The Fundamental Zone
# --------------------
#
# In the plots above, wireframes were included that defined bounded volumes
# within which all reduced representations fell. This is known as a
# Fundamental Zone (FZ), and contains a single unique (aka, fundamental)
# representation of every transformation with respect to a given symmetry.
# representations that fall within a fundamental zone are also garunteed to
# have the smallest possible angular component.
#
# There are several ways in which a fundamental zone can be defined, with most
# discrepencies stemming from how improper transforms
# (ie, inversions and rotoinversions) should be handled. ORIX uses the rules
# presented in :cite:`Krakow krakow2017onthree`, but expanded to all 1024
# misorientation groups. This can be verified by comparing the following plots
# to Figure 5 of the same paper.

name2group = {x.name: x for x in osm._groups}

fig5 = plt.figure(figsize=[5, 8])

base_pairs = [
    ["432", "3"],
    ["23", "3"],
    ["432", "1"],
    ["23", "1"],
    ["3", "422"],
    ["622", "1"],
    ["4", "211"],
    ["32", "1"],
    ["222", "1"],
    ["3", "4"],
    ["6", "1"],
    ["4", "1"],
    ["3", "1"],
    ["211", "1"],
    ["1", "1"],
]
for i, pair in enumerate(base_pairs):
    s1 = name2group[pair[0]]
    s2 = name2group[pair[1]]
    fz = oqu.OrientationRegion.from_symmetry(s1, s2)
    ax = fig5.add_subplot(5, 3, i + 1, projection="homochoric")
    ax.axis("off")
    ax._correct_aspect_ratio(fz)
    ax.plot_wireframe(fz)
    ax.set_title("{}: {} -> {}".format("abcdefghijklmno"[i], s1.name, s2.name))

plt.tight_layout()

#########################################################################################
# If users are unfamiliar with how Rodrigues, Homochoric, or NeoEulerian plots
# are used to plot rotations in 3D space, the same paper also contains a
# concise overview.
#
# It is not ran as part of this example since calculating and plotting 66
# fundamental zones is time consuming, but the following plot matches Table
# 3 of the same paper, showing the subdivisions of the above 15 zones caused
# by shared rotation elements between the point groups. Users may plot it
# themselves by running this code locally with `plot_me=True`


plot_me = False

names = ["432", "23", "622", "6", "32", "3", "422", "4", "222", "211", "1"]
if plot_me is True:
    table_fig = plt.figure()
    for i in range(len(names)):
        n1 = names[i]
        s1 = name2group[n1]
        for j in range(len(names) - i):
            n2 = names[-j - 1]
            s2 = name2group[n2]
            fz = oqu.OrientationRegion.from_symmetry(s1, s2)
            ij = j * 11 + i + 1
            ax = table_fig.add_subplot(11, 11, ij, projection="homochoric")
            ax.axis("off")
            ax._correct_aspect_ratio(fz)
            ax.plot_wireframe(fz)

#########################################################################################
# Defining a Mean in Rotation Space
# ---------------------------------
#
# Up until now, we have not defined what is meant by the mean of a group of
# transforms. Because rotation space is periodic, the concept of a Euclidean
# norm does not apply, and instead a Frobenius norm is used, which in this
# context can be thought of as the magnitude of the angular rotation necessary
# to align two transforms. Noteably, this is NOT simply the normalized average
# of two transform's quaternion representations, as is sometimes done in other
# software to get a fast approximation for clustered transforms. See the
# docstring for `Quaternion.mean` for details on this topic.
#
# The extension of this is a mean transform is defined as the transform whose
# total angular deviation from all transforms in the queried group is the
# minimum possible value.
#
# However, there is not a convenient algorithm to calculate this correctly,
# so instead ORIX does the following:
#
#     1) transforms are reduced to the appropriate fundamental zone.
#     2) A rough mean is calculated.
#     3) transforms with equivalents closer to the rough mean are updated to the closer value
#     4) A precise mean is recalculated.
#
# The plot below is provided to help visualize this process.

np.random.seed(2319)
qu_data = np.stack([norm.rvs(i, 0.1, 20) for i in [0.1, 0.1, 0.2, 0.3]]).T

o_cluster = oqu.Orientation(qu_data, symmetry=osm.D2)
o_reduced = o_cluster.reduce()
o_mean, o_neighbors = o_cluster.mean(return_neighbors=True)
o_flipped = o_neighbors[np.abs(o_neighbors.angle - o_reduced.angle) > 1e-3]


fig_ave, ax_ave = setup_plot(fz)
ax_ave.scatter(oqu.Rotation(o_cluster), color="black")
ax_ave.scatter(oqu.Rotation(o_reduced), color="red")
ax_ave.scatter(oqu.Rotation(o_flipped), color="green")
ax_ave.scatter(oqu.Rotation(o_mean), color="blue", marker="X", s=100)

fz = oqu.OrientationRegion.from_symmetry(end=osm.D2)
ax_ave.plot_wireframe(fz, color="grey")
ax_ave.set_xlim([-1, 1])
ax_ave.set_ylim([-1, 1])
ax_ave.set_zlim([-1, 1])
fig_ave.suptitle(
    "Original (black), Reduced (red), and flipped(green) \nrepresentations for point group D2 (222)"
)
