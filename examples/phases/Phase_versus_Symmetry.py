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
Phase versus Symmetry
=====================
"""

import matplotlib.pyplt as plt
import numpy as np

import orix.crystal_map as ocm
import orix.quaternion as oqu
import orix.plot as opl

opl.register_projections()

# %%
# Symmetry versus Phase
# ---------------------
#
# ORIX includes two different but related classes for describing crystallographic
# information, :class:`~orix.quaternion.symmetry.Symmetry`, and
# :class:`~orix.crystal_map.Phase`.
#
# A Symmetry object contains ONLY information on symmetrically equivalent transforms,
# and in most cases is a Laue and/or Point group. For example, the Symmetry of
# ruby would be defined as:

ruby_sym = oqu.symmetry.D3d  # <-- Schoenflies notation for point group '-3m'
ruby_sym.plot()
ruby_sym

##############################################################################
# On the other hand, a Phase object contains at minimum both the symmetry and
# the unit cell.

ruby_phase = ocm.Phase(
    point_group=ruby_sym,
    cell_parameters=[0.476, 0.476, 1.298, 90, 90, 120],
)

ax = plt.figure().add_subplot(projection="3d")
ax.set_aspect("equal")
ax.set_proj_type = "ortho"
v = np.array([[x, y, z] for x in [1, 0] for y in [1, 0] for z in [0, 1]]).dot(
    ruby_phase._diffpy_lattice
)
e = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    [0, 4],
    [4, 5],
    [5, 1],
    [5, 7],
    [7, 6],
    [6, 4],
    [6, 2],
    [3, 7],
]
ax.plot([0, 0], [0, 0], [0, 1], "k")
ax.plot([0, 0], [0, 1], [0, 0], "k")
ax.plot([0, 1], [0, 0], [0, 0], "k")
ax.text(1, 0, 0, "X")
ax.text(0, 1, 0, "Y")
ax.text(0, 0, 1, "Z")
ax.text(*v[2], "<100>")
ax.text(*v[4], "<010>")
ax.text(*v[7], "<001>")
ax.scatter(*v.T, color="r")
for edge in e:
    ax.plot(*np.stack([v[edge[0]], v[edge[1]]]).T, color="red")
ax.set_title("Unit cell for ruby ")
plt.tight_layout()

##############################################################################
# This distinction is important because quaternion-based transforms only
# require defining a symmtry, whereas any calculations involving Miller indices
# or diffraction require defining a Phase.

# TODO: flesh out with basic miller example, plus miller mistakes from lazy defintions
