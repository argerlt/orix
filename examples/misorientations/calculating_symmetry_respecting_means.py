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
=======================================
Calculating the mean of misorientations
=======================================

This example demonstrates how a mean is calculated for both
misorientations and orientations in a symmetry-respecting manner.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import orix.quaternion as oqu
import orix.quaternion.symmetry as osm
from orix.plot.rotation_plot import _setup_rotation_plot

# Reproducible random data.
np.random.seed(2319)  

# convenience function for plotting.
def setup_plot(fz):
    fig,ax = _setup_rotation_plot(projection='homochoric')
    fig.set_figwidth(6)
    fig.set_figheight(5)
    ax.axis('off')
    ax._correct_aspect_ratio(fz)
    return (fig, ax)

#########################################################################################
# The common Euclidean definition of a mean does not work for systems with
# periodic boundaries such as rotations. Instead, the Frobenius norm should
# be used to define an equivalent concept. Details on ORIX's implementation
# can be found in the docstring for orix.quaternion.Rotation.mean(), but to
# summarize, the "mean" is defined as the rotation whose total angular
# distance from all data points is the smallest.
#
# However, including symmetry complicates the question since multiple
# quaternions can represent an equivalent operation, and it's not necessarily
# clear which representation should be used when calculating the mean.
# A common simplification is to only consider equivalent representations that
# fall within the misorientation's Fundamental Zone (fz).

random_m = oqu.Misorientation.random(10,symmetry=[osm.C1, osm.D2])
equiv_m = random_m.equivalent()
reduced_m = random_m.reduce()
# For clarity when plotting, remove reduced representations from equivalents
equiv_m = equiv_m[equiv_m.dot_outer(reduced_m).max(axis=1)<0.90]
# Define the fundamental zone for C1(1) --> D2(222)
fz_C1_D2 = oqu.OrientationRegion.from_symmetry(*random_m.symmetry)

fig1,ax1 = setup_plot(fz_C1_D2)
ax1.scatter(reduced_m, color='red', s = 50)
ax1.scatter(equiv_m, c='black', s=5)
ax1.plot_wireframe(fz_C1_D2, color='grey')
fig1.suptitle("Equivalent (black) vs Reduced (red) representations\nfor C1(1) --> D2(222) crystal system")
plt.tight_layout()

#########################################################################################
# This will produce a rough mean that is typically close to the lowest
# achievable mean, but not garunteed to be. To get a more accurate value,
# it is necessary to check every symmetrically equivalent permutation of each
# misorientation and find values that are closer to the estimated mean value,
# then recalculate the mean.

qu_data = np.stack([norm.rvs(i,0.09,400) for i in [0.3,0.1,0.2,0.3]]).T
clustered_m = oqu.Misorientation(qu_data, symmetry=(osm.D4,osm.C2v)).reduce()
fz_D4_D2h = oqu.OrientationRegion.from_symmetry(*clustered_m.symmetry)
rough_mean = oqu.Rotation(clustered_m).mean()
precise_mean = clustered_m.mean()

#########################################################################################
# The final complication comes from how to handle improper representations
# occurring due to inversions and rotoinversions. Since no pure rotation can
# align an improper reference frame witha proper reference frame, the
# concept of a "minimum total angular distance" does not apply.
# The two solutions are to either ignore improper elements, or treat them
# as proper elements. This is often a negligable difference, but for certain
# misorientation-specific systems, it can have a notable impact on the mean.


precise_proper_mean = clustered_m.mean(proper_mean=True)

reduced = oqu.Rotation(clustered_m.data)
nearest = oqu.Rotation(clustered_m.data)
nearest_proper = oqu.Rotation(clustered_m.data)
max_dp = np.zeros(reduced.shape, dtype=float)
max_dp_proper = np.zeros(reduced.shape, dtype=float)
start_sym_group, end_sym_group = clustered_m.symmetry
for start in start_sym_group:
    for end in end_sym_group:
        candidates = end * reduced * start
        dp = np.abs(candidates.dot(rough_mean))
        mask_all = dp>max_dp
        nearest.data[mask_all,:] = candidates.data[mask_all, :]
        max_dp[mask_all] = dp[mask_all]
        dp[candidates.improper]=0
        mask_proper = dp>max_dp_proper
        nearest_proper.data[mask_proper, :] = candidates.data[mask_proper, :]
        max_dp_proper[mask_proper] = dp[mask_proper]

all_changes = reduced.dot(nearest)<0.999
proper_changes = reduced.dot(nearest_proper)<0.999
improper_changes = reduced.dot(nearest)<0.999
both_changes = proper_changes*improper_changes

fig2,ax2 = setup_plot(fz_D4_D2h)
ax2.plot_wireframe(fz_D4_D2h, color='grey')

ax2.scatter(reduced[~all_changes], c='grey', s =4,alpha = 0.3)
ax2.scatter(reduced[all_changes], c='black', s =4,alpha = 0.3)

ax2.scatter(nearest[improper_changes*~both_changes], c='red',s=4, alpha = 0.3)
ax2.scatter(precise_mean,s=60,c='red')
for i in np.arange(clustered_m.size)[improper_changes*~both_changes]:
    ax2.plot(oqu.Misorientation.stack([reduced[i],nearest[i]]),color='red',alpha = 0.2,linewidth = 1)

ax2.scatter(nearest_proper[proper_changes], c='blue',s=4, alpha = 0.3)
ax2.scatter(precise_proper_mean,s=60,c='blue')
for i in np.arange(clustered_m.size)[proper_changes]:
    ax2.plot(oqu.Misorientation.stack([reduced[i],nearest_proper[i]]),color='blue',alpha = 0.2,linewidth = 1)

fig2.suptitle("Proper (blue) vs improper(red) nearest neighbors outside the FZ\nfor D4(422) --> C2v(mm2) crystal system")
plt.tight_layout()

#########################################################################################
# Note how in the example above, closer improper representations were found
# inside the fundamental zone. This is because Rodriguez and Machenzie
# fundamental zones are defined using ONLY proper symmetry elements. This is
# only an issue for non-centrosymmetric crystal systems that contain
# inversions.