r"""
========================
Plot symmetry operations
========================

This example is one method for producing stereographic projections with
symmetry operators for the 32 crystallographic point groups. This method
loosely follows the one laid out in section 9.2 of "Structures of Materials"
(DeGraef et.al, 2nd edition, 2012).

There are generated proceedurally, and vary slightly from more curated
approaches. Consider, for example, the m$\bar{3}$m, plot in the image below,
which displays the inversion center as part of every marker (this is accurate
but redundant). Compare this to Figure 9.11 of "Structure of Materials" where
the inversion is only shown in the 4-fold z-axis and 3-fold <111>, or the
International Tables of Crystallography SG221, which displays all reduntant
inversion information plus additional information about the directionality of
the rotational symmetry.
"""

import matplotlib.pyplot as plt
import numpy as np
from orix import plot
from orix.quaternion.symmetry import *
from orix.vector import Vector3d

point_groups = [
    C1,
    C2,
    C3,
    C4,
    C6,
    D2,
    D3,
    D4,
    D6,
    Ci,
    Cs,
    S6,
    S4,
    C3h,
    C2h,
    C4h,
    C6h,
    C2v,
    C3v,
    C4v,
    C6v,
    D3d,
    D3h,
    D2h,
    D2d,
    D4h,
    D6h,
    T,
    O,
    Th,
    Oh,
    Td,
]


# Set marker sizes and colors to help differentiate elements
s = 120
colors = {1: "magenta", 2: "green", 3: "red", 4: "purple", 6: "olive"}
mirror_linewidth = 1
mirror_color = "blue"

# Create the plot and subplots using ORIX's stereographic projection
fig, ax = plt.subplots(
    4, 8, subplot_kw={"projection": "stereographic"}, figsize=[14, 10]
)
ax = ax.flatten()

# Iterate through the 32 Point groups
for i, pg in enumerate(point_groups):
    ax[i].set_title(pg.name)
    # get unique axis families (should just be <100>, <110>, and/or <111>)
    unique_axes = Vector3d(
        np.unique(np.around(pg.axis.in_fundamental_sector(pg).data, 5), axis=0)
    )
    # create some makes to find the proper rotations, mirror planes,
    # rotations, and rotoinversions more quickly.
    p_mask = ~pg.improper
    m_mask = (np.abs(pg.angle - np.pi) < 1e-4) * pg.improper
    r_mask = pg.angle**2 > 1e-4
    roto_mask = r_mask * ~p_mask * ~m_mask
    i_mask = (~r_mask) * pg.improper
    # to avoid repetition, look at only the unique fundamental representations
    # of the possible rotation axes
    fs_axes = pg.axis.in_fundamental_sector(pg)
    decorated_axes = []

    # iterate through each primary axis, plotting their symmetry elements
    # as we go.
    for axis in unique_axes:
        axis_mask = np.sum(np.abs((fs_axes - axis).data), 1) < 1e-4
        # plot any mirror planes perpendicular to the axis first
        if np.any(m_mask * axis_mask):
            for v in pg * axis:
                ax[i].plot(
                    v.get_circle(),
                    color=mirror_color,
                    linewidth=mirror_linewidth,
                )
        # if all rotations are proper rotations, plot the appropriate symbol
        if np.all(p_mask[axis_mask]):
            # if the only element is identity, move on.
            if not np.any(r_mask * axis_mask):
                continue
            min_ang = np.abs(pg[r_mask * axis_mask].angle).min()
            f = np.around(2 * np.pi / min_ang).astype(int)
            c = colors[f]
            ax[i].symmetry_marker((pg * axis), fold=f, s=s, color=c)
            decorated_axes.append(axis * 1)
        # if there is an inversion center, plot the appropriate symbol
        elif np.any(i_mask):
            # this might just be the 1-fold inversion center
            if not np.any(r_mask * p_mask):
                f = 1
            else:
                min_ang = np.abs(pg[r_mask * axis_mask].angle).min()
                f = np.around(2 * np.pi / min_ang).astype(int)
            c = colors[f]
            ax[i].symmetry_marker(
                (pg * axis), fold=f, s=s, color=c, inner="dot"
            )
            decorated_axes.append(axis * 1)
        # the other option (besides empty) is a rotoinversion
        elif np.any(roto_mask[axis_mask]):
            min_ang = np.abs(pg[roto_mask * axis_mask].angle).min()
            f = np.around(2 * np.pi / min_ang).astype(int)
            c = colors[f]
            ax[i].symmetry_marker(
                (pg * axis), fold=f, s=s, color=c, inner="half"
            )
            decorated_axes.append(axis * 1)
    # Three-fold rotations around the 111 create a special subset of mirror
    # planes, which for ease we will add in by hand.
    if np.any(unique_axes.dot(Vector3d([1, 1, 1])) > 1.73):
        m_vectors = Vector3d([[0, 0, 1], [1, 1, 1]])
        for v in (pg.outer(m_vectors)).flatten().unique():
            ax[i].plot(v.get_circle(), color="blue", linewidth=1)

    # Finally, the combination of inversion center and mirror planes
    # creates 2-fold symmetries not on the primary axes. Let's add in any
    # that didn't already get included from other operations
    if np.sum(m_mask) > 1 and np.sum(i_mask) > 0:
        dax = Vector3d([x.data for x in decorated_axes])
        dax_unique = dax.flatten().unique()
        two_folds = pg.axis[m_mask].in_fundamental_sector(pg)
        mask = np.abs(two_folds.dot_outer(dax_unique)).max(axis=1) < 0.99
        new_two_folds = two_folds[mask]
        symm_two_folds = pg.outer(new_two_folds).flatten().unique()
        ax[i].symmetry_marker(symm_two_folds, fold=2, s=s, color="g")
