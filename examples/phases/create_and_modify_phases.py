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
===============================================
Create and Modify Crystal Phases and PhaseLists
===============================================

This example shows various ways to create a crystal :class:`~orix.crystal_map.Phase`.

For alignment of the crystal axes with a Cartesian coordinate system, see the
example on :doc:`/examples/crystal_phase/crystal_reference_frame`.
"""

import diffpy.structure as dps
import orix.crystal_map as ocm

##############################################################################
# A Phase object, at minimum, contains the crystallographic symmetry of a phase
# and it's unit cell. for more information on how this class is different than
# :class:`~orix.quaternion.Symmetry`, see :doc:`/examples/phases/Phase_versus_Symmetry`
#
# Internally, orix uses diffpy.structure for all unit cell calculations. Diffpy
# works by defining a Lattice plus one or more Atoms within a Structure. For
# users interested more granular control, diffpy.structure can be used to define a
# phase as follows.

ferrite_structure = dps.Structure(
    title="ferrite",
    # a, b, c, alpha, beta, and gamma in nm and degrees for ferrite
    lattice=dps.Lattice(0.287, 0.287, 0.287, 90, 90, 90),
    atoms=[dps.Atom("Fe", [1e-5, 1e-5, 1e-5])],
)
ferrite_phase = ocm.Phase(
    space_group=229, structure=ferrite_structure, color="black"
).expand_asymmetric_unit()
print(ferrite_phase)
ferrite_phase.plot_unit_cell()

##############################################################################
# However, for users not interested in atomic positions, a Phase can also be
# set using the (a,b,c, alpha, beta, gamma) cell parameters without needing to
# import and define diffpy objects. Note that like diffpy, the (alpha, beta, gamma)
# angles are in degrees.

austenite_phase = ocm.Phase(
    name="Austenite",
    point_group="m3m",
    cell_parameters=[0.36, 0.36, 0.36, 90, 90, 90],
)
print(austenite_phase)
austenite_phase.plot_unit_cell()

##############################################################################
# Phases can also be imported directly from a Crystallographic Information File (CIF) file.
#
# E.g. one for titanium from an online repository like the Americam Mineralogist
# Crystal Structure Database:
# https://rruff.geo.arizona.edu/AMS/download.php?id=13417.cif&down=text
# phase_ti = Phase.from_cif("ti.cif")
# print(phase_ti)

##############################################################################
# Phases can also be defined from a space group. This is done using the space
# group number (see the following useful link for details: http://img.chem.ucl.ac.uk/sgp/large/sgp.htm)
phase_m3m = ocm.Phase(space_group=225)
print(phase_m3m)

##############################################################################
# Alternately, phases can be derived from a point group, in which case the
# space group will remain undefined as there  are multiple possible options.
phase_432 = ocm.Phase(name="unknown", point_group="432")
print(phase_432)

##############################################################################
# Phases can also be defined without a symmetry, though this will often cause
# errors
phase_non = ocm.Phase()
print(phase_non)

##############################################################################
# Creating a PhaseList
# --------------------
#
# Since CrystalMap objects can contain multiple phases, it is convenient to
# store sets of Phases as a in iteratable PhaseList object. This can be done
# by defining individual phases, or by defining the phases during creation
# of the list

phases_from_list = ocm.PhaseList([ferrite_phase, austenite_phase, phase_432])
print(phases_from_list)

new_phases = ocm.PhaseList(
    names=["Alpha", "Beta", "Gamma"],
    space_groups=[75, 229, 225],
    colors=["red", "orange", "yellow"],
)
print(new_phases)

##############################################################################
# These phases can then be referenced either by their index or their phase name
print(phases_from_list["Austenite"])
print(phases_from_list[0])
print(phases_from_list[:2])
print(phases_from_list["unknown", "ferrite"])

##############################################################################
# Modifying Phases and PhaseLists
# -------------------------------
#
# The following Phase attributes can all be modified after initialization:
#    - name
#    - space_group
#    - point_group
#    - structure
#    - color
#
# Note though that overwriting point_group when space_group has already been
# defined will throw a warning and the original space group will be erased.

ferrite_phase.point_group = "422"
print(ferrite_phase)

##############################################################################
# Similarly in the opposite direction, altering the space group will overwrite
# the point group. Keep in mind as well that space group attributes are stored
# as :class:`diffpy.structure.spacegroups.SpaceGroup` instances, whereas
# point groups are :class:`~orix/quaternion.symmetry.Symmetry` instances.
phase_m3m.space_group = 230
print(phase_m3m)

##############################################################################
# This can also be done with a PhaseList using indexing to choose the Phase to
# alter.
phases_from_list[1].space_group = 229
print(phases_from_list)

##############################################################################
# Phases can also be added to a PhaseList, either from another Phaselist or
# from a standalone and/or new Phase, and also deleted. there is also a
# convenience function for adding a `not indexed` phase, as this often becomes
# relevant in experimental crystal maps.
phases_from_list.add(new_phases[0])
phases_from_list.add(ocm.Phase("sigma", point_group="4/mmm"))
print(phases_from_list)
del phases_from_list["unknown"]
print(phases_from_list)
phases_from_list.add_not_indexed()
print(phases_from_list)


##############################################################################
# Shallow Copying Phases
# ----------------------
#
# Finally, note that PhaseLists generated from lists of Phases are shallow copies,
# meaning changes to the Phases will affect the PhaseLists. This can be seen above
# in the ferrite phase of `phases_from_list`, which originally had a space group
# of `Im-3m`, but in a later example has a space group of `None`, reflecting the
# change made to to `ferrite_phase`.
