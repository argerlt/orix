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

from diffpy.structure import Atom, Lattice, Structure

import orix.crystal_map as ocm

##############################################################################
# A Phase object, at minimum, contains the crystallographic symmetry of a phase
# and it's unit cell. for more information on how this class is different than
# :class:`~orix.quaternion.Symmetry`, see :doc:`/examples/phases/phase_versus_symmetry`
#
# Internally, orix uses diffpy.structure for all unit cell calculations. Diffpy
# works by defining a Lattice plus one or more Atoms within a Structure. For
# users interested more granular control, diffpy.structure can be used to directly
# define a phase:

ferrite_structure = Structure(
    title="ferrite",
    # a, b, c, alpha, beta, and gamma in nm and degrees for ferrite
    lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90),
    atoms=[Atom("Fe", [0, 0, 0])],
)
ferrite_phase = ocm.Phase(point_group="m3m", structure=ferrite_structure)
ferrite_phase
##############################################################################
# However, for users not interested in atomic positions, a Phase can also be
# set using the (a,b,c, alpha, beta, gamma) cell parameters without needing to
# import and define diffpy objects. The (a,b,c) cell dimensions are given in nanometers,
# and the (alpha, beta, gamma) angles in degrees.

austenite_phase = ocm.Phase(
    name="Austenite",
    point_group="m3m",
    cell_parameters=[0.36, 0.36, 0.36, 90, 90, 90],
)
austenite_phase

########################################################################################
# Phases can also be imported directly from a Crystallographic Information File (CIF) file.
#
# E.g. one for titanium from an online repository like the Americam Mineralogist
# Crystal Structure Database:
# https://rruff.geo.arizona.edu/AMS/download.php?id=13417.cif&down=text
# phase_ti = Phase.from_cif("ti.cif")
# print(phase_ti)

########################################################################################
# From a space group (note that the point group is derived)
phase_m3m = ocm.Phase(space_group=225)
print(phase_m3m)

########################################################################################
# From a point group (note that the space group is unknown since there are multiple
# options)
phase_432 = ocm.Phase(point_group="432")
print(phase_432)

########################################################################################
# Non-crystalline phase
phase_non = ocm.Phase()
print(phase_non)

########################################################################################
# Hexagonal alpha-titanium with a lattice and atoms
structure_ti = Structure(
    lattice=Lattice(4.5674, 4.5674, 2.8262, 90, 90, 120),
    atoms=[Atom("Ti", [0, 0, 0]), Atom("Ti", [1 / 3, 2 / 3, 1 / 2])],
)
print(structure_ti)

########################################################################################
phase_ti = ocm.Phase(space_group=191, structure=structure_ti)
print(phase_ti)
print(phase_ti.structure)
