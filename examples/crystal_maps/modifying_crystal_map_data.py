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
==========================
Modifying Crystal Map Data
==========================

This example shows how to both modify and extract data contained in a
:class:`~orix.crystal_map.CrystalMap`, obtain

For further information on slicing data and advanced methods for
extracing phases or subsections into new CrystalMpas, refer to
:ref:`examples/crystal_maps/modifying_crystal_maps`

# For further information on plotting, refer to
:ref:`examples/crystal_maps/plotting_crystal_maps`
"""

import matplotlib.pyplot as plt
import numpy as np

import orix.crystal_map as ocm
import orix.data as oda
import orix.plot as opl

opl.register_projections()  # Register our custom Matplotlib projections

##############################################################################
# For this example we will use one of ORIX's example datasets, an EBSD scan of
# a two-phase super-duplex stainless steel (SDSS). Details on both this dataset
# and how to import other data can be found in
# :ref:`examples/crystal_maps/creating_crystal_maps.py`.

xmap = oda.sdss_ferrite_austenite(allow_download=True)
xmap
xmap.plot(overlay="dp")  # Dot product values added to the alpha (RGBA) channel

##############################################################################
# Maps can be semgented out by ...

# TODO: add the parts from crystal_map related to properties, rotations versus orientations,
# how to pull out phases and see the phases in the data, etc.
