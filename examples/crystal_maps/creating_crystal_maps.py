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
==========================================
Creating, Loading, and Saving Crystal Maps
==========================================

This example shows how to initialize a Crystal Map, either from an
existing file or from scratch, as well as how to save it.

Crystal Maps are created and handled using :class:`~orix.crystal_map.CrystalMap`.
Additionally, many commonly used data formats (.ang, Bruker h5, ctf, etc) can be
loaded directly using :func:`~orix.io.load". Finally, ORIX includes functions to
download a small number of default example datasets in :mod:`~orix.data`.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import orix.data as oda
import orix.io as oio
import orix.plot as opl
import orix.crystal_map as ocm


# diffpy imports used for defining custom Phases.
from diffpy.structure import Atom, Lattice, Structure


# %%
# Loading ORIX datasets
# ---------------------
#
# Lets first look at one of orix's example datasets. These are external files
# downloaded and cached via the python Pooch package, and a full list can be
# inspected using `dir(oda)`. For this example, we will use `sdss_ferrite_austenite`,
# an EBSD scan of a two-phase super-duplex stainless steel (SDSS) provided courtesy
# of Prof. Jarle Hjelen from the Norwegian University of Science and Technology,
# and carrying a CC BY 4.0 license.

print("Available CrystalMaps:\n - {}".format("\n - ".join(dir(oda))))
xmap = oda.sdss_ferrite_austenite(allow_download=True)
xmap
xmap.plot(overlay="dp")  # Dot product values added to the alpha (RGBA) channel
xmap

###############################################################################
# From this output, we can see this is a two-phase (48.4% austenite, 51.6% ferrite)
# CrystalMap, with both phases belonging to the cubic point group 432.
#
# Loading from files
# ---------------------
#
# CrystalMaps can also be created from dataset files. A list of supported formats
# can be inspected using :func:`~orix.io.load". Here we will use the `ang` file
# downloaded in the previous example.

cache_folder = oda._data._fetcher.path
filename = cache_folder / "sdss/sdss_ferrite_austenite.ang"
loaded_xmap = oio.load(filename)
loaded_xmap
###############################################################################
# Note that sometimes the loader might make decisions about how to name or
# color a phase that you wish to change. This can be easily fixed as desired.
loaded_xmap.phases["austenite/austenite"].name = "Austenite"
loaded_xmap.phases["ferrite/ferrite"].name = "Alpha Phase"
loaded_xmap.phases["Austenite"].color = "red"
loaded_xmap.phases["Alpha Phase"].color = "grey"
loaded_xmap.plot(
    overlay="iq"
)  # image quality added to the alpha (RGBA) channel

# %%
# Creating from scratch
# ---------------------
#
# CrystalMaps can also be created from scratch.
# First, define the relevant phases as a PhaseList. Advanced usage is discussed
# in :ref:`crystal_phase/create_crystal_phase.py`, but the following is a basic
# example.

austenite_structure = Structure(
    title="austenite",
    # a, b, c, alpha, beta, and gamma in nm and degrees for austenite
    lattice=Lattice(0.360, 0.360, 0.360, 90, 90, 90),
)
ferrite_structure = Structure(
    title="ferrite",
    # a, b, c, alpha, beta, and gamma in nm and degrees for ferrite
    lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90),
)

phases = ocm.PhaseList(
    point_groups=["432", "432"],
    structures=[austenite_structure, ferrite_structure],
)

###############################################################################
# Defining the CrystalMap then requires the following per-pixel data as
# 1-dimensional flattened vectors
# - The rotations as oqu.Rotation objects
# - The Phase IDs as a numpy integer array
# - the x and y coordinates
# Additional data can also be included if desired, including a per-pixel properties
# dictonary and scan units.
# Again, we will use data from the examples above for convenience, but this
# data could also be generated from a simulation, random noise, or any other
# source.

rots = loaded_xmap.rotations
p_id = loaded_xmap.phase_id
x, y = loaded_xmap.x, loaded_xmap.y

# optional additions
properties_dict = loaded_xmap.prop

created_xmap = ocm.CrystalMap(
    rotations=rots,
    phase_list=phases,
    phase_id=p_id,
    x=x,
    y=y,
    prop=properties_dict,
    scan_unit="um",
)

created_xmap

# %%
# Saving CrystalMaps
# ---------------------
#
# Finally, CrystalMaps can be saved using :func:`~orix.io.save`.
# Once again, the supported formats can be listed, but here the data will be
# saved using both the ORIX's HDF5 format and the ASCII-readbale ang format

oio.save()
oio.save(
    filename="new_file.ang", object2write=xmap, confidence_index_prop="dp"
)
oio.save(filename="new_file.h5", object2write=xmap)

ang_size = int(os.path.getsize("new_file.ang") / (1024))
h5_size = int(os.path.getsize("new_file.h5") / (1024))

print(f"ang file is {ang_size} Kb.")
print(f"orix h5 file is {h5_size} Kb.")

# cleanup
os.remove("new_file.ang")
os.remove("new_file.h5")

###############################################################################
# Note that the ORIX file format is more space efficient, and also faster to
# load and save, whereas the .ang file has the advantage of being human
# readable. Both have their uses, and can be switched between as needed.
#
# For ang files, also note that points not in data are set to `not_indexed` when
# reloaded from the .ang file, and all properties in points not in the data set.
# the exception is the CI column, where out-of-data points are instead set to -1,
# as this is the expected convetnion in MTEX and EDAX TSL for EBSD data.

# Finally, it is worth mentioning that if a map has more than one rotation and
# phase ID per point, the index parameter can be passed to write any "layer" of
# the data to file.
