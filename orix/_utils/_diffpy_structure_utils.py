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

"""Utilities for interfacing with diffpy.structure."""

from typing import Callable

from diffpy.structure import __version__
from diffpy.structure.spacegroups import SpaceGroup
from packaging.version import Version

DIFFPY_STRUCTURE_VERSION = Version(__version__)

# TODO: Remove these checks (and most likely this whole file) once
# 3.4.0 is the minimal supported version
if Version(__version__) >= Version("3.4.0"):
    from diffpy.structure.spacegroups import get_space_group
else:
    from diffpy.structure.spacegroups import GetSpaceGroup as get_space_group

get_space_group: Callable[[str | int], SpaceGroup]

__all__ = [
    "get_space_group",
]
