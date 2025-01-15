# -*- coding: utf-8 -*-
#
# writemodfiles.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
writes mod files for all channels defined in ``neat.channels.channelcollection``
"""

import os

from . import channelcollection

print("--> Writing mod files")
for name, channel_class in list(channelcollection.__dict__.items()):
    if isinstance(channel_class, type) and name != "IonChannel" and name != "_func":
        chan = channel_class()
        chan.write_mod_file(
            os.path.join(os.path.dirname(__file__), "../simulations/neuron/mech")
        )
        del chan
