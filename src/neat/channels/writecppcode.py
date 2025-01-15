# -*- coding: utf-8 -*-
#
# writecppcode.py
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
Writes C++ simulation code for the ionchannels
"""

import os

# from neat.channels import channelcollection
from .channelcollection import channelcollection


def write():

    print("--> Writing c++ channel file")
    path = os.path.join(os.path.dirname(__file__), "../simulations/net/")
    # path = '../simulations/net/'

    fcc = open(os.path.join(path, "Ionchannels.cc"), "w")
    fh = open(os.path.join(path, "Ionchannels.h"), "w")
    fh.write("#include <iostream>" + "\n")
    fh.write("#include <string>" + "\n")
    fh.write("#include <vector>" + "\n")
    fh.write("#include <list>" + "\n")
    fh.write("#include <map>" + "\n")
    fh.write("#include <complex>" + "\n")
    fh.write("#include <string.h>" + "\n")
    fh.write("#include <stdlib.h>" + "\n")
    fh.write("#include <algorithm>" + "\n")
    fh.write("#include <math.h>" + "\n")
    fh.write("#include <time.h>" + "\n")
    fh.write("#include <time.h>" + "\n")
    fh.write("using namespace std;" + "\n\n")

    fh.write("class IonChannel{" + "\n")
    fh.write("protected:" + "\n")
    fh.write("    double m_g_bar = 0.0, m_e_rev = 50.00000000;" + "\n")
    fh.write("    bool m_instantaneous = false" + ";\n")
    fh.write("public:" + "\n")
    fh.write(
        "    void init(double g_bar, double e_rev){m_g_bar = g_bar; m_e_rev = e_rev;};"
        + "\n"
    )
    fh.write("    void setInstantaneous(bool b){m_instantaneous = b;};" + "\n")
    fh.write("    virtual void calcFunStatevar(double v){};" + "\n")
    fh.write("    virtual double calcPOpen(){return 0.0;};" + "\n")
    fh.write("    virtual void setPOpen(){};" + "\n")
    fh.write("    virtual void setPOpenEQ(double v){};" + "\n")
    fh.write("    virtual void advance(double dt){};" + "\n")
    fh.write("    virtual double getCond(){return 0.0;};" + "\n")
    fh.write("    virtual double getCondNewton(){return 0.0;};" + "\n")
    fh.write("    virtual double f(double v){return 0.0;};" + "\n")
    fh.write("    virtual double DfDv(double v){return 0.0;};" + "\n")
    fh.write("    virtual void setfNewtonConstant(double* vs, int v_size){};" + "\n")
    fh.write("    virtual double fNewton(double v){return 0.0;};" + "\n")
    fh.write("    virtual double DfDvNewton(double v){return 0.0;};" + "\n")
    fh.write("};" + "\n")

    fcc.write('#include "Ionchannels.h"' + "\n")
    fh.write("\n")
    fcc.write("\n")
    fcc.close()
    fh.close()

    for name, channel_class in list(channelcollection.__dict__.items()):
        if (
            isinstance(channel_class, type)
            and (name != "IonChannel" and name != "IonChannelSimplified")
            and "_func" not in name
        ):
            chan = channel_class()
            chan.write_cpp_code(path)

    fh = open(os.path.join(path, "Ionchannels.h"), "a")
    fcc = open(os.path.join(path, "Ionchannels.cc"), "a")
    fh.write("class ChannelCreator{\n")
    fh.write("public:\n")
    fh.write("    IonChannel* createInstance(string channel_name){\n")
    kk = 0
    for name, channel_class in list(channelcollection.__dict__.items()):
        if (
            isinstance(channel_class, type)
            and (name != "IonChannel" and name != "IonChannelSimplified")
            and "_func" not in name
        ):
            if kk == 0:
                fh.write('        if(channel_name == "' + name + '"){\n')
            else:
                fh.write('        else if(channel_name == "' + name + '"){\n')
            fh.write("            return new " + name + "();\n")
            fh.write("        }\n")
            kk += 1
    fh.write("    };" + "\n")
    fh.write("};" + "\n")

    fh.close()
    fcc.close()
