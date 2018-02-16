'''
Initialize MPITS

Written by R. Jolivet 2017

License:
    MPITS: Multi-Pixel InSAR Time Series
    Copyright (C) 2018  <Romain Jolivet>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

# Parent Class
from .massive import tsmassive

# Child Classes
from .nsbas import nsbas
from .timefn import timefn
from .classic import classic

# Solver classes
from .petscsolver import petscsolver
#from .sdsolver import sdsolver
from .cdsolver import cdsolver

# Utils
from . import utils 

