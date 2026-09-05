
# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""modelx-cython: Cythonize exported modelx models.

modelx-cython translates a modelx model exported as a pure-Python
package with modelx's ``Model.export()`` into a Cython package and
compiles it into C extension modules for faster execution.  The
translation is performed by the ``mx2cy`` command-line tool (see
:mod:`modelx_cython.cli`), which can also be run as
``python -m modelx_cython``.

Attributes
----------
version_info : tuple of int
    The version number as a tuple of integers.
__version__ : str
    The version number as a dotted string, derived from
    ``version_info``.
"""

version_info = (0, 1, 0)
__version__ = ".".join([str(x) for x in version_info])
