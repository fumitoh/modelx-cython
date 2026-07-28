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

"""Entry point for ``python -m modelx_cython``.

Running the package as a script invokes
:func:`modelx_cython.cli.entry_point_main`, the same function behind
the ``mx2cy`` console command.
"""

from modelx_cython.cli import entry_point_main

entry_point_main()
