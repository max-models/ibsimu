"""IBSimu: ion optics simulation library.

The C++ bindings live in :mod:`ibsimu._core` and are re-exported here. The
:class:`Simulation` class offers a declarative, high-level way to set up and run
a simulation and to analyse the result with xarray.
"""

from ._core import *
from .simulation import Simulation, SimulationOutput  # noqa: F401
