# IBSimu - Ion Beam Simulator

IBSimu is an ion optics simulation library with several capabilities for solving electric fields in a geometry and tracing particles in electric and magnetic fields.

## Features

1. **Finite Difference Method (FDM)** for solving Poisson's equation in 1D, 2D, and 3D spaces, including cylindrical symmetry.
2. **Particle trajectory calculation** in the solved potential.
3. **Space charge density calculation** from the trajectories.
4. **Vlasov iteration** for self-consistent beam simulations.
5. **Python API** for high-level simulation control and data analysis.

## Installation

### Prerequisites

IBSimu requires several C++ and system dependencies. On Ubuntu/Debian, you can install them with:

```bash
sudo apt-get update
sudo apt-get install -y \
    libgsl-dev libpng-dev libcairo2-dev libfreetype6-dev \
    libfontconfig1-dev pkg-config autoconf automake libtool \
    zlib1g-dev libsuitesparse-dev libgl1-mesa-dev xvfb
```

### Building the C++ Library

```bash
./reconf
./configure
make -j$(nproc)
```

### Installing the Python API

The Python API is built using `pybind11`. To install it from the source directory:

```bash
# Ensure the C++ library is built first (see above)
export LD_LIBRARY_PATH=$(pwd)/src/.libs
LDFLAGS="-L$(pwd)/src/.libs" pip install .
```

## Running the Python API

After installation, you can import `ibsimu` in Python.

### Simple Example

```python
import ibsimu

# Create a 2D geometry
h = 1e-3
geom = ibsimu.Geometry(ibsimu.MODE_2D, ibsimu.Int3D(101, 41, 1), ibsimu.Vec3D(0, 0, 0), h)
geom.build_mesh()

# Define fields and solvers
epot = ibsimu.EpotField(geom)
scharge = ibsimu.MeshScalarField(geom)
solver = ibsimu.EpotGSSolver(geom)

# Solve Poisson's equation
solver.solve(epot, scharge)

print("Simulation complete.")
```

### Running Tests and Examples

You can find more complex examples in the `examples_py/` directory:

```bash
python examples_py/sim_2d_lens.py
```

To run the test suite:

```bash
xvfb-run pytest tests/
```

*Note: `xvfb-run` is required if you are running in a headless environment (like a server or CI) because the library may initialize graphical components (GTK/OpenGL) for plotting.*

## Documentation

IBSimu uses Doxygen for its reference manual. To generate it:

```bash
make doc
```
The documentation will be available in the `doc/` directory.

## License

IBSimu is distributed under the GNU General Public License (GPL) version 2 or later. See the `COPYING` file for details.

## Author

Taneli Kalvas <taneli.kalvas@jyu.fi>
