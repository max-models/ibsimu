# IBSimu Python interface

`import ibsimu` gives the C++ library through pybind11 bindings (`ibsimu._core`,
re-exported at the top level) and a declarative layer, `ibsimu.Simulation`, whose
results come out as xarray objects. Class and method names follow the C++ API
(see the Doxygen reference, `make doc`); this page covers what is specific to
Python.

## Installing

Build the C++ library first, then the extension against it:

```bash
./reconf && ./configure && make -j$(nproc)
export LD_LIBRARY_PATH=$(pwd)/src/.libs
LDFLAGS="-L$(pwd)/src/.libs" pip install -e ".[dev,examples]"
xvfb-run pytest tests/
```

The extension is compiled from `python/ibsimu_py*.cpp`; it picks up the features of
the library build (`config.h`): `EpotUMFPACKSolver` exists only with UMFPACK, and
`GTKPlotter` and `Geom3DPlot.plot_png` only with GTK3 (`ibsimu.HAVE_GTK`).

On macOS, `setup.py` links the static `src/.libs/libibsimu-*.a` when it exists, so
no `DYLD_LIBRARY_PATH` is needed. Install the dependencies with Homebrew
(`gsl libpng cairo freetype fontconfig suite-sparse pkg-config autoconf automake
libtool`); GTK is optional.

## Two levels

- **`ibsimu.Simulation`**: describe the geometry, boundaries, beams and solver
  in one call, `run(iterations)`, and get a `SimulationOutput` whose
  `evaluate("epot" | "scharge" | "efield" | "bfield" | "trajdens" | "solid" |
  "trajectories" | "markers")` returns `xarray` objects with physical coordinates;
  `diagnostics(axis, value, [DIAG_*])` gives the crossings of a plane;
  `save(dir)` / `SimulationOutput.load(dir)` store a run. See the docstring of
  `Simulation` for every option. In 1D, IBSimu traces no particles: give the space
  charge with `scharge=` (a number, `f(x, y, z)` or an array) instead of `beams`.
- **The library classes** (`Geometry`, `EpotField`, `EpotGSSolver`,
  `ParticleDataBase2D`, ...) for anything else. `SimulationOutput` keeps them
  (`out.geom`, `out.epot`, `out.pdb`, ...).

## Data as numpy arrays

- `MeshScalarField.numpy()` (and `EpotField`) is a writable view of the node data,
  shape `geom.shape()`: `(nx,)`, `(nx, ny)` or `(nx, ny, nz)`, indexed `a[i, j, k]`.
  `MeshVectorField.numpy()` is a copy with a trailing component axis;
  `set_numpy(a)` writes back.
- `Mesh.node_coordinates()` returns the coordinate axes.
- `Geometry.mesh_numpy()` and `solid_numpy()` give the node flags and the
  solid/boundary number of each node; `nearsolid_data(i, j, k)` decodes the
  distances to a solid surface stored for a near-solid node.
- `Particle*.trajectory()` is an `(npoints, ncoord)` array with the columns of
  the `ParticleP*` classes (`t, x, vx, y, vy`, ...).
- Any field can be sampled on a mesh: `field.sample(geom)`.

## Solvers

All solver settings of the library are available, also as keyword arguments of the
constructors:

```python
solver = ibsimu.EpotBiCGSTABSolver(geom, eps=1e-6, newton_imax=20, gnewton=True)
solver.set_preconditioner(ibsimu.ILU1_Precond())   # ILU0 (default), ILU1, Diag, Empty
solver.set_analysis_callback(lambda: print(solver.get_iter(), solver.get_error_estimate()))

mg = ibsimu.EpotMGSolver(geom)
mg.set_mgcycmax(50); mg.set_mgeps(1e-6); mg.set_levels(4)
```

The library stores the BiCGSTAB analysis callbacks as plain function pointers, so
the most recently registered callback is shared by all BiCGSTAB solvers; pass
`None` to remove it. With `Simulation`, give the settings as
`solver_options={"preconditioner": ibsimu.ILU1_Precond(), "newton_imax": 20}`.

The sparse matrix classes (`CRowMatrix`, `CColMatrix`, `CoordMatrix`), `Vector`
and the preconditioners can also be used on their own; `CRowMatrix(n, m, ptr, col,
val)` and `.arrays()` convert from and to the `scipy.sparse` CSR layout.
`HBIO` reads and writes Harwell-Boeing matrix files.

## Geometry from files

- `STLSolid(filename)`, or `STLSolid()` with `add_stl_file(STLFile(...))` for
  several files. `STLFile(vertices, triangles)` builds a solid from a triangle
  mesh (`Vec3D` vertices, `VTriangle(i, j, k)` triangles, outward oriented);
  `VTriangleSurfaceSolid` does the same without the file wrapper.
- `DXFSolid(MyDXFFile(filename), layer)` for 2D/cylindrical electrodes. The
  contents of the file are open for inspection and editing first:
  `dxf.get_entities()` (lines, arcs, circles, polylines, splines, inserts as
  their own classes), `selection_layer()`, `selection_path_loop()`, `get_bbox()`,
  `scale()`, `translate()`, `rotate_z()`, `dxf.get_blocks()`, `dxf.get_tables()`.

## Callbacks and threads

Python callables can define solids (`FuncSolid(f)`), position-dependent
boundaries and functors (subclass `CallbackFunctorD_V` etc. and define
`__call__`), trajectory handlers, end and surface-collision callbacks, and
random variate transformations. The library runs solvers and particle tracing in
worker threads with the GIL released; callbacks take the GIL when called, so
they work but run one at a time. An exception raised in a callback is re-raised
when the library call returns.

## Ownership

The geometry deletes its solids and an `STLSolid` its `STLFile`s: after
`geom.set_solid(n, solid)` or `stl_solid.add_stl_file(f)` the object belongs to
C++ and must not be reused for another geometry. Objects the library only refers
to (fields given to plotters and graphs, graphs added to a `Frame`, the database
of a callback) are kept alive by their Python owner.

## Plotting

- High level: `GeomPlotter`, `FieldDiagPlotter` and `ParticleDiagPlotter` write
  PNG, PDF, SVG or EPS files (`plot_png`, ...).
- Building blocks for custom figures: a `Frame` with graphs (`XYGraph`,
  `FieldGraph`, `SolidGraph`, `EqPotGraph`, `ParticleGraph`, `MeshGraph`,
  `MeshColormap`), `LegendEntry`, colour `Palette`s, then
  `frame.plot_png(filename, width, height)` (or `plot_pdf`, `plot_svg`).
- `Geom3DPlot(geom, pdb)` sets up a 3D view (camera, cut planes, beam);
  `plot_png` renders it with the software renderer (GTK3 builds).
- `GTKPlotter` opens the interactive viewer of the C++ library (needs a display
  and a GTK3 build): set the data, open windows, `run()`.
- For figures in Python, `SimulationOutput.evaluate()` returns xarray data that
  plots directly with matplotlib or plasma-plots (see
  `examples_py/viz_plasma_plots.py`, `viz_pyvista.py`).

## Space charge from your own trajectories

`scharge_add_from_trajectory_pic(scharge, I, x1, x2)` deposits the charge of a
trajectory segment (`ParticleP2D`/`ParticlePCyl`/`ParticleP3D` end points) carrying
current `I`; `scharge_finalize_pic(scharge)` turns it into charge density.
`scharge_add_step_pic(scharge, Q, x)` and `scharge_finalize_step_pic` deposit
point charges (3D meshes only).

## Errors

Library errors raise `ibsimu.IBSimuError` (a `RuntimeError`) or one of its
subclasses, which also derive from the matching builtin exception:
`IBSimuRangeError` (`IndexError`), `IBSimuDimError` (`ValueError`),
`IBSimuUnimplementedError` (`NotImplementedError`), `IBSimuNoMemError`
(`MemoryError`), `IBSimuAssertError` (`AssertionError`) and `IBSimuErrnoError`
(`OSError`).

## Not wrapped

Internals of the library stay in C++: the particle iterator and stepper templates
(`ParticleIterator`, `ParticleStepper`, `Scheduler`; use `iterate_trajectories` and
`step_particles`), the multigrid sub-solver, the linear-interpolation space charge
kernels, the protected matrix assembly of `EpotMatrixSolver`, the DXF group
parser, the right-hand side function of the particle equations of motion
(`get_derivatives`) and drawing on a raw cairo context (`Frame` and the plotters
draw to files instead).
