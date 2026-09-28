"""High-level, declarative interface to IBSimu.

Example::

    import ibsimu

    sim = ibsimu.Simulation(
        mode="2d",
        size=(101, 41, 1),
        h=1e-3,
        solids={7: lambda x, y, z: x > 80e-3},
        boundaries={7: -10e3},
        beams=[dict(method="2d_beam_with_energy", N=100, J=1e-3, q=1, m=1,
                    E=1e3, Tp=0, Tt=0, x1=0, y1=5e-3, x2=0, y2=35e-3)],
    )
    out = sim.run(iterations=3)
    out.info()
    epot = out.evaluate("epot")        # xarray.DataArray with x, y coordinates
    traj = out.evaluate("trajectories")

The raw library objects remain available on the output (``out.geom``,
``out.epot``, ``out.pdb``, ...) for anything the high-level API does not cover.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import _core

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_MODES = {
    "1d": _core.MODE_1D,
    "2d": _core.MODE_2D,
    "cyl": _core.MODE_CYL,
    "3d": _core.MODE_3D,
}
_MODE_NAMES = {v: k for k, v in _MODES.items()}

# Axis names of the node arrays per geometry mode.
_DIMS = {
    _core.MODE_1D: ("x",),
    _core.MODE_2D: ("x", "y"),
    _core.MODE_CYL: ("x", "r"),
    _core.MODE_3D: ("x", "y", "z"),
}

# Column names of Particle*.trajectory() per geometry mode.
_TRAJ_COORDS = {
    _core.MODE_2D: ("t", "x", "vx", "y", "vy"),
    _core.MODE_CYL: ("t", "x", "vx", "r", "vr", "w"),
    _core.MODE_3D: ("t", "x", "vx", "y", "vy", "z", "vz"),
}

_PDB_CLASSES = {
    _core.MODE_2D: _core.ParticleDataBase2D,
    _core.MODE_CYL: _core.ParticleDataBaseCyl,
    _core.MODE_3D: _core.ParticleDataBase3D,
}

_SOLVERS = {
    "gs": _core.EpotGSSolver,
    "mg": _core.EpotMGSolver,
    "bicgstab": _core.EpotBiCGSTABSolver,
}
if hasattr(_core, "EpotUMFPACKSolver"):
    _SOLVERS["umfpack"] = _core.EpotUMFPACKSolver

_BOUND_TYPES = {
    "neumann": _core.BOUND_NEUMANN,
    "dirichlet": _core.BOUND_DIRICHLET,
}

# Field keys understood by SimulationOutput.evaluate(): (label, units).  The
# attributes follow the plasma-plots conventions (``label`` in mathtext,
# ``units``; coordinates carry ``long_name`` and ``units``).
_FIELD_KEYS = {
    "epot": (r"$\phi$", "V"),
    "scharge": (r"$\rho$", "C/m$^3$"),
    "efield": (r"$\mathbf{E}$", "V/m"),
    "bfield": (r"$\mathbf{B}$", "T"),
    "trajdens": (r"$J$", "A/m$^2$"),
    "solid": ("solid", ""),
}
_KEYS = tuple(_FIELD_KEYS) + ("trajectories", "markers")

# Labels and units of trajectory coordinates and plane diagnostics.
_QUANTITIES = {
    "t": ("$t$", "s"),
    "x": ("$x$", "m"),
    "y": ("$y$", "m"),
    "r": ("$r$", "m"),
    "z": ("$z$", "m"),
    "vx": ("$v_x$", "m/s"),
    "vy": ("$v_y$", "m/s"),
    "vr": ("$v_r$", "m/s"),
    "vz": ("$v_z$", "m/s"),
    "w": (r"$\omega$", "rad/s"),
    "vtheta": (r"$v_\theta$", "m/s"),
    "xp": ("$x'$", "rad"),
    "yp": ("$y'$", "rad"),
    "rp": ("$r'$", "rad"),
    "zp": ("$z'$", "rad"),
    "ap": (r"$\alpha'$", "rad"),
    "curr": ("$I$", "A"),
    "ek": ("$E_k$", "eV"),
    "qm": ("$q/m$", "C/kg"),
    "charge": ("$q$", "C"),
    "mass": ("$m$", "kg"),
    "no": ("particle", ""),
}


def _quantity_attrs(name: str) -> dict[str, str]:
    label, units = _QUANTITIES.get(name, (name, ""))
    return {"label": label, "long_name": label, "units": units}


def _mode_of(mode) -> _core.GeometryMode:
    if isinstance(mode, str):
        try:
            return _MODES[mode.lower().replace("mode_", "")]
        except KeyError:
            raise ValueError(
                f"Unknown geometry mode {mode!r}; use one of {list(_MODES)}"
            ) from None
    return _core.GeometryMode(mode)


def _vec3(v) -> _core.Vec3D:
    if isinstance(v, _core.Vec3D):
        return v
    x, y, z = (list(v) + [0.0, 0.0])[:3]
    return _core.Vec3D(float(x), float(y), float(z))


def _int3(v) -> _core.Int3D:
    if isinstance(v, _core.Int3D):
        return v
    i, j, k = (list(v) + [1, 1])[:3]
    return _core.Int3D(int(i), int(j), int(k))


def _bound_of(spec) -> _core.Bound:
    """Bound from a Bound, a number (Dirichlet), ``(type, value)`` or ``None`` (Neumann 0)."""
    if spec is None:
        return _core.Bound(_core.BOUND_NEUMANN, 0.0)
    if isinstance(spec, _core.Bound):
        return spec
    if isinstance(spec, (int, float)):
        return _core.Bound(_core.BOUND_DIRICHLET, float(spec))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        btype, value = spec
        if isinstance(btype, str):
            try:
                btype = _BOUND_TYPES[btype.lower().replace("bound_", "")]
            except KeyError:
                raise ValueError(f"Unknown boundary type {btype!r}") from None
        if isinstance(value, (int, float)):
            return _core.Bound(btype, float(value))
        return _core.Bound(btype, value)  # position dependent functor
    raise TypeError(f"Cannot interpret boundary specification {spec!r}")


def _solid_of(spec) -> _core.Solid:
    if isinstance(spec, _core.Solid):
        return spec
    if callable(spec):
        return _core.FuncSolid(spec)
    raise TypeError(
        f"Cannot interpret solid specification {spec!r}; pass a Solid or f(x, y, z) -> bool"
    )


def _apply_options(obj, options: Mapping[str, Any] | None, what: str) -> None:
    """Call ``obj.set_<key>(value)`` for every option (tuple values are unpacked)."""
    for key, value in (options or {}).items():
        name = key if key.startswith("set_") else f"set_{key}"
        setter = getattr(obj, name, None)
        if setter is None:
            raise ValueError(
                f"{what} has no option {key!r} ({type(obj).__name__}.{name} does not exist)"
            )
        if isinstance(value, tuple):
            setter(*value)
        else:
            setter(value)


def _xarray():
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover
        raise ImportError(
            "xarray is required for SimulationOutput.evaluate(); pip install xarray"
        ) from None
    return xr


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class Simulation:
    """Declarative description of an IBSimu simulation.

    Parameters
    ----------
    mode
        Geometry mode: ``"2d"``, ``"cyl"``, ``"3d"`` or an ``ibsimu.MODE_*`` value.
    size
        Number of mesh nodes per axis, e.g. ``(101, 41, 1)``.
    h
        Mesh spacing [m].
    origo
        Position of node ``(0, 0, 0)`` [m]. Default: the origin.
    solids
        ``{number: solid}`` with number >= 7. A solid is an ``ibsimu.Solid`` or a
        callable ``f(x, y, z) -> bool`` (second coordinate is ``r`` in cylindrical mode).
    boundaries
        ``{number: bound}`` for mesh boundaries (1-6) and solids. A bound is an
        ``ibsimu.Bound``, a number (Dirichlet potential [V]), ``("neumann" | "dirichlet",
        value)`` or ``None``. Mesh boundaries not listed default to Neumann 0; every
        solid must be listed.
    beams
        List of beam definitions added to the particle database before each
        trajectory iteration. Each item is either ``{"method": name, **kwargs}``
        calling ``pdb.add_<name>(**kwargs)`` (the ``add_`` prefix is optional) or a
        callable ``f(pdb)``.
    bfield
        A ``VectorField`` (e.g. ``MeshVectorField``) used as the magnetic field.
    solver
        ``"gs"`` (Gauss-Seidel, default), ``"mg"``, ``"bicgstab"``, ``"umfpack"``, a
        solver class or a solver factory ``f(geom) -> EpotSolver``.
    solver_options
        ``{name: value}`` applied as ``solver.set_<name>(value)`` (tuples are
        unpacked), e.g. ``{"eps": 1e-4, "imax": 10000}``.
    pdb_options
        Same for the particle database, e.g. ``{"max_steps": 10000, "polyint": True,
        "mirror": [False, False, True, False, False, False]}``. Trajectories are
        saved by default (``save_trajectories=True``).
    plasma
        Plasma model of the solver: ``{"kind": "pexp", "rhoe": None, "Te": 5.0,
        "Up": 5.0}`` (positive ion extraction; ``rhoe=None`` uses the beam space
        charge from the previous iteration), ``{"kind": "nsimp", "rhop": ..,
        "Ep": .., "rhoi": [..], "Ei": [..]}`` or ``{"kind": "shield", "Tm": ..,
        "Um": ..}``. Applied from the second iteration on.
    initial_plasma
        ``{"Up": .., "axis": ibsimu.AXIS_X, "x": ..}``: forces the potential to
        ``Up`` on the side of the plane ``axis = x`` for the first iteration.
    efield_extrapolation
        List of six ``ibsimu.FIELD_*`` extrapolation modes for the electric field.
    threads
        Number of worker threads (``None`` keeps the library default).
    name
        Run name, stored as ``attrs["run"]`` on the output arrays (plasma-plots
        uses it as figure title).
    """

    def __init__(
        self,
        mode,
        size: Sequence[int],
        h: float,
        origo: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        solids: Mapping[int, Any] | None = None,
        boundaries: Mapping[int, Any] | None = None,
        beams: Sequence[Any] = (),
        bfield: _core.VectorField | None = None,
        solver: Any = "gs",
        solver_options: Mapping[str, Any] | None = None,
        pdb_options: Mapping[str, Any] | None = None,
        plasma: Mapping[str, Any] | None = None,
        initial_plasma: Mapping[str, Any] | None = None,
        efield_extrapolation: Sequence[Any] | None = None,
        threads: int | None = None,
        name: str = "",
    ):
        self.mode = _mode_of(mode)
        if self.mode not in _PDB_CLASSES:
            raise ValueError("Simulation supports the 2d, cyl and 3d geometry modes")
        self.size = tuple(int(n) for n in (list(size) + [1, 1])[:3])
        self.h = float(h)
        self.origo = tuple(float(v) for v in (list(origo) + [0.0, 0.0])[:3])
        self.solids = dict(solids or {})
        self.boundaries = dict(boundaries or {})
        self.beams = list(beams)
        self.bfield = bfield
        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self.pdb_options = {"save_trajectories": True, **(pdb_options or {})}
        self.plasma = dict(plasma) if plasma else None
        self.initial_plasma = dict(initial_plasma) if initial_plasma else None
        self.efield_extrapolation = (
            list(efield_extrapolation) if efield_extrapolation else None
        )
        self.threads = threads
        self.name = name

        for n in self.solids:
            if n < 7:
                raise ValueError(
                    f"Solid numbers start at 7 (got {n}); 1-6 are the mesh boundaries"
                )
            if n not in self.boundaries:
                raise ValueError(
                    f"Solid {n} has no boundary condition; add boundaries={{{n}: potential}}"
                )

    # -- construction --------------------------------------------------------

    def build_geometry(self) -> _core.Geometry:
        """Create the geometry, assign solids and boundaries and build the mesh."""
        geom = _core.Geometry(self.mode, _int3(self.size), _vec3(self.origo), self.h)
        for n, solid in self.solids.items():
            geom.set_solid(n, _solid_of(solid))
        for n in range(1, geom.number_of_boundaries() + 1):
            geom.set_boundary(n, _bound_of(self.boundaries.get(n)))
        for n in self.solids:
            geom.set_boundary(n, _bound_of(self.boundaries[n]))
        unknown = (
            set(self.boundaries)
            - set(self.solids)
            - set(range(1, geom.number_of_boundaries() + 1))
        )
        if unknown:
            raise ValueError(
                f"Boundary conditions given for unknown boundaries {sorted(unknown)}"
            )
        geom.build_mesh()
        return geom

    def build_solver(self, geom: _core.Geometry) -> _core.EpotSolver:
        if isinstance(self.solver, str):
            try:
                factory = _SOLVERS[self.solver.lower()]
            except KeyError:
                raise ValueError(
                    f"Unknown solver {self.solver!r}; use one of {list(_SOLVERS)}"
                ) from None
        else:
            factory = self.solver
        solver = factory(geom)
        _apply_options(solver, self.solver_options, "solver")
        if self.initial_plasma:
            ip = self.initial_plasma
            solver.set_initial_plasma(
                float(ip["Up"]), _core.InitialPlasma(ip["axis"], float(ip["x"]))
            )
        return solver

    def build_pdb(self, geom: _core.Geometry) -> _core.ParticleDataBase:
        pdb = _PDB_CLASSES[self.mode](geom)
        _apply_options(pdb, self.pdb_options, "particle database")
        return pdb

    def add_beams(self, pdb: _core.ParticleDataBase) -> None:
        for beam in self.beams:
            if callable(beam):
                beam(pdb)
                continue
            spec = dict(beam)
            try:
                method = spec.pop("method")
            except KeyError:
                raise ValueError(
                    f"Beam definition needs a 'method' entry: {beam!r}"
                ) from None
            name = method if method.startswith("add_") else f"add_{method}"
            func = getattr(pdb, name, None)
            if func is None:
                raise ValueError(f"{type(pdb).__name__} has no beam method {name!r}")
            func(**spec)

    def _apply_plasma(
        self, solver: _core.EpotSolver, pdb: _core.ParticleDataBase
    ) -> None:
        p = dict(self.plasma)
        kind = p.pop("kind")
        if kind == "pexp":
            rhoe = p.pop("rhoe", None)
            if rhoe is None:
                rhoe = -pdb.get_rhosum()
            solver.set_pexp_plasma(float(rhoe), float(p.pop("Te")), float(p.pop("Up")))
        elif kind == "nsimp":
            rhop = p.pop("rhop", None)
            if rhop is None:
                rhop = -pdb.get_rhosum()
            solver.set_nsimp_plasma(
                float(rhop),
                float(p.pop("Ep")),
                list(p.pop("rhoi", [])),
                list(p.pop("Ei", [])),
            )
        elif kind == "shield":
            solver.set_shield_plasma(float(p.pop("Tm")), float(p.pop("Um")))
        else:
            raise ValueError(
                f"Unknown plasma kind {kind!r}; use 'pexp', 'nsimp' or 'shield'"
            )
        if p:
            raise ValueError(f"Unused plasma parameters {sorted(p)}")

    # -- running ---------------------------------------------------------------

    def run(
        self,
        iterations: int = 1,
        *,
        callback: Callable[[int, SimulationOutput], None] | None = None,
        verbose: bool = True,
    ) -> SimulationOutput:
        """Run ``iterations`` Vlasov iterations (solve, trace, deposit space charge).

        ``callback(i, out)`` is invoked after every iteration with the partially
        filled output, e.g. to record custom diagnostics or to stop early by
        raising ``StopIteration``.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.threads is not None:
            _core.ibsimu.set_thread_count(int(self.threads))

        t0 = time.perf_counter()
        geom = self.build_geometry()
        solver = self.build_solver(geom)
        epot = _core.EpotField(geom)
        scharge = _core.MeshScalarField(geom)
        efield = _core.EpotEfield(epot)
        if self.efield_extrapolation:
            efield.set_extrapolation(list(self.efield_extrapolation))
        bfield = self.bfield if self.bfield is not None else _core.MeshVectorField()
        pdb = self.build_pdb(geom)

        out = SimulationOutput(
            self,
            geom=geom,
            solver=solver,
            epot=epot,
            scharge=scharge,
            efield=efield,
            bfield=bfield,
            pdb=pdb,
            name=self.name,
        )

        prev_epot = None
        prev_scharge = None
        for i in range(iterations):
            ti = time.perf_counter()
            if i == 1 and self.plasma:
                self._apply_plasma(solver, pdb)
            solver.solve(epot, scharge)
            efield.recalculate()
            pdb.clear()
            self.add_beams(pdb)
            pdb.iterate_trajectories(scharge, efield, bfield)

            e = epot.numpy()
            s = scharge.numpy()
            stats = pdb.get_statistics()
            record = {
                "iteration": i,
                "particles": pdb.size(),
                "epot_change": float(np.max(np.abs(e - prev_epot)))
                if prev_epot is not None
                else float("nan"),
                "scharge_change": float(np.max(np.abs(s - prev_scharge)))
                if prev_scharge is not None
                else float("nan"),
                "collisions": int(stats.total_collisions()),
                "current": float(stats.total_current()),
                "wall_time": time.perf_counter() - ti,
            }
            prev_epot, prev_scharge = e.copy(), s.copy()
            out.history.append(record)
            out.iterations = i + 1
            if verbose:
                print(
                    f"iteration {i + 1}/{iterations}: {record['particles']} particles, "
                    f"max |d epot| = {record['epot_change']:.3g} V, {record['wall_time']:.2f} s"
                )
            if callback is not None:
                try:
                    callback(i, out)
                except StopIteration:
                    break
        out.wall_time = time.perf_counter() - t0
        return out


# ---------------------------------------------------------------------------
# SimulationOutput
# ---------------------------------------------------------------------------


class SimulationOutput:
    """Result of :meth:`Simulation.run`.

    Attributes hold the library objects: ``geom``, ``solver``, ``epot``,
    ``scharge``, ``efield``, ``bfield`` and ``pdb``. :meth:`evaluate` converts
    them to :class:`xarray.DataArray` objects with physical coordinates.
    """

    def __init__(
        self,
        simulation: Simulation | None,
        *,
        geom,
        epot,
        scharge,
        pdb,
        solver=None,
        efield=None,
        bfield=None,
        name: str = "",
    ):
        self.simulation = simulation
        self.name = name
        self.geom = geom
        self.solver = solver
        self.epot = epot
        self.scharge = scharge
        self.efield = efield if efield is not None else _core.EpotEfield(epot)
        if efield is None:
            self.efield.recalculate()
        self.bfield = bfield if bfield is not None else _core.MeshVectorField()
        self.pdb = pdb
        self.history: list[dict[str, Any]] = []
        self.iterations = 0
        self.wall_time = float("nan")

    # -- introspection ---------------------------------------------------------

    @property
    def mode(self) -> _core.GeometryMode:
        return self.geom.geom_mode()

    @property
    def dims(self) -> tuple[str, ...]:
        """Axis names of the node arrays, e.g. ``("x", "y")``."""
        return _DIMS[self.mode]

    def coords(self) -> dict[str, np.ndarray]:
        """Node coordinates [m] keyed by axis name."""
        return dict(zip(self.dims, self.geom.node_coordinates()))

    def _xr_coords(self, xr) -> dict[str, Any]:
        return {
            d: xr.DataArray(c, dims=d, attrs=_quantity_attrs(d))
            for d, c in self.coords().items()
        }

    def _attrs(self, label: str, units: str, **extra) -> dict[str, Any]:
        attrs = {"label": label, "long_name": label, "units": units, **extra}
        if self.name:
            attrs["run"] = self.name
        return attrs

    def keys(self) -> tuple[str, ...]:
        """Keys accepted by :meth:`evaluate`."""
        return _KEYS

    def info(self) -> None:
        """Print a summary of the geometry, the run and the particle statistics."""
        g = self.geom
        mode = _MODE_NAMES[self.mode]
        extent = ", ".join(
            f"{d} = {g.origo(i) * 1e3:g} .. {g.max(i) * 1e3:g} mm"
            for i, d in enumerate(self.dims)
        )
        lines = [
            "IBSimu simulation output",
            f"  geometry     : {mode}, {' x '.join(str(n) for n in g.shape())} nodes, h = {g.h() * 1e3:g} mm",
            f"  extent       : {extent}",
            f"  solids       : {g.number_of_solids()}, boundaries: {g.number_of_boundaries()}",
        ]
        if self.solver is not None:
            lines.append(f"  solver       : {type(self.solver).__name__}")
        emin, emax = self.epot.get_minmax()
        lines.append(f"  potential    : {emin:.6g} .. {emax:.6g} V")
        smin, smax = self.scharge.get_minmax()
        lines.append(f"  space charge : {smin:.4g} .. {smax:.4g} C/m^3")
        lines.append(
            f"  particles    : {self.pdb.size()} (trajectories saved: {bool(self.pdb.get_save_trajectories())})"
        )
        stats = self.pdb.get_statistics()
        nb = stats.number_of_boundaries()
        if nb:
            per_bound = ", ".join(
                f"{b}: {stats.bound_collisions(b)} ({stats.bound_current(b):.4g} A)"
                for b in range(1, nb + 1)
                if stats.bound_collisions(b)
            )
            lines.append(
                f"  collisions   : {stats.total_collisions()} total, {stats.total_current():.4g} A"
            )
            if per_bound:
                lines.append(f"                 by boundary: {per_bound}")
            lines.append(
                f"  ended by     : time limit {stats.end_time()}, step limit {stats.end_step()}, "
                f"bad definition {stats.end_baddef()}"
            )
        if self.history:
            lines.append(
                f"  iterations   : {self.iterations}, wall time {self.wall_time:.2f} s"
            )
            for r in self.history:
                lines.append(
                    f"    {r['iteration'] + 1:3d}: {r['particles']} particles, "
                    f"max |d epot| = {r['epot_change']:.3g} V, "
                    f"max |d rho| = {r['scharge_change']:.3g} C/m^3, {r['wall_time']:.2f} s"
                )
        lines.append(f"  keys         : {', '.join(self.keys())}")
        print("\n".join(lines))

    # -- data access -------------------------------------------------------------

    def evaluate(self, key: str, **kwargs):
        """Return the quantity ``key`` as an :class:`xarray.DataArray`.

        Mesh quantities (``"epot"``, ``"scharge"``, ``"efield"``, ``"bfield"``,
        ``"trajdens"``, ``"solid"``) have one dimension per geometry axis with the
        node coordinates in metres; vector fields carry an extra ``component``
        dimension. ``"trajectories"`` has dimensions ``(particle, point, coord)``
        and is NaN padded to the longest trajectory; use
        ``da.sel(coord="x")`` to pick a coordinate. ``"markers"`` is an
        :class:`xarray.Dataset` with one variable per coordinate over
        ``(t, marker)``, interpolated onto a common time grid (``nt`` points,
        default 200), which is the layout plasma-plots expects for marker plots.
        """
        xr = _xarray()
        key = key.lower()
        if key == "trajectories":
            return self._trajectories(xr, **kwargs)
        if key == "markers":
            return self._markers(xr, **kwargs)
        if key not in _FIELD_KEYS:
            raise KeyError(f"Unknown key {key!r}; available: {', '.join(self.keys())}")
        if kwargs:
            raise TypeError(f"evaluate({key!r}) takes no keyword arguments")

        coords = self._xr_coords(xr)
        dims = self.dims
        attrs = self._attrs(*_FIELD_KEYS[key])
        if key == "epot":
            data = self.epot.numpy().copy()
        elif key == "scharge":
            data = self.scharge.numpy().copy()
        elif key == "trajdens":
            tdens = _core.MeshScalarField(self.geom)
            self.pdb.build_trajectory_density_field(tdens)
            data = tdens.numpy().copy()
        elif key == "solid":
            data = self.geom.solid_numpy()
            attrs["description"] = "boundary/solid number of each node, 0 = vacuum"
        else:
            field = self.efield if key == "efield" else self.bfield
            data = field.sample(self.geom)
            dims = dims + ("component",)
            coords = {**coords, "component": ["x", "y", "z"]}
        return xr.DataArray(data, dims=dims, coords=coords, name=key, attrs=attrs)

    def _particle_trajectories(self, particles):
        idx = list(range(self.pdb.size())) if particles is None else list(particles)
        return idx, [self.pdb.particle(i).trajectory() for i in idx]

    def _trajectories(self, xr, particles: Sequence[int] | None = None):
        names = _TRAJ_COORDS[self.mode]
        idx, trajs = self._particle_trajectories(particles)
        npoints = max((t.shape[0] for t in trajs), default=0)
        data = np.full((len(idx), npoints, len(names)), np.nan)
        for n, t in enumerate(trajs):
            data[n, : t.shape[0], :] = t
        return xr.DataArray(
            data,
            dims=("particle", "point", "coord"),
            coords={"particle": idx, "point": np.arange(npoints), "coord": list(names)},
            name="trajectories",
            attrs=self._attrs(
                "trajectories",
                "",
                description="SI units; NaN padded beyond the end of each trajectory",
            ),
        )

    def _markers(self, xr, nt: int = 200, particles: Sequence[int] | None = None):
        names = _TRAJ_COORDS[self.mode]
        idx, trajs = self._particle_trajectories(particles)
        t_end = max((float(t[-1, 0]) for t in trajs if t.shape[0]), default=0.0)
        tgrid = np.linspace(0.0, t_end, int(nt))
        data = {name: np.full((len(tgrid), len(idx)), np.nan) for name in names[1:]}
        for n, t in enumerate(trajs):
            if t.shape[0] == 0:
                continue
            for c, name in enumerate(names[1:], start=1):
                # Interpolate along the flight time; NaN after the particle ended.
                data[name][:, n] = np.interp(tgrid, t[:, 0], t[:, c], right=np.nan)
        return xr.Dataset(
            {
                name: (("t", "marker"), arr, _quantity_attrs(name))
                for name, arr in data.items()
            },
            coords={
                "t": xr.DataArray(tgrid, dims="t", attrs=_quantity_attrs("t")),
                "marker": idx,
            },
            attrs=self._attrs(
                "markers",
                "",
                description="trajectories interpolated on a common time grid",
            ),
        )

    def diagnostics(self, axis, value: float, diagnostics: Sequence[Any]):
        """Trajectory crossings of the plane ``axis = value`` as an :class:`xarray.Dataset`.

        ``diagnostics`` is a list of ``ibsimu.DIAG_*`` values; the variables are
        named after them (``DIAG_Y`` -> ``"y"``) and share the ``marker`` dimension.
        """
        xr = _xarray()
        diag = list(diagnostics)
        tdata = _core.TrajectoryDiagnosticData()
        self.pdb.trajectories_at_plane(tdata, axis, float(value), diag)
        data_vars = {}
        for j, d in enumerate(diag):
            name = str(d).split(".")[-1].removeprefix("DIAG_").lower()
            data_vars[name] = ("marker", tdata.column(j).data(), _quantity_attrs(name))
        attrs = self._attrs("diagnostics", "", axis=str(axis), value=float(value))
        return xr.Dataset(
            data_vars, coords={"marker": np.arange(tdata.traj_size())}, attrs=attrs
        )

    # -- persistence ---------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Write geometry, potential, space charge, particles and history to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        self.geom.save(os.path.join(directory, "geom.dat"))
        self.epot.save(os.path.join(directory, "epot.dat"))
        self.scharge.save(os.path.join(directory, "scharge.dat"))
        self.pdb.save(os.path.join(directory, "pdb.dat"))
        meta = {
            "name": self.name,
            "mode": _MODE_NAMES[self.mode],
            "iterations": self.iterations,
            "wall_time": self.wall_time,
            "history": self.history,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> SimulationOutput:
        """Load an output written by :meth:`save`."""
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        geom = _core.Geometry(os.path.join(directory, "geom.dat"))
        epot = _core.EpotField(os.path.join(directory, "epot.dat"), geom)
        scharge = _core.MeshScalarField(os.path.join(directory, "scharge.dat"))
        pdb = _PDB_CLASSES[geom.geom_mode()](os.path.join(directory, "pdb.dat"), geom)
        out = cls(
            None,
            geom=geom,
            epot=epot,
            scharge=scharge,
            pdb=pdb,
            name=meta.get("name", ""),
        )
        out.history = meta.get("history", [])
        out.iterations = meta.get("iterations", 0)
        out.wall_time = meta.get("wall_time", float("nan"))
        return out
