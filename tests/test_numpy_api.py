"""numpy interop, file loaders and exception mapping of the Python bindings."""

import numpy as np
import pytest

import ibsimu


@pytest.fixture(scope="module")
def solved_2d():
    """Small 2D problem: grounded box with a -1 kV electrode at x > 40 mm."""
    ibsimu.ibsimu.set_thread_count(1)
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(51, 21, 1), ibsimu.Vec3D(0, 0, 0), h
    )
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, y, z: x > 40e-3))
    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -1000.0))
    geom.build_mesh()

    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    ibsimu.EpotGSSolver(geom).solve(epot, scharge)

    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_save_trajectories(True)
    pdb.add_2d_beam_with_energy(5, 1e-3, 1.0, 1.0, 1000.0, 0, 0, 0, 5e-3, 0, 15e-3)
    pdb.iterate_trajectories(scharge, ibsimu.EpotEfield(epot), ibsimu.MeshVectorField())
    return geom, epot, scharge, pdb


def test_mesh_shape_and_coordinates(solved_2d):
    geom = solved_2d[0]
    assert geom.shape() == (51, 21)
    assert tuple(geom.size_vec()) == (51, 21, 1)
    xs, ys = geom.node_coordinates()
    assert xs.shape == (51,) and ys.shape == (21,)
    assert xs[1] == pytest.approx(geom.h())
    assert ys[-1] == pytest.approx(geom.max(1))


def test_scalar_field_numpy_is_a_writable_view(solved_2d):
    _, epot, _, _ = solved_2d
    a = epot.numpy()
    assert a.shape == (51, 21)
    assert a.flags.f_contiguous  # x index varies fastest, as in the C++ storage
    assert a.base is epot
    assert a[45, 10] == epot.get3(45, 10, 0) == -1000.0
    # Writes go straight into the field.
    old = epot.get3(0, 0, 0)
    a[0, 0] = 42.0
    assert epot.get3(0, 0, 0) == 42.0
    a[0, 0] = old


def test_scalar_field_array_protocol(solved_2d):
    _, epot, _, _ = solved_2d
    assert np.asarray(epot).shape == (51, 21)
    assert np.array(epot).base is None  # np.array copies
    assert np.asarray(epot, dtype=np.float32).dtype == np.float32
    assert epot.numpy().min() == pytest.approx(-1000.0)


def test_scalar_field_set_numpy(solved_2d):
    geom = solved_2d[0]
    f = ibsimu.MeshScalarField(geom)
    f.set_numpy(np.full(geom.shape(), 2.5))
    assert f.get3(3, 3, 0) == 2.5
    xs, ys = geom.node_coordinates()
    f.set_numpy(xs[:, None] + 0 * ys[None, :])  # broadcast x-coordinate
    assert f.get3(10, 7, 0) == pytest.approx(10e-3)
    with pytest.raises(ValueError):
        f.set_numpy(np.zeros((3, 3)))


def test_vector_field_numpy_roundtrip(solved_2d):
    geom = solved_2d[0]
    bf = ibsimu.MeshVectorField(geom, [True, True, False])
    arr = np.zeros(geom.shape() + (3,))
    arr[..., 0] = 1.0
    arr[5, 5, 1] = 7.0
    bf.set_numpy(arr)
    assert tuple(bf.get3(5, 5, 0)) == (1.0, 7.0, 0.0)
    out = bf.numpy()
    assert out.shape == (51, 21, 3)
    np.testing.assert_array_equal(out, arr)
    bf *= 2.0
    assert bf.numpy()[5, 5, 1] == 14.0
    with pytest.raises(ValueError):
        bf.set_numpy(np.zeros((51, 21)))


def test_vector_field_numpy_3d_layout():
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(4, 5, 6), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    bf = ibsimu.MeshVectorField(geom, [True, True, True])
    i, j, k = np.meshgrid(np.arange(4), np.arange(5), np.arange(6), indexing="ij")
    arr = np.stack([i, j, k], axis=-1).astype(float)
    bf.set_numpy(arr)
    assert tuple(bf.get3(3, 1, 4)) == (3.0, 1.0, 4.0)
    np.testing.assert_array_equal(bf.numpy(), arr)


def test_geometry_solid_numpy(solved_2d):
    geom = solved_2d[0]
    sm = geom.solid_numpy()
    assert sm.shape == (51, 21) and sm.dtype == np.uint32
    assert sm[45, 10] == 7  # inside the electrode
    assert sm[10, 10] == 0  # vacuum
    assert sm[10, 0] in (1, 2, 3, 4)  # mesh boundary
    raw = geom.mesh_numpy()
    assert raw.shape == (51, 21) and not raw.flags.writeable
    assert raw[45, 10] & 0xFF == 7


def test_trajectory_array(solved_2d):
    _, _, _, pdb = solved_2d
    p = pdb.particle(0)
    traj = p.trajectory()
    assert traj.shape == (p.traj_size(), 5)
    assert traj[0, ibsimu.PARTICLE_T] == 0.0
    assert traj[0, ibsimu.PARTICLE_Y] == pytest.approx(p.traj(0).y())
    assert traj[-1, ibsimu.PARTICLE_X] == pytest.approx(p.traj(p.traj_size() - 1).x())
    assert traj[-1, ibsimu.PARTICLE_X] > 39e-3  # reached the electrode


def test_diagnostic_column_and_histograms(solved_2d):
    _, _, _, pdb = solved_2d
    tdata = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_Y])
    pdb.trajectories_at_plane(tdata, ibsimu.AXIS_X, 30e-3, [ibsimu.DIAG_Y])
    y = tdata.column(0).data()
    assert isinstance(y, np.ndarray) and y.shape == (5,)

    h1 = ibsimu.Histogram1D(10, y)
    assert h1.get_data().shape == (10,)
    assert h1.get_data().sum() == pytest.approx(5.0)

    h2 = ibsimu.Histogram2D(4, 5, [0.0, 0.0, 1.0, 1.0])
    h2.accumulate_closest(0.9, 0.1, 1.0)
    d = h2.get_data()
    assert d.shape == (4, 5)
    assert d[3, 0] == h2(3, 0) == 1.0


def test_save_and_load_roundtrip(solved_2d, tmp_path):
    geom, epot, _, pdb = solved_2d
    geom.save(str(tmp_path / "geom.dat"))
    epot.save(str(tmp_path / "epot.dat"))
    pdb.save(str(tmp_path / "pdb.dat"))

    geom2 = ibsimu.Geometry(str(tmp_path / "geom.dat"))
    assert geom2.shape() == geom.shape()
    assert geom2.get_boundary(7).value() == -1000.0
    assert len(geom2.get_boundaries()) == geom.number_of_boundaries()

    epot2 = ibsimu.EpotField(str(tmp_path / "epot.dat"), geom2)
    np.testing.assert_array_equal(epot2.numpy(), epot.numpy())

    pdb2 = ibsimu.ParticleDataBase2D(str(tmp_path / "pdb.dat"), geom2)
    assert pdb2.size() == pdb.size()
    np.testing.assert_array_equal(
        pdb2.particle(0).trajectory(), pdb.particle(0).trajectory()
    )

    with pytest.raises(RuntimeError):
        ibsimu.Geometry(str(tmp_path / "missing.dat"))


def test_func_solid_cannot_be_saved(solved_2d, tmp_path):
    geom = solved_2d[0]
    with pytest.raises(RuntimeError, match="FuncSolid"):
        geom.save(str(tmp_path / "geom.dat"), save_solids=True)


def test_bound_api():
    b = ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 5.0)
    assert b.is_constant()
    b.set_value(6.0)
    assert b.value() == 6.0
    assert "BOUND_DIRICHLET" in repr(b)

    class Ramp(ibsimu.CallbackFunctorD_V):
        def __call__(self, x):
            return 100.0 * x[0]

    b2 = ibsimu.Bound(ibsimu.BOUND_DIRICHLET, Ramp())
    assert not b2.is_constant()
    assert b2.value_at(ibsimu.Vec3D(0.5, 0, 0)) == 50.0


def test_library_errors_become_python_exceptions():
    assert issubclass(ibsimu.IBSimuError, RuntimeError)
    with pytest.raises(ibsimu.IBSimuError, match="histogram size"):
        ibsimu.Histogram2D(2, 2, [0.0, 0.0, 1.0, 1.0])


def test_trajectory_end_callback_runs_from_worker_threads(solved_2d):
    """Python callbacks are invoked by the tracer while the GIL is released."""
    geom, epot, scharge, _ = solved_2d
    ibsimu.ibsimu.set_thread_count(2)
    seen = []

    class Recorder(ibsimu.TrajectoryEndCallback):
        def __call__(self, particle, pdb):
            seen.append((particle.get_status(), particle.location().x))

    pdb = ibsimu.ParticleDataBase2D(geom)
    cb = Recorder()
    pdb.set_trajectory_end_callback(cb)
    pdb.add_2d_beam_with_energy(8, 1e-3, 1.0, 1.0, 1000.0, 0, 0, 0, 5e-3, 0, 15e-3)
    pdb.iterate_trajectories(scharge, ibsimu.EpotEfield(epot), ibsimu.MeshVectorField())
    ibsimu.ibsimu.set_thread_count(1)
    assert len(seen) == 8
    assert all(status == ibsimu.PARTICLE_COLL for status, _ in seen)
    assert all(x > 39e-3 for _, x in seen)


def test_exception_in_callback_is_raised_in_caller(solved_2d):
    geom, epot, scharge, _ = solved_2d

    class Boom(ibsimu.TrajectoryEndCallback):
        def __call__(self, particle, pdb):
            raise KeyError("from callback")

    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_trajectory_end_callback(Boom())  # keep_alive: no reference needed
    pdb.add_2d_beam_with_energy(4, 1e-3, 1.0, 1.0, 1000.0, 0, 0, 0, 5e-3, 0, 15e-3)
    with pytest.raises(KeyError, match="from callback"):
        pdb.iterate_trajectories(
            scharge, ibsimu.EpotEfield(epot), ibsimu.MeshVectorField()
        )
    # The error slot is cleared: the next call reports its own problem.
    pdb.reset_trajectories()  # ended particles are not re-traced otherwise
    pdb.set_trajectory_end_callback(ibsimu.TrajectoryEndCallback())
    with pytest.raises(RuntimeError, match="__call__"):
        pdb.iterate_trajectories(
            scharge, ibsimu.EpotEfield(epot), ibsimu.MeshVectorField()
        )


def test_exception_in_func_solid_is_raised(solved_2d):
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(11, 11, 1), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, y, z: 1 / 0 > 1))
    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    with pytest.raises(ZeroDivisionError):
        geom.build_mesh()
