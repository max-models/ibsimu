"""Sparse matrices, vectors, preconditioners and the electrostatic solver settings."""

import numpy as np
import pytest

import ibsimu

# 3x3 test matrix [[4, 1, 0], [1, 4, 0], [0, 1, 4]] in CSR form.
PTR = np.array([0, 2, 4, 6])
COL = np.array([0, 1, 0, 1, 1, 2])
VAL = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 4.0])
DENSE = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 1.0, 4.0]])


def test_vector_numpy_and_arithmetic():
    v = ibsimu.Vector(np.array([1.0, 2.0, 3.0]))
    assert len(v) == 3
    np.testing.assert_array_equal(np.asarray(v), [1.0, 2.0, 3.0])
    v.numpy()[0] = 10.0
    assert v[0] == 10.0 and v[-1] == 3.0
    w = 2.0 * v - ibsimu.Vector(3, 1.0)
    np.testing.assert_array_equal(np.asarray(w), [19.0, 3.0, 5.0])
    assert v.dot(ibsimu.Vector(3, 1.0)) == pytest.approx(15.0)
    assert v.norm2() == pytest.approx(np.sqrt(100 + 4 + 9))
    assert v.max() == 10.0 and v.min() == 2.0
    with pytest.raises(IndexError):
        v[3]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ibsimu.CRowMatrix(3, 3, PTR, COL, VAL),
        lambda: ibsimu.CColMatrix(ibsimu.CRowMatrix(3, 3, PTR, COL, VAL)),
        lambda: ibsimu.CoordMatrix(3, 3, *np.nonzero(DENSE), DENSE[np.nonzero(DENSE)]),
    ],
    ids=["crow", "ccol", "coord"],
)
def test_sparse_matrices(build):
    a = build()
    assert a.shape() == (3, 3)
    assert a.nz_elements() == 6
    np.testing.assert_array_equal(a.toarray(), DENSE)
    x = a @ ibsimu.Vector(np.ones(3))
    np.testing.assert_allclose(np.asarray(x), DENSE @ np.ones(3))
    a[2, 0] = 7.0
    assert a.get(2, 0) == 7.0
    with pytest.raises(IndexError):
        a.get(3, 0)


def test_matrix_arrays_round_trip():
    a = ibsimu.CRowMatrix(3, 3, PTR, COL, VAL)
    ptr, col, val = a.arrays()
    np.testing.assert_array_equal(ptr, PTR)
    np.testing.assert_array_equal(col, COL)
    np.testing.assert_array_equal(val, VAL)
    c = ibsimu.CColMatrix(a)
    ptr, _row, _val = c.arrays()
    np.testing.assert_array_equal(ptr, [0, 2, 5, 6])
    b = ibsimu.CRowMatrix(3, 3)
    for i in range(3):
        for j in range(3):
            if DENSE[i, j]:
                b.construct_add(i, j, DENSE[i, j])
    np.testing.assert_array_equal(b.toarray(), DENSE)


@pytest.mark.parametrize(
    "pc",
    [
        ibsimu.ILU0_Precond,
        ibsimu.ILU1_Precond,
        ibsimu.Diag_Precond,
        ibsimu.Empty_Precond,
    ],
)
def test_preconditioners(pc):
    a = ibsimu.CRowMatrix(3, 3, PTR, COL, VAL)
    p = pc()
    p.prepare(a)
    p.construct(a)
    assert p.is_prepared()
    b = a @ ibsimu.Vector(np.ones(3))
    x = np.asarray(p.solve(b))
    if pc in (ibsimu.ILU0_Precond, ibsimu.ILU1_Precond):
        # No fill-in is dropped for this matrix: the ILU is the exact LU.
        np.testing.assert_allclose(x, np.ones(3))
    elif pc is ibsimu.Diag_Precond:
        np.testing.assert_allclose(x, np.asarray(b) / np.diag(DENSE))
    else:
        np.testing.assert_allclose(x, np.asarray(b))


def test_hbio_round_trip(tmp_path):
    h = ibsimu.HBIO()
    h.set_title("test matrix")
    h.set_key("T1")
    h.set_matrix(ibsimu.CColMatrix(ibsimu.CRowMatrix(3, 3, PTR, COL, VAL)))
    h.set_rhs_vector(ibsimu.Vector(np.array([1.0, 2.0, 3.0])))
    fn = str(tmp_path / "a.hb")
    h.write(fn)
    g = ibsimu.HBIO(fn)
    np.testing.assert_allclose(g.get_matrix().toarray(), DENSE)
    np.testing.assert_allclose(np.asarray(g.get_rhs_vector()), [1.0, 2.0, 3.0])
    assert g.get_title().strip() == "test matrix"


@pytest.fixture()
def capacitor_1d():
    """1D capacitor 0 V / 10 V over 10 mm filled with 1e-6 C/m^3."""
    geom = ibsimu.Geometry(
        ibsimu.MODE_1D, ibsimu.Int3D(11, 1, 1), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 10.0))
    geom.build_mesh()
    scharge = ibsimu.MeshScalarField(geom)
    scharge.numpy()[...] = 1e-6
    exact = 5.0 + 1e-6 / (2 * ibsimu.EPSILON0) * 5e-3 * 5e-3
    return geom, scharge, exact


def test_bicgstab_settings_and_callbacks(capacitor_1d):
    geom, scharge, exact = capacitor_1d
    s = ibsimu.EpotBiCGSTABSolver(geom, eps=1e-10, newton_imax=5, gnewton=False)
    s.set_preconditioner(ibsimu.Diag_Precond())
    s.set_newton_eps(1e-6)
    s.set_imax(1000)
    calls = []
    s.set_analysis_callback(lambda: calls.append(s.get_iter()))
    epot = ibsimu.EpotField(geom)
    s.solve(epot, scharge)
    s.set_analysis_callback(None)
    assert epot.numpy()[5] == pytest.approx(exact, rel=1e-6)
    assert calls and s.get_iter() >= 1
    assert s.get_error_estimate() >= 0.0


def test_bicgstab_callback_error_propagates(capacitor_1d):
    geom, scharge, _ = capacitor_1d
    s = ibsimu.EpotBiCGSTABSolver(geom, eps=1e-10)
    # ILU0 is exact for the tridiagonal 1D matrix; force some iterations.
    s.set_preconditioner(ibsimu.Diag_Precond())

    def boom():
        raise RuntimeError("from callback")

    s.set_analysis_callback(boom)
    try:
        with pytest.raises(RuntimeError, match="from callback"):
            s.solve(ibsimu.EpotField(geom), scharge)
    finally:
        s.set_analysis_callback(None)


def test_mg_settings(capacitor_1d):
    geom, scharge, exact = capacitor_1d
    s = ibsimu.EpotMGSolver(geom)
    s.set_mgcycmax(100)
    s.set_mgeps(1e-10)
    s.set_local_imax(5)
    s.set_w(1.0)
    epot = ibsimu.EpotField(geom)
    s.solve(epot, scharge)
    assert s.get_mgcyc() < 100
    assert epot.numpy()[5] == pytest.approx(exact, rel=1e-6)


@pytest.mark.skipif(
    not hasattr(ibsimu, "EpotUMFPACKSolver"), reason="built without UMFPACK"
)
def test_umfpack_settings(capacitor_1d):
    geom, scharge, exact = capacitor_1d
    s = ibsimu.EpotUMFPACKSolver(geom, newton_imax=5)
    s.set_gnewton(False)
    s.set_newton_residual_eps(1e-8)
    s.set_newton_step_eps(1e-9)
    s.set_newton_imax(3)
    epot = ibsimu.EpotField(geom)
    s.solve(epot, scharge)
    assert epot.numpy()[5] == pytest.approx(exact, rel=1e-9)
    assert s.get_newton_residual() >= 0.0


def test_solver_set_parameters(capacitor_1d):
    geom, _, _ = capacitor_1d
    a = ibsimu.EpotGSSolver(geom)
    a.set_pexp_plasma(-1e-6, 5.0, 10.0)
    b = ibsimu.EpotBiCGSTABSolver(geom)
    b.set_parameters(a)
    b.set_nsimp_initial_plasma(ibsimu.InitialPlasma(ibsimu.AXIS_X, 2e-3))
