"""Exceptions, small API additions, space charge deposition, plotting building blocks
and the 1D Simulation."""

import numpy as np
import pytest

import ibsimu


@pytest.fixture(scope="module")
def solved_2d():
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(41, 21, 1), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, y, z: x > 30e-3 and y < 8e-3))
    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 5.0))
    geom.build_mesh()
    epot = ibsimu.EpotField(geom)
    ibsimu.EpotGSSolver(geom).solve(epot, ibsimu.MeshScalarField(geom))
    return geom, epot


# -- exceptions and small additions ---------------------------------------------


def test_exception_hierarchy():
    assert issubclass(ibsimu.IBSimuRangeError, ibsimu.IBSimuError)
    assert issubclass(ibsimu.IBSimuRangeError, IndexError)
    assert issubclass(ibsimu.IBSimuDimError, ValueError)
    assert issubclass(ibsimu.IBSimuUnimplementedError, NotImplementedError)
    assert issubclass(ibsimu.IBSimuErrnoError, OSError)
    with pytest.raises(ibsimu.IBSimuError, match="histogram"):
        ibsimu.Histogram2D(1, 1, [0.0, 0.0, 1.0, 1.0])


def test_vector_and_transformation_extras():
    v = ibsimu.Vec3D(1.0, 2.0, 3.0)
    p = v.arb_perpendicular()
    assert ibsimu.dot(v, p) == pytest.approx(0.0, abs=1e-12)
    assert tuple(ibsimu.Vec3D.standard_basis(1)) == (0, 1, 0)
    assert ibsimu.Vec3D(3, -1, 2).min_element() == 1
    h = ibsimu.Vec4D(2.0, 4.0, 6.0, 2.0)
    h.homogenize()
    assert [h[i] for i in range(4)] == [1, 2, 3, 1]
    t = ibsimu.Transformation()
    t.translate(ibsimu.Vec3D(1, 0, 0))
    hp = t.transform_homogenous_point(ibsimu.Vec3D(0, 0, 0))
    assert hp[0] == pytest.approx(1.0) and hp[3] == pytest.approx(1.0)


def test_ibsimu_singleton_extras():
    ibsimu.ibsimu.set_thread_count(2)
    assert ibsimu.ibsimu.get_thread_count() == 2
    assert ibsimu.ibsimu.get_rng_type() in (ibsimu.RNG_SOBOL, ibsimu.RNG_MT)
    ibsimu.ibsimu.message("", 5)
    ibsimu.ibsimu.flush()


def test_histogram_accumulate():
    h = ibsimu.Histogram1D(4, [0.0, 1.0])
    h.accumulate(2, 3.0)
    h.accumulate(2)
    np.testing.assert_array_equal(h.get_data(), [0, 0, 4, 0])
    with pytest.raises(IndexError):
        h.accumulate(4)
    h2 = ibsimu.Histogram2D(4, 4, [0.0, 0.0, 1.0, 1.0])
    h2.accumulate(1, 2, 0.5)
    assert h2(1, 2) == 0.5


def test_particle_coordinate_extras():
    x1 = ibsimu.ParticleP2D(0.0, 0.0, 1.0, 0.0, 0.0)
    x2 = ibsimu.ParticleP2D(1.0, 1.0, 1.0, 0.0, 0.0)
    assert ibsimu.ParticleP2D(0, 0, 3.0, 0, 4.0).speed() == 5.0
    assert ibsimu.ParticleP2D.dim() == 2 and ibsimu.ParticleP3D.dim() == 3
    cross = ibsimu.ParticleP2D.trajectory_intersections_at_plane(0, 0.25, x1, x2)
    assert len(cross) == 1
    assert cross[0].x() == pytest.approx(0.25)
    assert cross[0].t() == pytest.approx(0.25)


def test_readascii(tmp_path):
    fn = tmp_path / "data.txt"
    fn.write_text("# comment\n1 2 3\n4 5 6\n")
    r = ibsimu.ReadAscii(str(fn))
    assert (r.rows(), r.columns()) == (2, 3)
    np.testing.assert_array_equal(r.numpy(), [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(r[1], [2, 5])


# -- space charge deposition --------------------------------------------------------


def test_scharge_from_trajectory_pic(solved_2d):
    geom, _ = solved_2d
    sc = ibsimu.MeshScalarField(geom)
    current, dt = 1e-3, 1e-7
    x1 = ibsimu.ParticleP2D(0.0, 5.2e-3, 1e5, 5.3e-3, 0.0)
    x2 = ibsimu.ParticleP2D(dt, 15.2e-3, 1e5, 5.3e-3, 0.0)
    ibsimu.scharge_add_from_trajectory_pic(sc, current, x1, x2)
    ibsimu.scharge_finalize_pic(sc)
    # Total charge per unit length: I dt.
    assert sc.numpy().sum() * geom.h() ** 2 == pytest.approx(current * dt, rel=1e-6)


def test_scharge_step_pic_3d_only(solved_2d):
    geom, _ = solved_2d
    with pytest.raises(ValueError):
        ibsimu.scharge_add_step_pic(
            ibsimu.MeshScalarField(geom), 1e-12, ibsimu.Vec3D(5e-3, 5e-3, 0)
        )
    g3 = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(5, 5, 5), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    sc = ibsimu.MeshScalarField(g3)
    ibsimu.scharge_add_step_pic(sc, 1e-12, ibsimu.Vec3D(1.5e-3, 1.5e-3, 1.5e-3))
    ibsimu.scharge_finalize_step_pic(sc)
    assert sc.numpy().sum() * 1e-9 == pytest.approx(1e-12)


# -- plotting building blocks -------------------------------------------------------


def test_palette():
    pal = ibsimu.Palette([(ibsimu.Vec3D(0, 0, 1), 0.0), (ibsimu.Vec3D(1, 0, 0), 1.0)])
    assert tuple(pal(0.5)) == pytest.approx((0.5, 0.0, 0.5))
    pal.set_steps(4)
    assert pal.get_steps() == 4


def test_mesh_colormap():
    cm = ibsimu.MeshColormap([0.0, 0.0, 1.0, 1.0], 3, 2, [0, 1, 2, 3, 4, 5])
    assert cm.has_data()
    assert cm.get_value(0.0, 1.0) == pytest.approx(3.0)
    cm.clear_data()
    assert not cm.has_data()


def test_frame_xy_plot(tmp_path):
    x = np.linspace(0, 6, 30)
    g = ibsimu.XYGraph(list(x), list(np.sin(x)))
    g.set_color(ibsimu.Vec3D(1, 0, 0))
    g.set_point_style(ibsimu.XYGRAPH_POINT_CIRCLE)
    f = ibsimu.Frame()
    f.add_graph(
        ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, g, ibsimu.LegendEntry(g, "sin")
    )
    f.enable_legend(True)
    f.set_title("sine")
    f.set_axis_label(ibsimu.PLOT_AXIS_X1, "x")
    xmin, _ymin, xmax, _ymax = g.get_bbox()
    assert (xmin, xmax) == (0.0, 6.0)
    for ext in ("png", "pdf", "svg"):
        plot = getattr(f, f"plot_{ext}", None)
        if plot is None:
            continue
        fn = tmp_path / f"xy.{ext}"
        plot(str(fn), 400, 300)
        assert fn.stat().st_size > 0


def test_frame_field_plot(solved_2d, tmp_path):
    geom, epot = solved_2d
    f = ibsimu.Frame()
    fg = ibsimu.FieldGraph(geom, epot, ibsimu.FIELD_EPOT)
    fg.build_plot()
    assert fg.field_type() == ibsimu.FIELD_EPOT
    assert fg.get_value(35e-3, 4e-3) == pytest.approx(5.0)
    f.add_graph(ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, fg)
    f.add_graph(ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, ibsimu.SolidGraph(geom))
    eq = ibsimu.EqPotGraph(epot, geom)
    eq.set_eqlines_auto(5)
    f.add_graph(ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, eq)
    f.add_graph(ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, ibsimu.MeshGraph(geom))
    f.enable_colormap_legend(True)
    f.set_fixed_aspect(ibsimu.PLOT_FIXED_ASPECT_EXTEND_RANGE)
    fn = tmp_path / "field.png"
    f.plot_png(str(fn))
    assert fn.stat().st_size > 0


def test_geom3dplot(tmp_path):
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(11, 11, 11), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, y, z: x > 6e-3))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 1.0))
    geom.build_mesh()
    geom.build_surface()
    p = ibsimu.Geom3DPlot(geom)
    p.set_projection_zoom(2.0)
    assert p.get_projection_zoom() == 2.0
    p.set_clevel(0, 3)
    assert p.get_clevel(0) == 3
    if ibsimu.HAVE_GTK:
        fn = tmp_path / "geom3d.png"
        p.plot_png(str(fn), 200, 150)
        assert fn.stat().st_size > 0


# -- Simulation in 1D and external space charge ------------------------------------------


def test_simulation_1d(tmp_path):
    sim = ibsimu.Simulation(
        "1d", (11,), 1e-3, boundaries={1: 0.0, 2: 10.0}, scharge=1e-6, solver="mg"
    )
    out = sim.run(verbose=False)
    exact = 5.0 + 1e-6 / (2 * ibsimu.EPSILON0) * 5e-3 * 5e-3
    assert out.evaluate("epot").sel(x=5e-3).item() == pytest.approx(exact, rel=1e-4)
    assert "trajectories" not in out.keys()  # noqa: SIM118 (a method, not a dict)
    with pytest.raises(KeyError):
        out.evaluate("trajectories")
    out.info()
    out.save(str(tmp_path / "run"))
    again = ibsimu.SimulationOutput.load(str(tmp_path / "run"))
    assert again.pdb is None
    np.testing.assert_allclose(again.epot.numpy(), out.epot.numpy())


def test_simulation_1d_rejects_beams():
    with pytest.raises(ValueError, match="1D"):
        ibsimu.Simulation("1d", (11,), 1e-3, beams=[{"method": "2d_beam_with_energy"}])


def test_simulation_external_scharge_2d():
    sim = ibsimu.Simulation(
        "2d",
        (11, 11),
        1e-3,
        boundaries={1: 0.0, 2: 10.0},
        scharge=lambda x, y, z: 1e-6 if x > 5e-3 else 0.0,
        solver="bicgstab",
        solver_options={"preconditioner": ibsimu.ILU1_Precond(), "eps": 1e-9},
    )
    out = sim.run(verbose=False)
    rho = out.evaluate("scharge").values
    assert rho[:6].max() == 0.0 and rho[6:].min() == 1e-6
    with pytest.raises(ValueError):
        ibsimu.Simulation("2d", (11, 11), 1e-3, scharge=np.zeros((3, 3))).run(
            verbose=False
        )
