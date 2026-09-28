"""The cairo based plotters write image files from Python."""

import ibsimu
import pytest


@pytest.fixture(scope="module")
def lens():
    ibsimu.ibsimu.set_thread_count(1)
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(61, 31, 1), ibsimu.Vec3D(0, 0, 0), h
    )
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, y, z: abs(y - 15e-3) > 10e-3))
    geom.set_solid(
        8, ibsimu.FuncSolid(lambda x, y, z: 25e-3 < x < 35e-3 and abs(y - 15e-3) > 6e-3)
    )
    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -3000.0))
    geom.build_mesh()

    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    ibsimu.EpotGSSolver(geom).solve(epot, scharge)
    efield = ibsimu.EpotEfield(epot)
    efield.recalculate()

    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_save_trajectories(True)
    pdb.add_2d_beam_with_energy(
        N=40,
        J=1e-3,
        q=1.0,
        m=1.0,
        E=5e3,
        Tp=0.0,
        Tt=0.0,
        x1=0.0,
        y1=9e-3,
        x2=0.0,
        y2=21e-3,
    )
    pdb.iterate_trajectories(scharge, efield, ibsimu.MeshVectorField())
    return geom, epot, scharge, efield, pdb


def test_geomplotter_png(lens, tmp_path):
    geom, epot, _, _, pdb = lens
    tdens = ibsimu.MeshScalarField(geom)
    pdb.build_trajectory_density_field(tdens)

    gp = ibsimu.GeomPlotter(geom)
    gp.set_size(600, 400)
    gp.set_font_size(12)
    gp.set_epot(epot)
    gp.set_eqlines_auto(10)
    gp.set_particle_database(pdb)
    gp.set_particle_div(2)
    gp.set_trajdens(tdens)
    gp.set_fieldgraph_plot(ibsimu.FIELD_TRAJDENS)
    gp.fieldgraph().set_zscale(ibsimu.ZSCALE_RELLOG)
    gp.fieldgraph().set_interpolation(ibsimu.INTERPOLATION_BILINEAR)
    gp.enable_colormap_legend(True)
    gp.set_view(ibsimu.VIEW_XY)
    out = tmp_path / "geom.png"
    gp.plot_png(str(out))
    assert out.stat().st_size > 1000


@pytest.mark.parametrize(
    "plot_type",
    [
        ibsimu.PARTICLE_DIAG_PLOT_SCATTER,
        ibsimu.PARTICLE_DIAG_PLOT_HISTO1D,
        ibsimu.PARTICLE_DIAG_PLOT_HISTO2D,
    ],
)
def test_particle_diag_plotter(lens, tmp_path, plot_type):
    geom, _, _, _, pdb = lens
    pp = ibsimu.ParticleDiagPlotter(
        geom, pdb, ibsimu.AXIS_X, 55e-3, plot_type, ibsimu.DIAG_Y, ibsimu.DIAG_YP
    )
    pp.set_size(400, 300)
    pp.set_histogram_n(32)
    pp.set_histogram_m(32)
    pp.set_emittance_ellipse(True)
    out = tmp_path / "diag.png"
    pp.plot_png(str(out))
    assert out.stat().st_size > 1000
    assert pp.get_isum() > 0.0
    if plot_type == ibsimu.PARTICLE_DIAG_PLOT_HISTO1D:
        with pytest.raises(ibsimu.IBSimuError, match="not emittance"):
            pp.calculate_emittance()
    else:
        assert pp.calculate_emittance().epsilon() >= 0.0


def test_field_diag_plotter(lens, tmp_path):
    geom, epot, _, efield, _ = lens
    fp = ibsimu.FieldDiagPlotter(geom)
    fp.set_epot(epot)
    fp.set_efield(efield)
    fp.set_coordinates(100, ibsimu.Vec3D(0, 15e-3, 0), ibsimu.Vec3D(60e-3, 15e-3, 0))
    fp.set_diagnostic(
        [ibsimu.FIELD_EPOT, ibsimu.FIELD_EFIELD_X],
        [ibsimu.FIELDD_LOC_X, ibsimu.FIELDD_LOC_NONE],
    )
    out = tmp_path / "field.png"
    fp.plot_png(str(out))
    assert out.stat().st_size > 1000
    data = tmp_path / "field.dat"
    fp.export_data(str(data))
    assert data.stat().st_size > 0
