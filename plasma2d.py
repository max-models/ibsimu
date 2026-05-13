import ibsimu
import math


def solid1(x, y, z):
    return (
        x <= 2.0e-3
        and y >= 0.5e-3
        and y >= 2.0 * x - 1.0e-3
        and (x >= 0.5e-3 or y >= 1.5e-3)
    )


def solid2(x, y, z):
    return x >= 10.0e-3 and y >= 1.5e-3 and y >= 12.0e-3 - x


def test():
    # Geometry
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(76, 45, 1), ibsimu.Vec3D(0, 0, 0), 1.6e-4
    )

    s1 = ibsimu.FuncSolid(solid1)
    geom.set_solid(7, s1)
    s2 = ibsimu.FuncSolid(solid2)
    geom.set_solid(8, s2)

    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -8.0e3))
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(4, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -8.0e3))
    geom.build_mesh()

    # Solver
    solver = ibsimu.EpotGSSolver(geom)
    initp = ibsimu.InitialPlasma(ibsimu.AXIS_X, 0.0006)
    solver.set_initial_plasma(5.0, initp)

    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    bfield = ibsimu.MeshVectorField()
    efield = ibsimu.EpotEfield(epot)

    # efield.set_extrapolation([FIELD_EXTRAPOLATE, FIELD_EXTRAPOLATE, FIELD_SYMMETRIC_POTENTIAL, FIELD_EXTRAPOLATE, FIELD_EXTRAPOLATE, FIELD_EXTRAPOLATE])
    efield.set_extrapolation(
        [
            ibsimu.FIELD_EXTRAPOLATE,
            ibsimu.FIELD_EXTRAPOLATE,
            ibsimu.FIELD_SYMMETRIC_POTENTIAL,
            ibsimu.FIELD_EXTRAPOLATE,
            ibsimu.FIELD_EXTRAPOLATE,
            ibsimu.FIELD_EXTRAPOLATE,
        ]
    )

    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_mirror([False, False, True, False, False, False])
    pdb.set_polyint(True)

    conv = ibsimu.Convergence()
    conv.add_epot(epot)
    conv.add_scharge(scharge)
    emit = ibsimu.Emittance()
    conv.add_emittance(0, emit)

    for i in range(5):
        if i == 1:
            rhoe = pdb.get_rhosum()
            solver.set_pexp_plasma(-rhoe, 5.0, 5.0)

        solver.solve(epot, scharge)
        efield.recalculate()

        pdb.clear()
        pdb.add_2d_beam_with_energy(
            50000, 600.0, 1.0, 1.0, 5.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.5e-3
        )
        pdb.iterate_trajectories(scharge, efield, bfield)

        pp = ibsimu.ParticleDiagPlotter(
            geom,
            pdb,
            ibsimu.AXIS_X,
            11.90e-3,
            ibsimu.PARTICLE_DIAG_PLOT_SCATTER,
            ibsimu.DIAG_Y,
            ibsimu.DIAG_YP,
        )
        emit = pp.calculate_emittance()
        conv.evaluate_iteration()

    conv.print_history("plasma2d_conv_py.dat")

    pplotter1 = ibsimu.ParticleDiagPlotter(
        geom,
        pdb,
        ibsimu.AXIS_X,
        1e-6,
        ibsimu.PARTICLE_DIAG_PLOT_HISTO2D,
        ibsimu.DIAG_Y,
        ibsimu.DIAG_YP,
    )
    pplotter1.set_font_size(20)
    pplotter1.set_size(800, 600)
    pplotter1.plot_png("plasma2d_emit1_py.png")

    tdens = ibsimu.MeshScalarField(geom)
    pdb.build_trajectory_density_field(tdens)
    gplotter = ibsimu.GeomPlotter(geom)
    gplotter.set_size(800, 600)
    gplotter.set_font_size(20)
    gplotter.set_epot(epot)
    gplotter.set_eqlines_manual([-4.0, -2.0, 0.01, 2.0, 4.0])
    gplotter.set_particle_database(pdb)
    gplotter.set_particle_div(0)
    gplotter.set_trajdens(tdens)
    gplotter.set_fieldgraph_plot(ibsimu.FIELD_TRAJDENS)
    gplotter.fieldgraph().set_zscale(ibsimu.ZSCALE_RELLOG)
    gplotter.plot_png("plasma2d_py.png")


if __name__ == "__main__":
    test()
