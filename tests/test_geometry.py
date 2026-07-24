"""Geometry, solid and solver behaviour of the Python bindings."""

import gc

import ibsimu


def cyl_lens_geometry(v_lens, mesh_r=41):
    """Grounded pipe at r = 15 mm with a biased ring electrode inside it."""
    h = 0.5e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_CYL, ibsimu.Int3D(201, mesh_r, 1), ibsimu.Vec3D(0, 0, 0), h
    )
    # Passed as temporaries on purpose: the geometry must take ownership.
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, r, z: r > 15e-3))
    geom.set_solid(8, ibsimu.FuncSolid(lambda x, r, z: 45e-3 < x < 55e-3 and r > 10e-3))
    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, v_lens))
    gc.collect()  # drop anything the geometry failed to keep alive
    geom.build_mesh()
    return geom


def solve(geom):
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    ibsimu.EpotGSSolver(geom).solve(epot, scharge)
    return epot


def test_solid_survives_as_temporary():
    """set_solid must keep the solid (and its Python callable) alive."""
    geom = cyl_lens_geometry(-10000.0)
    assert geom.number_of_solids() == 2
    assert geom.inside(ibsimu.Vec3D(0.05, 0.012, 0.0)) == 8
    assert geom.inside(ibsimu.Vec3D(0.02, 0.02, 0.0)) == 7
    assert geom.inside(ibsimu.Vec3D(0.02, 0.005, 0.0)) == 0


def test_geometry_outlives_python_solid_references():
    geom = cyl_lens_geometry(-10000.0)
    gc.collect()
    epot = solve(geom)
    # The callable is still needed here: solving touches the solid data.
    assert epot(ibsimu.Vec3D(0.05, 0.012, 0.0)) == -10000.0


def test_biased_electrode_creates_potential_well():
    """Regression: a solid outside the mesh leaves the geometry ungrounded.

    With the pipe wall inside the mesh the on-axis potential must be pulled
    down near the electrode and return towards ground at both ends.
    """
    epot = solve(cyl_lens_geometry(-10000.0))
    phi = [epot(ibsimu.Vec3D(x, 0.0, 0.0)) for x in (0.0, 0.02, 0.05, 0.08, 0.1)]
    assert phi[2] < -1000.0  # deep well at the electrode
    assert phi[0] > -500.0 and phi[-1] > -500.0  # grounded ends
    assert phi[0] > phi[1] > phi[2] < phi[3] < phi[4]


def test_solid_missing_from_mesh_leaves_no_ground():
    """The failure mode itself: r > 15 mm never hits a node of a 15 mm mesh."""
    epot = solve(cyl_lens_geometry(-10000.0, mesh_r=31))
    phi = [epot(ibsimu.Vec3D(x, 0.0, 0.0)) for x in (0.0, 0.05, 0.1)]
    assert all(p < -9000.0 for p in phi)


def test_lens_voltage_changes_beam_size():
    radii = []
    for v_lens in (0.0, -10000.0):
        geom = cyl_lens_geometry(v_lens)
        epot = ibsimu.EpotField(geom)
        scharge = ibsimu.MeshScalarField(geom)
        ibsimu.EpotGSSolver(geom).solve(epot, scharge)
        efield = ibsimu.EpotEfield(epot)
        efield.recalculate()

        pdb = ibsimu.ParticleDataBaseCyl(geom)
        pdb.add_2d_beam_with_energy(
            50, 1.0e-3, 1.0, 1.0, 20000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8e-3
        )
        pdb.iterate_trajectories(scharge, efield, ibsimu.MeshVectorField())

        tdata = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_R])
        pdb.trajectories_at_plane(tdata, ibsimu.AXIS_X, 95e-3, [ibsimu.DIAG_R])
        r_data = tdata.column(0).data()
        assert len(r_data) == 50
        radii.append(max(r_data))

    assert radii[1] < radii[0]
