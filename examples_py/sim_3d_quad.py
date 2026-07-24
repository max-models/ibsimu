import ibsimu
import matplotlib.pyplot as plt
import numpy as np


def run_simulation():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry
    # MODE_3D, 100mm x 20mm x 20mm area, 1mm mesh
    h = 1e-3
    # Origin at (0, -10mm, -10mm)
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(101, 21, 21), ibsimu.Vec3D(0, -10e-3, -10e-3), h
    )

    # Quadrupole electrodes
    def quad_y(x, y, z):
        return 40e-3 < x < 60e-3 and abs(y) > 8e-3 and abs(z) < 15e-3

    def quad_z(x, y, z):
        return 40e-3 < x < 60e-3 and abs(z) > 8e-3 and abs(y) < 15e-3

    s7 = ibsimu.FuncSolid(quad_y)
    geom.set_solid(7, s7)
    s8 = ibsimu.FuncSolid(quad_z)
    geom.set_solid(8, s8)

    # Boundaries
    for i in range(1, 7):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))

    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 1000.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -1000.0))

    geom.build_mesh()

    # 2. Solver & Fields
    solver = ibsimu.EpotGSSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    bfield = ibsimu.MeshVectorField()

    # 3. Particle database
    pdb = ibsimu.ParticleDataBase3D(geom)

    print("Solving for potential (3D Quadrupole)...")
    solver.solve(epot, scharge)
    efield.recalculate()

    pdb.clear()
    # 50keV Proton beam: N=1000, J=100A/m2, q=1, m=1, E=50keV
    # Rectangular beam at x=0, from -5mm to 5mm in y and z
    pdb.add_rectangular_beam_with_energy(
        1000,
        100.0,
        1.0,
        1.0,
        50000.0,
        0.0,
        0.0,
        ibsimu.Vec3D(0, 0, 0),
        ibsimu.Vec3D(0, 1, 0),
        ibsimu.Vec3D(0, 0, 1),
        5e-3,
        5e-3,
    )
    pdb.iterate_trajectories(scharge, efield, bfield)

    # 5. Extraction
    print("Extracting beam profile at exit (x = 95mm)...")
    tdata = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_Y, ibsimu.DIAG_Z])
    pdb.trajectories_at_plane(
        tdata, ibsimu.AXIS_X, 95e-3, [ibsimu.DIAG_Y, ibsimu.DIAG_Z]
    )

    y = np.array(tdata.column(0).data())
    z = np.array(tdata.column(1).data())

    plt.figure(figsize=(8, 8))
    plt.scatter(y * 1000, z * 1000, alpha=0.5, s=2)
    plt.xlabel("y [mm]")
    plt.ylabel("z [mm]")
    plt.title("3D Beam Cross-section at Exit (Quadrupole)")
    plt.axis("equal")
    plt.grid(True)

    plt.savefig("examples_py/beam_profile_3d.png")
    print("Saved examples_py/beam_profile_3d.png")


if __name__ == "__main__":
    run_simulation()
