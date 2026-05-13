
import matplotlib.pyplot as plt
import numpy as np

import ibsimu


def run_simulation():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry
    # MODE_2D, 100mm x 40mm area, 1mm mesh
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(101, 41, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # Solid 7: Left pipe (grounded)
    def left_pipe(x, y, z):
        return x < 40e-3 and abs(y - 20e-3) > 10e-3

    # Solid 8: Middle electrode (negative potential)
    def middle_electrode(x, y, z):
        return 45e-3 < x < 55e-3 and abs(y - 20e-3) > 5e-3

    # Solid 9: Right pipe (grounded)
    def right_pipe(x, y, z):
        return 60e-3 < x < 100e-3 and abs(y - 20e-3) > 10e-3

    s7 = ibsimu.FuncSolid(left_pipe)
    geom.set_solid(7, s7)
    s8 = ibsimu.FuncSolid(middle_electrode)
    geom.set_solid(8, s8)
    s9 = ibsimu.FuncSolid(right_pipe)
    geom.set_solid(9, s9)

    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # xmin
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # xmax
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # ymin
    geom.set_boundary(4, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # ymax

    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -5000.0))
    geom.set_boundary(9, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))

    geom.build_mesh()

    # 2. Solver & Fields
    solver = ibsimu.EpotGSSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    bfield = ibsimu.MeshVectorField()

    # 4. Iteration
    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_save_trajectories(True)

    print("Solving for potential...")
    for i in range(2):
        solver.solve(epot, scharge)
        efield.recalculate()

        pdb.clear()
        # 10keV Proton beam: N=200, J=1mA/m, q=1, m=1, E=10keV
        # Starting at x=0, y from 15mm to 25mm
        pdb.add_2d_beam_with_energy(
            200, 1.0e-3, 1.0, 1.0, 10000.0, 0.0, 0.0, 0.0, 15e-3, 0.0, 25e-3
        )
        pdb.iterate_trajectories(scharge, efield, bfield)

    # 5. Extraction
    print("Extracting beam profile at exit (x = 95mm)...")
    tdata = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_Y, ibsimu.DIAG_YP])
    pdb.trajectories_at_plane(
        tdata, ibsimu.AXIS_X, 95e-3, [ibsimu.DIAG_Y, ibsimu.DIAG_YP]
    )

    y = np.array(tdata.column(0).data())
    yp = np.array(tdata.column(1).data())

    # Initial distribution
    tdata0 = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_Y, ibsimu.DIAG_YP])
    pdb.trajectories_at_plane(
        tdata0, ibsimu.AXIS_X, 0.0, [ibsimu.DIAG_Y, ibsimu.DIAG_YP]
    )
    y0 = np.array(tdata0.column(0).data())
    yp0 = np.array(tdata0.column(1).data())

    # 6. Plotting
    plt.figure(figsize=(10, 12))

    # 6.1 Phase space
    plt.subplot(3, 1, 1)
    plt.scatter((y0 - 20e-3) * 1000, yp0 * 1000, label="Initial (x=0)", alpha=0.5, s=10)
    plt.scatter((y - 20e-3) * 1000, yp * 1000, label="Exit (x=95mm)", alpha=0.5, s=10)
    plt.xlabel("y - 20mm [mm]")
    plt.ylabel("y' [mrad]")
    plt.title("2D Beam Phase Space (Gap Lens)")
    plt.legend()
    plt.grid(True)

    # 6.2 Potential color plot
    print("Extracting potential and field for plotting...")
    nx, ny = geom.size(0), geom.size(1)
    pot_map = np.zeros((ny, nx))
    ef_map = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            pot_map[j, i] = epot.get3(i, j, 0)
            # Sample E-field at nodes
            ef_map[j, i] = efield(ibsimu.Vec3D(i * h, j * h, 0)).norm2()

    plt.subplot(3, 1, 2)
    ext = [
        geom.origo(0) * 1000,
        geom.max(0) * 1000,
        geom.origo(1) * 1000,
        geom.max(1) * 1000,
    ]
    im1 = plt.imshow(pot_map, extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im1, label="Potential [V]")
    plt.title("Electric Potential")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")

    # 6.3 E-field magnitude color plot
    plt.subplot(3, 1, 3)
    im2 = plt.imshow(ef_map, extent=ext, origin="lower", cmap="plasma")
    plt.colorbar(im2, label="|E| [V/m]")
    plt.title("Electric Field Magnitude")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")

    plt.tight_layout()
    plt.savefig("examples_py/sim_2d_lens.png")
    print("Saved examples_py/sim_2d_lens.png")


if __name__ == "__main__":
    run_simulation()
