import matplotlib.pyplot as plt
import numpy as np

import ibsimu


def run_vlasov2d():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry
    # 120mm (x) x 50mm (y) area, 0.5mm mesh
    h = 0.5e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(241, 101, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # --- Electrode Definitions from vlasov2d.cpp ---
    def solid1_func(x, y, z):
        return x <= 0.02 and y >= 0.018

    def solid2_func(x, y, z):
        return 0.03 <= x <= 0.04 and y >= 0.02

    def solid3_func(x, y, z):
        return x >= 0.06 and y >= 0.03 and y >= (0.07 - 0.5 * x)

    # Keep references to solids to avoid GC issues
    s1 = ibsimu.FuncSolid(solid1_func)
    geom.set_solid(7, s1)
    s2 = ibsimu.FuncSolid(solid2_func)
    geom.set_solid(8, s2)
    s3 = ibsimu.FuncSolid(solid3_func)
    geom.set_solid(9, s3)

    # Boundaries
    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -3000.0))
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -1000.0))
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(4, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))

    # Electrode Voltages
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -3000.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -14000.0))
    geom.set_boundary(9, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -1000.0))

    print("Building mesh...")
    geom.build_mesh()

    # 2. Solver & Fields
    solver = ibsimu.EpotBiCGSTABSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)

    # E-field extrapolation matching vlasov2d.cpp
    # { FIELD_EXTRAPOLATE, FIELD_EXTRAPOLATE, FIELD_SYMMETRIC_POTENTIAL, FIELD_EXTRAPOLATE, ... }
    ext = [ibsimu.FIELD_EXTRAPOLATE] * 6
    ext[2] = ibsimu.FIELD_SYMMETRIC_POTENTIAL
    efield.set_extrapolation(ext)

    bfield = ibsimu.MeshVectorField()

    # 3. Particle database
    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_save_trajectories(True)
    # Mirroring matching vlasov2d.cpp
    mirror = [False] * 6
    mirror[2] = True
    pdb.set_mirror(mirror)

    # 4. Vlasov Iteration Loop (5 iterations as per vlasov2d.cpp)
    print("Starting Vlasov iterations...")
    for i in range(5):
        solver.solve(epot, scharge)
        efield.recalculate()

        pdb.clear()
        # 1000 protons, 3keV, starting from x=0, y=0..12mm
        # Current density 50 A/m^2
        pdb.add_2d_beam_with_energy(
            1000, 50.0, 1.0, 1.0, 3000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012
        )
        pdb.iterate_trajectories(scharge, efield, bfield)
        print(f"  Iteration {i + 1} complete")

    # 5. Extraction at x = 115mm
    tdata_exit = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_Y, ibsimu.DIAG_YP])
    pdb.trajectories_at_plane(
        tdata_exit, ibsimu.AXIS_X, 115e-3, [ibsimu.DIAG_Y, ibsimu.DIAG_YP]
    )

    # 6. Plotting
    plt.figure(figsize=(12, 16))
    mesh_ext = [
        geom.origo(0) * 1000,
        geom.max(0) * 1000,
        geom.origo(1) * 1000,
        geom.max(1) * 1000,
    ]

    # 6.1 Potential
    plt.subplot(4, 1, 1)
    nx, ny = geom.size(0), geom.size(1)
    pot_map = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            pot_map[j, i] = epot.get3(i, j, 0)
    im1 = plt.imshow(pot_map, extent=mesh_ext, origin="lower", cmap="viridis")
    plt.colorbar(im1, label="Potential [V]")
    plt.title("Vlasov2D Tutorial: Electric Potential")
    plt.ylabel("y [mm]")

    # 6.2 E-field Magnitude
    plt.subplot(4, 1, 2)
    ef_map = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            ef_map[j, i] = efield(ibsimu.Vec3D(i * h, j * h, 0)).norm2()
    im2 = plt.imshow(ef_map, extent=mesh_ext, origin="lower", cmap="plasma")
    plt.colorbar(im2, label="|E| [V/m]")
    plt.title("Electric Field Magnitude")
    plt.ylabel("y [mm]")

    # 6.3 Trajectories
    plt.subplot(4, 1, 3)
    # Draw electrodes
    plt.gca().add_patch(
        plt.Rectangle((0, 18), 20, 32, color="gray", alpha=0.3, label="Electrodes")
    )
    plt.gca().add_patch(plt.Rectangle((30, 20), 10, 30, color="gray", alpha=0.3))
    points = [[60, 40], [120, 40], [120, 10], [60, 40]]
    plt.gca().add_patch(plt.Polygon(points, color="gray", alpha=0.3))

    # Plot trajectories (every 20th)
    for i in range(0, pdb.size(), 20):
        p = pdb.particle(i)
        tx, ty = [], []
        for j in range(p.traj_size()):
            pt = p.traj(j)
            tx.append(pt.x() * 1000)
            ty.append(pt.y() * 1000)
        plt.plot(tx, ty, color="blue", alpha=0.4, linewidth=0.5)

    plt.xlim(mesh_ext[0], mesh_ext[1])
    plt.ylim(mesh_ext[2], mesh_ext[3])
    plt.title("Beam Trajectories")
    plt.ylabel("y [mm]")
    plt.grid(True)

    # 6.4 Phase Space
    plt.subplot(4, 1, 4)
    y_exit = np.array(tdata_exit.column(0).data()) * 1000
    yp_exit = np.array(tdata_exit.column(1).data()) * 1000
    plt.scatter(y_exit, yp_exit, s=5, alpha=0.6, c="red")
    plt.xlabel("y [mm]")
    plt.ylabel("y' [mrad]")
    plt.title("Phase Space at Exit (x=115mm)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("examples_py/vlasov2d.png")
    print("Saved examples_py/vlasov2d.png")


if __name__ == "__main__":
    run_vlasov2d()
