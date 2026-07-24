import ibsimu
import matplotlib.pyplot as plt
import numpy as np


def run_simulation():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry
    # 200mm (x) x 40mm (y) area, 1mm mesh
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(201, 41, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # --- Electrodes Design ---
    # Stage 1: Anode (x=0 to 10mm)
    def anode(x, y, z):
        return x < 10e-3 and abs(y - 20e-3) > 5e-3

    # Stage 2: Intermediate Electrode (x=90 to 110mm)
    def intermediate(x, y, z):
        return 90e-3 < x < 110e-3 and abs(y - 20e-3) > 5e-3

    # Stage 3: Ground Electrode (x=190 to 200mm)
    def ground_el(x, y, z):
        return 190e-3 < x < 200e-3 and abs(y - 20e-3) > 5e-3

    s7 = ibsimu.FuncSolid(anode)
    geom.set_solid(7, s7)
    s8 = ibsimu.FuncSolid(intermediate)
    geom.set_solid(8, s8)
    s9 = ibsimu.FuncSolid(ground_el)
    geom.set_solid(9, s9)

    # Boundaries
    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))

    # Voltages
    V_anode = 50000.0  # 50 kV
    V_intermediate = 10000.0  # 10 kV
    V_ground = 0.0  # 0 V

    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, V_anode))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, V_intermediate))
    geom.set_boundary(9, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, V_ground))

    geom.build_mesh()

    # 2. Solver & Fields
    solver = ibsimu.EpotGSSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    bfield = ibsimu.MeshVectorField()

    # 3. Particle database
    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_save_trajectories(True)

    print("Solving for two-stage acceleration potential...")
    # Solve 5 iterations to handle potential space charge effects
    for iter_count in range(5):
        solver.solve(epot, scharge)
        efield.recalculate()

        pdb.clear()
        # Proton beam starting at x=1mm, y from 17mm to 23mm
        # Initial energy 10eV (representing thermal/source energy)
        pdb.add_2d_beam_with_energy(
            200, 1.0e-3, 1.0, 1.0, 10.0, 0.0, 0.0, 1e-3, 17e-3, 0.0, 23e-3
        )
        pdb.iterate_trajectories(scharge, efield, bfield)
        print(f"  Iteration {iter_count + 1} complete")

    # 4. Extraction
    print("Extracting results...")
    tdata_exit = ibsimu.TrajectoryDiagnosticData(
        [ibsimu.DIAG_Y, ibsimu.DIAG_YP, ibsimu.DIAG_EK]
    )
    pdb.trajectories_at_plane(
        tdata_exit,
        ibsimu.AXIS_X,
        195e-3,
        [ibsimu.DIAG_Y, ibsimu.DIAG_YP, ibsimu.DIAG_EK],
    )

    # 5. Plotting
    plt.figure(figsize=(12, 16))

    # 5.1 Potential Color Plot
    nx, ny = geom.size(0), geom.size(1)
    pot_map = np.zeros((ny, nx))
    ef_map = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            pot_map[j, i] = epot.get3(i, j, 0)
            ef_map[j, i] = efield(ibsimu.Vec3D(i * h, j * h, 0)).norm2()

    ext = [
        geom.origo(0) * 1000,
        geom.max(0) * 1000,
        geom.origo(1) * 1000,
        geom.max(1) * 1000,
    ]

    plt.subplot(4, 1, 1)
    im1 = plt.imshow(pot_map, extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im1, label="Potential [V]")
    plt.title("Two-Stage Accelerator: Electric Potential")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")

    # 5.2 E-field Magnitude
    plt.subplot(4, 1, 2)
    im2 = plt.imshow(ef_map, extent=ext, origin="lower", cmap="plasma")
    plt.colorbar(im2, label="|E| [V/m]")
    plt.title("Electric Field Magnitude (Acceleration Gaps)")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")

    # 5.3 Trajectories
    plt.subplot(4, 1, 3)
    # Draw electrodes (simple boxes for visualization)
    plt.axvspan(0, 10, color="gray", alpha=0.3, label="Electrodes")
    plt.axvspan(90, 110, color="gray", alpha=0.3)
    plt.axvspan(190, 200, color="gray", alpha=0.3)

    # Plot a subset of trajectories for clarity
    num_particles = pdb.size()
    step = max(1, num_particles // 50)
    for i in range(0, num_particles, step):
        p = pdb.particle(i)
        # Get trajectory points
        tx = []
        ty = []
        for j in range(p.traj_size()):
            pt = p.traj(j)
            tx.append(pt.x() * 1000)
            ty.append(pt.y() * 1000)
        plt.plot(tx, ty, color="blue", alpha=0.3, linewidth=0.5)

    plt.xlim(ext[0], ext[1])
    plt.ylim(ext[2], ext[3])
    plt.title("Particle Trajectories")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.grid(True)

    # 5.4 Beam Energy / Phase Space
    plt.subplot(4, 1, 4)
    y_exit = (np.array(tdata_exit.column(0).data()) - 20e-3) * 1000
    ek_exit = np.array(tdata_exit.column(2).data()) / 1000.0  # to keV

    plt.scatter(y_exit, ek_exit, alpha=0.6, s=15, c="blue")
    plt.xlabel("y - 20mm [mm]")
    plt.ylabel("Kinetic Energy [keV]")
    plt.title(f"Beam Energy at Exit (Mean: {np.mean(ek_exit):.2f} keV)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("examples_py/sim_two_stage_accel.png")
    print("Saved examples_py/sim_two_stage_accel.png")


if __name__ == "__main__":
    run_simulation()
