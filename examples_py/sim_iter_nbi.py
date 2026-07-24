import math

import matplotlib.pyplot as plt
import numpy as np

import ibsimu


def run_simulation():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry
    # 200mm (x) x 10mm (r) area, 0.5mm mesh
    h = 0.5e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_CYL, ibsimu.Int3D(401, 21, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # --- ITER-like Electrode Design ---
    # We define 7 electrodes (6 gaps)
    # Each electrode has a thickness of 6mm and a chamfered aperture

    electrode_x = [5e-3, 20e-3, 50e-3, 80e-3, 110e-3, 140e-3, 170e-3]
    thickness = 6e-3
    r_aperture = 7e-3
    chamfer_angle = math.radians(30)

    def make_electrode_func(x_center):
        def electrode_func(x, r, z):
            # Distance from center of electrode
            dx = x - x_center
            if abs(dx) > thickness / 2.0:
                return False
            # Aperture with chamfer (narrowest at the upstream side, wider at downstream)
            # Let's say it's narrowest at x_center - thickness/2
            local_x = dx + thickness / 2.0  # 0 to thickness
            r_limit = r_aperture + local_x * math.tan(chamfer_angle)
            return r > r_limit

        return electrode_func

    for i, x_c in enumerate(electrode_x):
        geom.set_solid(7 + i, ibsimu.FuncSolid(make_electrode_func(x_c)))

    # Boundaries
    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    # Boundary 3 is axis (r=0)
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))

    # Voltages (120kV total acceleration)
    # G0: 120kV, G1: 110kV (Extraction), G2..G6: Acceleration down to 0V
    voltages = [120000.0, 110000.0, 85000.0, 65000.0, 45000.0, 25000.0, 0.0]
    for i, V in enumerate(voltages):
        geom.set_boundary(7 + i, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, V))

    print("Building mesh for ITER-like NBI system...")
    geom.build_mesh()

    # 2. Solver & Fields
    solver = ibsimu.EpotGSSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    bfield = ibsimu.MeshVectorField()

    # 3. Particle database
    pdb = ibsimu.ParticleDataBaseCyl(geom)
    pdb.set_save_trajectories(True)

    print("Solving potential...")
    # Single iteration for simplicity in this design demo
    solver.solve(epot, scharge)
    efield.recalculate()

    # 4. Particles
    # 500 protons, starting at x=1mm, r=0..6.5mm
    pdb.clear()
    pdb.add_2d_beam_with_energy(
        500, 100.0, 1.0, 1.0, 10.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 6.5e-3
    )

    print("Iterating trajectories...")
    pdb.iterate_trajectories(scharge, efield, bfield)

    # 5. extraction
    tdata_exit = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_R, ibsimu.DIAG_RP])
    pdb.trajectories_at_plane(
        tdata_exit, ibsimu.AXIS_X, 195e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP]
    )

    # 6. Plotting
    plt.figure(figsize=(12, 16))
    ext = [
        geom.origo(0) * 1000,
        geom.max(0) * 1000,
        geom.origo(1) * 1000,
        geom.max(1) * 1000,
    ]

    # 6.1 Potential
    plt.subplot(4, 1, 1)
    nx, nr = geom.size(0), geom.size(1)
    pot_map = np.zeros((nr, nx))
    for j in range(nr):
        for i in range(nx):
            pot_map[j, i] = epot.get3(i, j, 0)
    im1 = plt.imshow(pot_map, extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im1, label="Potential [V]")
    plt.title("ITER NBI Aperture: Electric Potential")
    plt.ylabel("r [mm]")

    # 6.2 E-field
    plt.subplot(4, 1, 2)
    ef_map = np.zeros((nr, nx))
    for j in range(nr):
        for i in range(nx):
            ef_map[j, i] = efield(ibsimu.Vec3D(i * h, j * h, 0)).norm2()
    im2 = plt.imshow(ef_map, extent=ext, origin="lower", cmap="plasma")
    plt.colorbar(im2, label="|E| [V/m]")
    plt.title("Electric Field Magnitude")
    plt.ylabel("r [mm]")

    # 6.3 Trajectories
    plt.subplot(4, 1, 3)
    # Draw electrodes
    for xc in electrode_x:
        plt.gca().add_patch(
            plt.Rectangle(
                (xc * 1000 - thickness * 500, r_aperture * 1000),
                thickness * 1000,
                10,
                color="gray",
                alpha=0.4,
            )
        )

    for i in range(0, pdb.size(), 10):
        p = pdb.particle(i)
        tx, tr = [], []
        for j in range(p.traj_size()):
            pt = p.traj(j)
            tx.append(pt.x() * 1000)
            tr.append(pt.r() * 1000)
        plt.plot(tx, tr, color="blue", alpha=0.4, linewidth=0.5)
        # Mirror for visualization
        plt.plot(tx, [-r for r in tr], color="blue", alpha=0.4, linewidth=0.5)

    plt.xlim(ext[0], ext[1])
    plt.ylim(-10, 10)
    plt.title("Beam Trajectories (Cylindrical Symmetry)")
    plt.ylabel("r [mm]")
    plt.grid(True)

    # 6.4 Phase Space at Exit
    plt.subplot(4, 1, 4)
    r_exit = np.array(tdata_exit.column(0).data()) * 1000
    rp_exit = np.array(tdata_exit.column(1).data()) * 1000
    plt.scatter(r_exit, rp_exit, s=5, alpha=0.6)
    plt.scatter(-r_exit, -rp_exit, s=5, alpha=0.6)  # symmetric points
    plt.xlabel("r [mm]")
    plt.ylabel("r' [mrad]")
    plt.title("Phase Space at x=195mm")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("examples_py/sim_iter_nbi.png")
    print("Saved examples_py/sim_iter_nbi.png")


if __name__ == "__main__":
    run_simulation()
