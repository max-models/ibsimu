import matplotlib.pyplot as plt
import numpy as np

import ibsimu


def run_plasmacyl():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry
    # 12mm (x) x 7mm (r) area, 0.05mm mesh
    # Mesh: 241 x 141 nodes
    h = 5e-5
    geom = ibsimu.Geometry(
        ibsimu.MODE_CYL, ibsimu.Int3D(241, 141, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # --- Electrode Definitions from plasmacyl.cpp ---
    def solid1_func(x, r, z):
        return x <= 1.87e-3 and r >= 0.54e-3 and r >= (2.28 * x - 1.0e-3)

    def solid2_func(x, r, z):
        return x >= 9.5e-3 and r >= 2.3333e-3 and r >= (12.8333e-3 - x)

    geom.set_solid(7, ibsimu.FuncSolid(solid1_func))
    geom.set_solid(8, ibsimu.FuncSolid(solid2_func))

    # Boundaries
    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -12000.0))
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(4, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))

    # Electrode Voltages
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -12000.0))

    print("Building mesh...")
    geom.build_mesh()

    # 2. Solver & Fields
    solver = ibsimu.EpotBiCGSTABSolver(geom)

    # Initial plasma guess: 5V, x < 0.55mm
    initp = ibsimu.InitialPlasma(ibsimu.AXIS_X, 0.00055)
    solver.set_initial_plasma(5.0, initp)

    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)

    # E-field extrapolation matching plasmacyl.cpp
    ext = [ibsimu.FIELD_EXTRAPOLATE] * 6
    ext[2] = ibsimu.FIELD_SYMMETRIC_POTENTIAL
    efield.set_extrapolation(ext)

    bfield = ibsimu.MeshVectorField()

    # 3. Particle database
    pdb = ibsimu.ParticleDataBaseCyl(geom)
    pdb.set_save_trajectories(True)
    # Mirroring matching plasmacyl.cpp
    mirror = [False] * 6
    mirror[2] = True
    pdb.set_mirror(mirror)
    pdb.set_polyint(True)

    # 4. Vlasov Iteration Loop (15 iterations as per plasmacyl.cpp)
    print("Starting Vlasov iterations...")
    for i in range(15):
        if i == 1:
            rhoe = pdb.get_rhosum()
            # Positive exponential plasma model
            solver.set_pexp_plasma(-rhoe, 5.0, 5.0)

        solver.solve(epot, scharge)
        efield.recalculate()

        pdb.clear()
        # 15000 Helium ions (He+), 5eV
        # add_2d_beam_with_energy(n, J, q, m, E, x0, v0, x1, r0, x2, r1)
        pdb.add_2d_beam_with_energy(
            15000, 600.0, 1.0, 4.0, 5.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.5e-3
        )
        pdb.iterate_trajectories(scharge, efield, bfield)
        print(f"  Iteration {i + 1} complete")

    # 5. extraction at x = 11.9mm
    tdata_exit = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_R, ibsimu.DIAG_RP])
    pdb.trajectories_at_plane(
        tdata_exit, ibsimu.AXIS_X, 11.9e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP]
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
    nx, nr = geom.size(0), geom.size(1)
    pot_map = np.zeros((nr, nx))
    for j in range(nr):
        for i in range(nx):
            pot_map[j, i] = epot.get3(i, j, 0)
    im1 = plt.imshow(pot_map, extent=mesh_ext, origin="lower", cmap="viridis")
    plt.colorbar(im1, label="Potential [V]")
    plt.title("PlasmaCyl Tutorial: Electric Potential")
    plt.ylabel("r [mm]")

    # 6.2 E-field Magnitude
    plt.subplot(4, 1, 2)
    ef_map = np.zeros((nr, nx))
    for j in range(nr):
        for i in range(nx):
            ef_map[j, i] = efield(ibsimu.Vec3D(i * h, j * h, 0)).norm2()
    im2 = plt.imshow(ef_map, extent=mesh_ext, origin="lower", cmap="plasma")
    plt.colorbar(im2, label="|E| [V/m]")
    plt.title("Electric Field Magnitude")
    plt.ylabel("r [mm]")

    # 6.3 Trajectories
    plt.subplot(4, 1, 3)
    # Draw electrodes (approximate boxes)
    plt.gca().add_patch(
        plt.Rectangle(
            (0, 0.54), 1.87, 6.46, color="gray", alpha=0.3, label="Electrodes"
        )
    )
    plt.gca().add_patch(
        plt.Rectangle((9.5, 2.3333), 2.5, 4.6667, color="gray", alpha=0.3)
    )

    # Plot trajectories (every 100th)
    for i in range(0, pdb.size(), 100):
        p = pdb.particle(i)
        tx, tr = [], []
        for j in range(p.traj_size()):
            pt = p.traj(j)
            tx.append(pt.x() * 1000)
            tr.append(pt.r() * 1000)
        plt.plot(tx, tr, color="blue", alpha=0.4, linewidth=0.5)

    plt.xlim(mesh_ext[0], mesh_ext[1])
    plt.ylim(mesh_ext[2], mesh_ext[3])
    plt.title("Beam Trajectories (Cylindrical)")
    plt.ylabel("r [mm]")
    plt.grid(True)

    # 6.4 Phase Space
    plt.subplot(4, 1, 4)
    r_exit = np.array(tdata_exit.column(0).data()) * 1000
    rp_exit = np.array(tdata_exit.column(1).data()) * 1000
    plt.scatter(r_exit, rp_exit, s=2, alpha=0.4, c="red")
    plt.xlabel("r [mm]")
    plt.ylabel("r' [mrad]")
    plt.title("Phase Space at Exit (x=11.9mm)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("examples_py/plasmacyl.png")
    print("Saved examples_py/plasmacyl.png")


if __name__ == "__main__":
    run_plasmacyl()
