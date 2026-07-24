import matplotlib.pyplot as plt

import ibsimu


def run_sector_magnet():
    # 0. Setup
    ibsimu.ibsimu.set_thread_count(1)
    h = 2e-3
    # 200mm x 200mm area
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(101, 101, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.build_mesh()

    # 1. Fields
    epot = ibsimu.EpotField(geom)
    efield = ibsimu.EpotEfield(epot)
    scharge = ibsimu.MeshScalarField(geom)

    # Define a dipole B-field (Bz component in 2D simulation)
    # Field region: 50mm < x < 100mm, all y
    bfield = ibsimu.MeshVectorField(
        ibsimu.MODE_2D,
        [False, False, True],
        ibsimu.Int3D(101, 101, 1),
        ibsimu.Vec3D(0, 0, 0),
        h,
    )
    B0 = 0.2  # 0.2 Tesla
    for j in range(101):
        for i in range(101):
            x = i * h
            if 50e-3 < x < 100e-3:
                bfield.set3(i, j, 0, ibsimu.Vec3D(0, 0, B0))  # Set Bz

    # 2. Particles
    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.set_save_trajectories(True)
    # 50keV Proton beam, narrow
    pdb.add_2d_beam_with_energy(
        50, 1.0e-3, 1.0, 1.0, 50000.0, 0.0, 0.0, 0.0, 95e-3, 0.0, 105e-3
    )

    pdb.iterate_trajectories(scharge, efield, bfield)

    # 3. Plotting
    plt.figure(figsize=(10, 10))

    # Plot B-field region
    plt.axvspan(50, 100, color="yellow", alpha=0.2, label="Dipole Field (0.2T)")

    # Plot trajectories
    for i in range(pdb.size()):
        p = pdb.particle(i)
        tx, ty = [], []
        for j in range(p.traj_size()):
            tx.append(p.traj(j).x() * 1000)
            ty.append(p.traj(j).y() * 1000)
        plt.plot(tx, ty, "b-", alpha=0.3)

    plt.title("Beam Deflection in a Sector Dipole Magnet")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.savefig("examples_py/tutorial_sector_magnet.png")
    print("Saved examples_py/tutorial_sector_magnet.png")


if __name__ == "__main__":
    run_sector_magnet()
