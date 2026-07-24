import matplotlib.pyplot as plt
import numpy as np

import ibsimu


def run_solenoid():
    # 0. Setup
    ibsimu.ibsimu.set_thread_count(1)
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_CYL, ibsimu.Int3D(201, 31, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # Simple grounded pipe
    def pipe_func(x, r, z):
        return r > 25e-3

    s1 = ibsimu.FuncSolid(pipe_func)
    geom.set_solid(7, s1)

    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.build_mesh()

    # 1. Fields
    epot = ibsimu.EpotField(geom)  # All zero (grounded)
    efield = ibsimu.EpotEfield(epot)
    scharge = ibsimu.MeshScalarField(geom)

    # Define a solenoidal B-field using AxisymmetricVectorField
    # Bz(z) = B0 * exp(-(z-z0)^2 / w^2)
    z_points = np.linspace(0, 200e-3, 201)
    B0 = 0.5  # 0.5 Tesla peak
    z0 = 100e-3
    w = 30e-3
    Bz_data = B0 * np.exp(-((z_points - z0) ** 2) / w**2)

    bfield = ibsimu.AxisymmetricVectorField(ibsimu.MODE_CYL, 0.0, h, Bz_data.tolist())

    # 2. Particles
    pdb = ibsimu.ParticleDataBaseCyl(geom)
    pdb.set_save_trajectories(True)
    # 10keV Proton beam, slightly divergent
    pdb.add_2d_beam_with_energy(
        100,
        1.0e-3,
        1.0,
        1.0,
        10000.0,
        0.0,
        0.02,  # 20mrad divergence
        0.0,
        0.0,
        0.0,
        5e-3,
    )

    pdb.iterate_trajectories(scharge, efield, bfield)

    # 3. Plotting
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(z_points * 1000, Bz_data, "r-", label="Bz (Tesla)")
    plt.title("Magnetic Field Profile (Solenoid)")
    plt.ylabel("B [T]")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i in range(0, pdb.size(), 5):
        p = pdb.particle(i)
        tx, tr = [], []
        for j in range(p.traj_size()):
            tx.append(p.traj(j).x() * 1000)
            tr.append(p.traj(j).r() * 1000)
        plt.plot(tx, tr, "b-", alpha=0.5)
        plt.plot(tx, [-r for r in tr], "b-", alpha=0.5)

    plt.title("Beam Focusing in Solenoid")
    plt.xlabel("z [mm]")
    plt.ylabel("r [mm]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("examples_py/tutorial_solenoid.png")
    print("Saved examples_py/tutorial_solenoid.png")


if __name__ == "__main__":
    run_solenoid()
