import matplotlib.pyplot as plt
import numpy as np

import ibsimu


def run_simulation():
    # 1. Geometry
    # MODE_CYL, 100mm x 20mm area, 0.5mm mesh
    h = 0.5e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_CYL, ibsimu.Int3D(201, 41, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # Solid 7: Grounded pipe
    def pipe(x, r, z):
        return r > 15e-3

    # Solid 8: Lens electrode
    def lens(x, r, z):
        return 40e-3 < x < 60e-3 and r > 10e-3

    s7 = ibsimu.FuncSolid(pipe)
    geom.set_solid(7, s7)
    s8 = ibsimu.FuncSolid(lens)
    geom.set_solid(8, s8)

    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # xmin
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # xmax
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # rmin (axis)
    geom.set_boundary(4, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))  # rmax

    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -15000.0))

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

    print("Solving for potential (cylindrical)...")
    for i in range(2):
        solver.solve(epot, scharge)
        efield.recalculate()

        pdb.clear()
        # 30keV H+ beam: N=500, J=100A/m2, q=1, m=1, E=30keV
        # Starting at x=0, r from 0 to 8mm
        pdb.add_2d_beam_with_energy(
            500, 100.0, 1.0, 1.0, 30000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0e-3
        )
        pdb.iterate_trajectories(scharge, efield, bfield)

    # 5. Extraction
    print("Extracting beam profile at exit (x = 95mm)...")
    tdata = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_R, ibsimu.DIAG_RP])
    pdb.trajectories_at_plane(
        tdata, ibsimu.AXIS_X, 95e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP]
    )

    r = np.array(tdata.column(0).data())
    rp = np.array(tdata.column(1).data())

    # 6. Plotting
    plt.figure(figsize=(10, 12))

    # 6.1 Phase space
    plt.subplot(3, 1, 1)
    plt.scatter(r * 1000, rp * 1000, alpha=0.5, s=5)
    plt.xlabel("r [mm]")
    plt.ylabel("r' [mrad]")
    plt.title("Cylindrical Beam Phase Space (r, r')")
    plt.grid(True)

    # 6.2 Potential color plot
    print("Extracting potential and field for plotting...")
    nx, nr = geom.size(0), geom.size(1)
    pot_map = np.zeros((nr, nx))
    ef_map = np.zeros((nr, nx))
    for j in range(nr):
        for i in range(nx):
            pot_map[j, i] = epot.get3(i, j, 0)
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
    plt.title("Electric Potential (Cylindrical)")
    plt.xlabel("x [mm]")
    plt.ylabel("r [mm]")

    # 6.3 E-field magnitude color plot
    plt.subplot(3, 1, 3)
    im2 = plt.imshow(ef_map, extent=ext, origin="lower", cmap="plasma")
    plt.colorbar(im2, label="|E| [V/m]")
    plt.title("Electric Field Magnitude")
    plt.xlabel("x [mm]")
    plt.ylabel("r [mm]")

    plt.tight_layout()
    plt.savefig("examples_py/sim_cyl_lens.png")
    print("Saved examples_py/sim_cyl_lens.png")


if __name__ == "__main__":
    run_simulation()
