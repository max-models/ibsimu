import math

import ibsimu
import numpy as np
import pyvista as pv


def run_solenoid_3d_viz():
    # 0. Setup
    ibsimu.ibsimu.set_thread_count(1)
    h = 2e-3
    # 200mm x 50mm x 50mm
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(101, 26, 26), ibsimu.Vec3D(0, -25e-3, -25e-3), h
    )

    for i in range(1, 7):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.build_mesh()

    # 1. Fields
    epot = ibsimu.EpotField(geom)
    efield = ibsimu.EpotEfield(epot)
    scharge = ibsimu.MeshScalarField(geom)

    # Define a solenoidal B-field in 3D
    # Peak at x=100mm, width 40mm, B0=1.0T
    B0 = 1.0
    x0 = 100e-3
    w = 40e-3

    # We use a custom callback for a 3D magnetic field
    class SolenoidBField(ibsimu.CallbackFunctorD_3D):
        def __call__(self, x, y, z):
            # Approximate axial field: Bx(x) = B0 * exp(-(x-x0)^2/w^2)
            # Br is small near axis, we just do axial for demo
            return B0 * math.exp(-((x - x0) ** 2) / w**2)

    bx_cb = SolenoidBField()

    bfield = ibsimu.MeshVectorField(
        ibsimu.MODE_3D,
        [True, False, False],
        ibsimu.Int3D(101, 26, 26),
        ibsimu.Vec3D(0, -25e-3, -25e-3),
        h,
    )
    for k in range(26):
        for j in range(26):
            for i in range(101):
                x = i * h
                bfield.set3(i, j, k, ibsimu.Vec3D(bx_cb(x, 0, 0), 0, 0))

    # 2. Particles
    pdb = ibsimu.ParticleDataBase3D(geom)
    pdb.set_save_trajectories(True)
    # 10keV Proton beam, starting with some off-axis offset and angular momentum
    # to see the rotation
    for r in [5e-3, 10e-3, 15e-3]:
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            y = r * math.cos(angle)
            z = r * math.sin(angle)
            # ParticleP3D(t, x, vx, y, vy, z, vz)
            # Starting at x=0 with forward velocity 440km/s
            pdb.add_particle(
                1.0e-12,
                1.0,
                1.0,
                ibsimu.ParticleP3D(0.0, 0.0, 440000.0, y, 0.0, z, 0.0),
            )

    pdb.iterate_trajectories(scharge, efield, bfield)

    # 3. PyVista
    print("Opening interactive 3D Solenoid visualization...")
    trajectories = []
    for i in range(pdb.size()):
        p = pdb.particle(i)
        points = [
            [p.traj(j).x(), p.traj(j).y(), p.traj(j).z()] for j in range(p.traj_size())
        ]
        if len(points) > 1:
            trajectories.append(pv.MultipleLines(np.array(points)))

    plotter = pv.Plotter()
    for t in trajectories:
        plotter.add_mesh(t, color="cyan", line_width=2)

    # Add a tube representing the solenoid
    solenoid_coil = pv.Cylinder(
        center=(100e-3, 0, 0), direction=(1, 0, 0), radius=22e-3, height=60e-3
    )
    plotter.add_mesh(solenoid_coil, color="orange", opacity=0.3, label="Solenoid Coil")

    plotter.add_mesh(
        pv.Box(bounds=(0, 200e-3, -25e-3, 25e-3, -25e-3, 25e-3)).outline(),
        color="black",
    )

    plotter.view_isometric()
    plotter.add_text("3D Solenoid Beam Rotation", font_size=12)
    plotter.show()


if __name__ == "__main__":
    run_solenoid_3d_viz()
