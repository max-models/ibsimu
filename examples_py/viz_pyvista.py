import numpy as np
import pyvista as pv

import ibsimu


def run_3d_quad_viz():
    # 0. Global setup
    ibsimu.ibsimu.set_thread_count(1)

    # 1. Geometry (3D)
    # 100mm (x) x 30mm (y) x 30mm (z), 1mm mesh
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(101, 31, 31), ibsimu.Vec3D(0, -15e-3, -15e-3), h
    )

    # Quadrupole electrodes (4 hyperbolic-like rods or just boxes for demo)
    def quad_y_pos(x, y, z):
        return 20e-3 < x < 80e-3 and y > 10e-3 and abs(z) < 15e-3

    def quad_y_neg(x, y, z):
        return 20e-3 < x < 80e-3 and y < -10e-3 and abs(z) < 15e-3

    def quad_z_pos(x, y, z):
        return 20e-3 < x < 80e-3 and z > 10e-3 and abs(y) < 15e-3

    def quad_z_neg(x, y, z):
        return 20e-3 < x < 80e-3 and z < -10e-3 and abs(y) < 15e-3

    s1 = ibsimu.FuncSolid(quad_y_pos)
    geom.set_solid(7, s1)
    s2 = ibsimu.FuncSolid(quad_y_neg)
    geom.set_solid(8, s2)
    s3 = ibsimu.FuncSolid(quad_z_pos)
    geom.set_solid(9, s3)
    s4 = ibsimu.FuncSolid(quad_z_neg)
    geom.set_solid(10, s4)

    for i in range(1, 7):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 5000.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 5000.0))
    geom.set_boundary(9, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -5000.0))
    geom.set_boundary(10, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, -5000.0))

    geom.build_mesh()

    # 2. Solver
    solver = ibsimu.EpotGSSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    bfield = ibsimu.MeshVectorField()

    print("Solving 3D potential...")
    solver.solve(epot, scharge)
    efield.recalculate()

    # 3. Particles
    pdb = ibsimu.ParticleDataBase3D(geom)
    pdb.set_save_trajectories(True)
    # 50keV Proton beam
    pdb.add_rectangular_beam_with_energy(
        200,
        10.0,
        1.0,
        1.0,
        50000.0,
        0.0,
        0.0,
        ibsimu.Vec3D(0, 0, 0),
        ibsimu.Vec3D(0, 1, 0),
        ibsimu.Vec3D(0, 0, 1),
        8e-3,
        8e-3,
    )
    pdb.iterate_trajectories(scharge, efield, bfield)

    # 4. PyVista Visualization
    print("Preparing PyVista visualization...")

    # 4.1 Field Data
    nx, ny, nz = geom.size(0), geom.size(1), geom.size(2)
    # Build a 3D numpy array for potential
    # Note: MeshScalarField is (x, y, z)
    pot_data = np.zeros((nx, ny, nz))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                pot_data[i, j, k] = epot.get3(i, j, k)

    # Create PyVista grid
    grid = pv.ImageData(
        dimensions=(nx, ny, nz), spacing=(h, h, h), origin=tuple(geom.origo_vec())
    )
    # PyVista/VTK uses Fortran-style ordering forImageData (x varies fastest)
    grid.point_data["Potential"] = pot_data.flatten(order="F")

    # 4.2 Trajectory Data
    trajectories = []
    for i in range(pdb.size()):
        p = pdb.particle(i)
        points = []
        for j in range(p.traj_size()):
            pt = p.traj(j)
            points.append([pt.x(), pt.y(), pt.z()])
        if points:
            trajectories.append(pv.MultipleLines(np.array(points)))

    # 5. Plotting
    plotter = pv.Plotter()

    # Volume or slices for potential
    slices = grid.slice_orthogonal(x=50e-3, y=0, z=0)
    plotter.add_mesh(
        slices, cmap="viridis", opacity=0.7, scalar_bar_args={"title": "Potential [V]"}
    )

    # Add solids (using contour of field data if possible, or just geometry)
    # For now, let's just add the trajectories
    for t in trajectories:
        plotter.add_mesh(t, color="white", line_width=1)

    # Add a bounding box
    plotter.add_mesh(grid.outline(), color="black")

    plotter.view_isometric()
    plotter.add_text("IBSimu 3D Quadrupole Visualization", font_size=12)

    print("Opening interactive 3D window...")
    plotter.show()


if __name__ == "__main__":
    run_3d_quad_viz()
