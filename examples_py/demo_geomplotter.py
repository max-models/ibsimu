import ibsimu


def run_geomplotter_demo():
    # 0. Setup
    ibsimu.ibsimu.set_thread_count(1)
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(101, 41, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # Define some electrodes
    def pipe(x, y, z):
        return abs(y - 20e-3) > 10e-3

    s1 = ibsimu.FuncSolid(pipe)
    geom.set_solid(7, s1)

    geom.set_boundary(1, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(2, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(3, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(4, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))

    geom.build_mesh()

    epot = ibsimu.EpotField(geom)
    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.add_2d_beam_with_energy(
        100, 1.0e-3, 1.0, 1.0, 10000.0, 0.0, 0.0, 0.0, 15e-3, 0.0, 25e-3
    )

    # 1. Use GeomPlotter exactly like in C++
    print("Using GeomPlotter...")
    geomplotter = ibsimu.GeomPlotter(geom)
    geomplotter.set_size(750, 750)
    geomplotter.set_epot(epot)
    geomplotter.set_particle_database(pdb)

    # Additional configurations
    geomplotter.set_eqlines_auto(10)
    geomplotter.enable_colormap_legend(True)

    print("Generating plot1.png...")
    geomplotter.plot_png("examples_py/plot1.png")
    print("Done.")


if __name__ == "__main__":
    run_geomplotter_demo()
