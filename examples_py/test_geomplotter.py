import ibsimu


def test_geomplotter():
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(101, 41, 1), ibsimu.Vec3D(0, 0, 0), h
    )
    geom.build_mesh()

    epot = ibsimu.EpotField(geom)
    pdb = ibsimu.ParticleDataBase2D(geom)

    print("Creating GeomPlotter...")
    gp = ibsimu.GeomPlotter(geom)
    print("Setting size...")
    gp.set_size(750, 750)
    print("Setting epot...")
    gp.set_epot(epot)
    print("Setting particle database...")
    gp.set_particle_database(pdb)
    print("Plotting...")
    gp.plot_png("test_geomplotter.png")
    print("Done.")


if __name__ == "__main__":
    test_geomplotter()
