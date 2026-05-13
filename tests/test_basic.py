import ibsimu


def test_basic_simulation():
    print("Starting ibsimu Python basic simulation test...")
    h = 1e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(51, 21, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    def my_solid(x, y, z):
        return x > 25e-3

    s = ibsimu.FuncSolid(my_solid)
    geom.set_solid(7, s)

    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))

    print("Building mesh...")
    geom.build_mesh()

    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    solver = ibsimu.EpotGSSolver(geom)

    print("Solving...")
    solver.solve(epot, scharge)

    pdb = ibsimu.ParticleDataBase2D(geom)
    pdb.add_2d_beam_with_energy(10, 1e-3, 1.0, 1.0, 1000.0, 0, 0, 0, 5e-3, 0, 15e-3)

    print("Tracking particles...")
    pdb.iterate_trajectories(scharge, ibsimu.EpotEfield(epot), ibsimu.MeshVectorField())

    print("Checking results...")
    assert pdb.size() == 10
    print("Smoke test passed!")


if __name__ == "__main__":
    smoke_test()
