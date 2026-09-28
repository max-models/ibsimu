"""Cylindrical lens set up with the high-level Simulation class.

The same lens as sim_cyl_lens.py, but declared instead of assembled by hand.
Results come back as xarray objects, which plot directly with matplotlib.
"""

import matplotlib.pyplot as plt

import ibsimu


def main():
    sim = ibsimu.Simulation(
        mode="cyl",
        size=(201, 41, 1),
        h=0.5e-3,
        solids={
            7: lambda x, r, z: r > 15e-3,  # grounded beam pipe
            8: lambda x, r, z: 45e-3 < x < 55e-3 and r > 10e-3,  # lens electrode
        },
        boundaries={7: 0.0, 8: -10e3},
        beams=[
            {
                "method": "2d_beam_with_energy",
                "N": 50,
                "J": 1e-3,
                "q": 1.0,
                "m": 1.0,
                "E": 20e3,
                "Tp": 0.0,
                "Tt": 0.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.0,
                "y2": 8e-3,
            }
        ],
        solver_options={"eps": 1e-5},
        threads=2,
    )
    out = sim.run(iterations=3)
    out.info()

    epot = out.evaluate("epot")
    traj = out.evaluate("trajectories")
    exit_plane = out.diagnostics(ibsimu.AXIS_X, 95e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    (epot * 1e-3).plot(
        ax=ax1, x="x", y="r", cmap="viridis", cbar_kwargs={"label": "potential [kV]"}
    )
    ax1.plot(traj.sel(coord="x").T, traj.sel(coord="r").T, color="w", lw=0.5, alpha=0.7)
    ax1.set_title("Potential and trajectories")
    ax1.set_aspect("equal")

    ax2.plot(exit_plane["r"] * 1e3, exit_plane["rp"] * 1e3, ".")
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel("r' [mrad]")
    ax2.set_title("Phase space at x = 95 mm")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig("examples_py/sim_simulation_class.png", dpi=120)
    print("Saved examples_py/sim_simulation_class.png")


if __name__ == "__main__":
    main()
