"""Plot Simulation output with the plasma-plots xarray accessors.

Requires plasma-plots (https://github.com/struphy-hub/plasma-plots):
``pip install git+https://github.com/struphy-hub/plasma-plots`` or an editable
install of a local checkout. ``import plasma_plots`` adds ``.plasma`` to every
xarray object, so the arrays returned by ``SimulationOutput.evaluate`` plot
with one call each.
"""

import plasma_plots

import ibsimu


def main():
    sim = ibsimu.Simulation(
        mode="cyl",
        size=(201, 41, 1),
        h=0.5e-3,
        solids={
            7: lambda x, r, z: r > 15e-3,
            8: lambda x, r, z: 45e-3 < x < 55e-3 and r > 10e-3,
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
        threads=2,
        name="cylindrical lens",
    )
    out = sim.run(iterations=3)
    out.info()

    epot = out.evaluate("epot")
    markers = out.evaluate("markers")
    exit_plane = out.diagnostics(
        ibsimu.AXIS_X, 95e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP, ibsimu.DIAG_CURR]
    )

    with plasma_plots.figure(2, 2, figsize=(12, 8)) as fig:
        epot.plasma.plot.slice(x="x", y="r", ax=fig[0, 0], levels=[-8e3, -4e3, -1e3])
        epot.plasma.plot.lineout(x="x", r=0.0, ax=fig[0, 1], title="on-axis potential")
        markers.plasma.plot.paths(x="x", y="r", markers=10, ax=fig[1, 0])
        exit_plane.plasma.plot.scatter(x="r", y="rp", color="curr", ax=fig[1, 1])
    fig.save("examples_py/viz_plasma_plots.png")
    print("Saved examples_py/viz_plasma_plots.png")

    # Interactive alternatives (need a display or a notebook):
    # epot.plasma.plot.slice(x="x", y="r", backend="plotly").show()
    # markers.plasma.plot.animation(x="x", y="r", step=5)


if __name__ == "__main__":
    main()
