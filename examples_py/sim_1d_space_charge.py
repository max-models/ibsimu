"""Space-charge-filled gap in 1D.

A 10 mm gap between 0 V and 10 kV electrodes holds a uniform space charge
density. The potential has the analytic solution
phi(x) = U x / d + rho x (d - x) / (2 epsilon_0), which the example compares
with three solvers (multigrid, BiCGSTAB with an ILU1 preconditioner, UMFPACK
when available). The result is drawn with IBSimu's own plotting classes
(Frame and XYGraph) to sim_1d_space_charge.png.
"""

import numpy as np

import ibsimu

D, U, RHO = 10e-3, 10e3, 1e-3  # gap [m], voltage [V], charge density [C/m^3]


def exact(x):
    return U * x / D + RHO * x * (D - x) / (2 * ibsimu.EPSILON0)


def main():
    solvers = {
        "mg": {"mgcycmax": 100, "mgeps": 1e-8},
        "bicgstab": {"preconditioner": ibsimu.ILU1_Precond(), "eps": 1e-10},
    }
    if hasattr(ibsimu, "EpotUMFPACKSolver"):
        solvers["umfpack"] = {}

    frame = ibsimu.Frame()
    colors = [ibsimu.Vec3D(1, 0, 0), ibsimu.Vec3D(0, 0.6, 0), ibsimu.Vec3D(0, 0, 1)]
    graphs = []
    for (name, options), color in zip(solvers.items(), colors):
        sim = ibsimu.Simulation(
            "1d",
            (41,),
            D / 40,
            boundaries={1: 0.0, 2: U},
            scharge=RHO,
            solver=name,
            solver_options=options,
            name=name,
        )
        epot = sim.run(verbose=False).evaluate("epot")
        x = epot["x"].values
        err = np.max(np.abs(epot.values - exact(x)))
        print(f"{name:9s}: max |phi - exact| = {err:.3g} V")

        g = ibsimu.XYGraph(list(x * 1e3), list(epot.values / 1e3))
        g.set_color(color)
        g.set_line_style(ibsimu.XYGRAPH_LINE_DISABLE)
        g.set_point_style(ibsimu.XYGRAPH_POINT_CIRCLE, False, 2.0 + len(graphs))
        graphs.append(g)
        frame.add_graph(
            ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, g, ibsimu.LegendEntry(g, name)
        )

    xs = np.linspace(0, D, 200)
    ref = ibsimu.XYGraph(list(xs * 1e3), list(exact(xs) / 1e3))
    ref.set_color(ibsimu.Vec3D(0, 0, 0))
    ref.set_line_style(ibsimu.XYGRAPH_LINE_SOLID, 1.5)
    ref.set_point_style(ibsimu.XYGRAPH_POINT_DISABLE)
    frame.add_graph(
        ibsimu.PLOT_AXIS_X1, ibsimu.PLOT_AXIS_Y1, ref, ibsimu.LegendEntry(ref, "exact")
    )
    frame.set_axis_label(ibsimu.PLOT_AXIS_X1, "x (mm)")
    frame.set_axis_label(ibsimu.PLOT_AXIS_Y1, "Potential (kV)")
    frame.enable_legend(True)
    frame.set_legend_position(ibsimu.LEGEND_POS_TOP_LEFT)
    frame.plot_png("sim_1d_space_charge.png", 640, 480)


if __name__ == "__main__":
    main()
