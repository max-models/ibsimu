import ibsimu
import matplotlib.pyplot as plt
import numpy as np

# Diagnostic plane near the downstream end of the simulated beam line.
X_TARGET = 195e-3

# Protons are decelerated inside the lens electrode, so the beam is reflected
# once the electrode potential approaches -20 kV (the beam energy).
V_MIN = -19.5e3

# RMS radius the lens should be tuned to produce at the target plane.
R_GOAL = 3.0e-3


def simulate_at_voltage(V_lens):
    # Setup a simple cylindrical Einzel lens
    h = 0.5e-3
    geom = ibsimu.Geometry(
        ibsimu.MODE_CYL, ibsimu.Int3D(401, 41, 1), ibsimu.Vec3D(0, 0, 0), h
    )

    # Electrodes. The mesh must extend past the pipe wall radius, otherwise the
    # pipe solid contains no mesh nodes and the geometry has no grounded region.
    def pipe_func(x, r, z):
        return r > 15e-3

    def lens_func(x, r, z):
        return 45e-3 < x < 55e-3 and r > 10e-3

    s_pipe = ibsimu.FuncSolid(pipe_func)
    s_lens = ibsimu.FuncSolid(lens_func)
    geom.set_solid(7, s_pipe)
    geom.set_solid(8, s_lens)

    for i in range(1, 5):
        geom.set_boundary(i, ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 0.0))
    geom.set_boundary(8, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, V_lens))
    geom.build_mesh()

    solver = ibsimu.EpotGSSolver(geom)
    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    bfield = ibsimu.MeshVectorField()

    solver.solve(epot, scharge)
    efield.recalculate()

    pdb = ibsimu.ParticleDataBaseCyl(geom)
    # 20keV Proton beam, starting at x=0, r=0..8mm
    pdb.add_2d_beam_with_energy(
        200, 1.0e-3, 1.0, 1.0, 20000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8e-3
    )
    pdb.iterate_trajectories(scharge, efield, bfield)

    # Calculate RMS beam radius at the target plane
    tdata = ibsimu.TrajectoryDiagnosticData([ibsimu.DIAG_R])
    pdb.trajectories_at_plane(tdata, ibsimu.AXIS_X, X_TARGET, [ibsimu.DIAG_R])
    r_data = np.array(tdata.column(0).data())
    rms_radius = np.sqrt(np.mean(r_data**2))

    return rms_radius


def solve_for_radius(r_goal, v_lo, v_hi, r_lo, r_hi, iterations=5):
    """Bisect on lens voltage to hit a wanted RMS radius at the target plane.

    The response is monotonic for this geometry, so plain bisection converges
    without needing a derivative of the (noisy, simulated) objective.
    """
    print(f"\nSolving for RMS radius = {r_goal * 1000:.2f} mm by bisection...")
    for _ in range(iterations):
        v_mid = 0.5 * (v_lo + v_hi)
        r_mid = simulate_at_voltage(v_mid)
        print(f"  V = {v_mid:8.0f} V -> RMS Radius = {r_mid * 1000:5.3f} mm")
        if (r_mid > r_goal) == (r_lo > r_goal):
            v_lo, r_lo = v_mid, r_mid
        else:
            v_hi, r_hi = v_mid, r_mid
    return (v_lo, r_lo) if abs(r_lo - r_goal) < abs(r_hi - r_goal) else (v_hi, r_hi)


def run_optimization():
    print("Scanning lens voltage...")
    voltages = np.linspace(V_MIN, 0, 15)
    radii = []

    for V in voltages:
        rms = simulate_at_voltage(V)
        print(f"  V = {V:8.0f} V -> RMS Radius = {rms * 1000:5.3f} mm")
        radii.append(rms)

    # For this single-electrode decel lens the focusing strength grows
    # monotonically with |V|: the beam is smallest at the most negative
    # voltage that still transmits it, and no interior optimum exists.
    best_idx = int(np.argmin(radii))
    print(
        f"\nSmallest beam in scan range: {voltages[best_idx]:.0f} V "
        f"(RMS Radius: {radii[best_idx] * 1000:.3f} mm)"
    )

    if min(radii) <= R_GOAL <= max(radii):
        v_goal, r_goal_actual = solve_for_radius(
            R_GOAL, voltages[0], voltages[-1], radii[0], radii[-1]
        )
        print(
            f"\nVoltage for {R_GOAL * 1000:.2f} mm target: {v_goal:.0f} V "
            f"(RMS Radius: {r_goal_actual * 1000:.3f} mm)"
        )
    else:
        v_goal = None
        print(f"\nTarget radius {R_GOAL * 1000:.2f} mm is outside the scanned range")

    plt.figure(figsize=(8, 6))
    plt.plot(voltages, np.array(radii) * 1000, "bo-")
    if v_goal is not None:
        plt.axhline(R_GOAL * 1000, color="0.5", linestyle=":", label="Target radius")
        plt.axvline(v_goal, color="r", linestyle="--", label=f"Solution: {v_goal:.0f} V")
        plt.legend()
    plt.xlabel("Lens Voltage [V]")
    plt.ylabel(f"RMS Beam Radius at x = {X_TARGET * 1000:.0f} mm [mm]")
    plt.title("Lens Voltage Scan and Tuning to a Target Beam Radius")
    plt.grid(True)
    plt.savefig("examples_py/tutorial_optimization.png")
    print("Saved examples_py/tutorial_optimization.png")


if __name__ == "__main__":
    run_optimization()
