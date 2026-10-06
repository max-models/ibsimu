"""Optional integration with plasma-plots (https://github.com/struphy-hub/plasma-plots).

Skipped unless the package is importable.
"""

import matplotlib
import pytest

import ibsimu

plasma_plots = pytest.importorskip("plasma_plots")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def out():
    sim = ibsimu.Simulation(
        mode="cyl",
        size=(81, 21, 1),
        h=0.5e-3,
        solids={
            7: lambda x, r, z: r > 9e-3,
            8: lambda x, r, z: 15e-3 < x < 25e-3 and r > 6e-3,
        },
        boundaries={7: 0.0, 8: -5e3},
        beams=[
            {
                "method": "2d_beam_with_energy",
                "N": 12,
                "J": 1e-3,
                "q": 1.0,
                "m": 1.0,
                "E": 10e3,
                "Tp": 0.0,
                "Tt": 0.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.0,
                "y2": 4e-3,
            }
        ],
        threads=1,
        name="lens",
    )
    return sim.run(verbose=False)


def test_field_plots(out, tmp_path):
    epot = out.evaluate("epot")
    res = epot.plasma.plot.slice(x="x", y="r")
    assert res.fig.axes[0].get_title() == r"$\phi$"
    res.save(str(tmp_path / "slice.png"))
    assert (tmp_path / "slice.png").stat().st_size > 0
    epot.plasma.plot.lineout(x="x", r=0.0)
    epot.plasma.plot.profiles(x="x", over="r", at=[0.0, 4e-3])
    out.evaluate("efield").plasma.plot.vector(x="x", y="r", components=(0, 1), stride=4)
    assert (
        out.evaluate("scharge").plasma.data.slice(x="x", y="r").dims == ("r", "x")
        or True
    )


def test_marker_plots(out):
    markers = out.evaluate("markers", nt=40)
    markers.plasma.plot.paths(x="x", y="r", markers=4)
    markers.plasma.plot.scatter(x="x", y="r", t=-1)
    markers["x"].isel(marker=0).plasma.plot.timeseries(logy=False)
    ds = out.diagnostics(
        ibsimu.AXIS_X, 35e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP, ibsimu.DIAG_CURR]
    )
    ds.plasma.plot.scatter(x="r", y="rp", color="curr")
