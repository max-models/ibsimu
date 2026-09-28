"""The high-level Simulation / SimulationOutput interface."""

import json

import numpy as np
import pytest
import xarray as xr

import ibsimu


def lens_simulation(**overrides):
    """Cylindrical einzel-type lens: grounded pipe, biased ring electrode."""
    kwargs = {
        "mode": "cyl",
        "size": (121, 31, 1),
        "h": 0.5e-3,
        "solids": {
            7: lambda x, r, z: r > 14e-3,
            8: lambda x, r, z: 25e-3 < x < 35e-3 and r > 9e-3,
        },
        "boundaries": {7: 0.0, 8: -8e3},
        "beams": [
            {
                "method": "2d_beam_with_energy",
                "N": 30,
                "J": 1e-3,
                "q": 1.0,
                "m": 1.0,
                "E": 20e3,
                "Tp": 0.0,
                "Tt": 0.0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.0,
                "y2": 6e-3,
            }
        ],
        "solver_options": {"eps": 1e-4},
        "threads": 1,
    }
    kwargs.update(overrides)
    return ibsimu.Simulation(**kwargs)


@pytest.fixture(scope="module")
def out():
    return lens_simulation().run(iterations=2, verbose=False)


def test_run_returns_output_with_library_objects(out):
    assert isinstance(out, ibsimu.SimulationOutput)
    assert isinstance(out.geom, ibsimu.Geometry)
    assert isinstance(out.pdb, ibsimu.ParticleDataBaseCyl)
    assert isinstance(out.solver, ibsimu.EpotGSSolver)
    assert out.iterations == 2
    assert len(out.history) == 2
    assert np.isnan(out.history[0]["epot_change"])
    assert out.history[1]["epot_change"] >= 0.0
    assert out.pdb.size() == 30


def test_info_prints_summary(out, capsys):
    out.info()
    text = capsys.readouterr().out
    assert "cyl, 121 x 31 nodes" in text
    assert "solver       : EpotGSSolver" in text
    assert "iterations   : 2" in text
    assert "particles    : 30" in text


def test_evaluate_scalar_fields(out):
    epot = out.evaluate("epot")
    assert isinstance(epot, xr.DataArray)
    assert epot.dims == ("x", "r")
    assert epot.shape == (121, 31)
    assert epot.attrs["units"] == "V"
    np.testing.assert_allclose(epot.coords["x"][:3], [0.0, 0.5e-3, 1.0e-3])
    assert float(epot.sel(x=30e-3, r=12e-3, method="nearest")) == -8e3
    assert float(epot.sel(x=0.0, r=0.0, method="nearest")) > -2e3

    assert out.evaluate("scharge").min() >= 0.0
    solid = out.evaluate("solid")
    assert int(solid.sel(x=30e-3, r=12e-3, method="nearest")) == 8
    assert int(solid.sel(x=30e-3, r=5e-3, method="nearest")) == 0
    assert int(solid.sel(x=30e-3, r=0.0, method="nearest")) == 3  # symmetry axis
    assert out.evaluate("trajdens").sum() > 0.0


def test_evaluate_vector_fields(out):
    ef = out.evaluate("efield")
    assert ef.dims == ("x", "r", "component")
    assert list(ef.coords["component"].values) == ["x", "y", "z"]
    # The field points along -x in front of the negative electrode.
    at = ef.sel(x=20e-3, r=12e-3, method="nearest")
    assert float(at.sel(component="x")) > 0.0
    assert out.evaluate("bfield").shape == (121, 31, 3)


def test_evaluate_trajectories(out):
    tr = out.evaluate("trajectories")
    assert tr.dims == ("particle", "point", "coord")
    assert tr.shape[0] == 30
    assert list(tr.coords["coord"].values) == ["t", "x", "vx", "r", "vr", "w"]
    x = tr.sel(coord="x")
    assert float(x.isel(point=0).max()) == 0.0
    assert float(x.max()) > 55e-3  # beam reaches the far end
    sub = out.evaluate("trajectories", particles=[0, 5])
    assert list(sub.coords["particle"].values) == [0, 5]


def test_diagnostics_dataset(out):
    ds = out.diagnostics(ibsimu.AXIS_X, 55e-3, [ibsimu.DIAG_R, ibsimu.DIAG_RP])
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"r", "rp"}
    assert ds["r"].size == 30
    assert float(ds["r"].max()) < 6e-3  # focused by the lens


def test_unknown_key(out):
    with pytest.raises(KeyError, match="available"):
        out.evaluate("nope")


def test_save_and_load_roundtrip(out, tmp_path):
    out.save(str(tmp_path))
    loaded = ibsimu.SimulationOutput.load(str(tmp_path))
    xr.testing.assert_allclose(loaded.evaluate("epot"), out.evaluate("epot"))
    xr.testing.assert_allclose(
        loaded.evaluate("trajectories"), out.evaluate("trajectories")
    )
    assert loaded.iterations == 2
    assert json.dumps(loaded.history) == json.dumps(out.history)


def test_callback_and_early_stop():
    seen = []

    def cb(i, out):
        seen.append((i, out.iterations))
        raise StopIteration

    out = lens_simulation().run(iterations=5, callback=cb, verbose=False)
    assert seen == [(0, 1)]
    assert out.iterations == 1


def test_beam_callable_and_bound_variants():
    def beam(pdb):
        pdb.add_2d_beam_with_energy(
            4, 1e-3, 1.0, 1.0, 1e3, 0.0, 0.0, 0.0, 0.0, 0.0, 4e-3
        )

    class Ramp(ibsimu.CallbackFunctorD_V):
        def __call__(self, x):
            return -100.0 * x[0] / 60e-3

    sim = lens_simulation(beams=[beam], boundaries={7: ("dirichlet", Ramp()), 8: -8e3})
    out = sim.run(verbose=False)
    assert out.pdb.size() == 4
    epot = out.evaluate("epot")
    assert float(epot.sel(x=59e-3, r=14.5e-3, method="nearest")) < float(
        epot.sel(x=1e-3, r=14.5e-3, method="nearest")
    )


def test_configuration_errors():
    with pytest.raises(ValueError, match="no boundary condition"):
        ibsimu.Simulation("2d", (11, 11, 1), 1e-3, solids={7: lambda x, y, z: x > 5e-3})
    with pytest.raises(ValueError, match="Unknown geometry mode"):
        ibsimu.Simulation("4d", (11, 11, 1), 1e-3)
    with pytest.raises(ValueError, match="Unknown solver"):
        ibsimu.Simulation("2d", (11, 11, 1), 1e-3, solver="magic").run(verbose=False)
    with pytest.raises(ValueError, match="no option"):
        ibsimu.Simulation("2d", (11, 11, 1), 1e-3, solver_options={"bogus": 1}).run(
            verbose=False
        )
    with pytest.raises(ValueError, match="no beam method"):
        ibsimu.Simulation("2d", (11, 11, 1), 1e-3, beams=[{"method": "nothing"}]).run(
            verbose=False
        )


def test_plasma_model_applied_from_second_iteration():
    seen = []
    sim = ibsimu.Simulation(
        mode="2d",
        size=(41, 21, 1),
        h=0.5e-3,
        solids={7: lambda x, y, z: x > 15e-3 and abs(y - 5e-3) > 2e-3},
        boundaries={2: -2e3, 7: -2e3},
        beams=[
            {
                "method": "2d_beam_with_energy",
                "N": 200,
                "J": 50.0,
                "q": 1.0,
                "m": 1.0,
                "E": 5.0,
                "Tp": 0.0,
                "Tt": 0.5,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.0,
                "y2": 3e-3,
            }
        ],
        initial_plasma={"Up": 5.0, "axis": ibsimu.AXIS_X, "x": 1e-3},
        plasma={"kind": "pexp", "rhoe": None, "Te": 5.0, "Up": 5.0},
        pdb_options={"mirror": [False, False, True, False, False, False]},
        threads=1,
    )
    out = sim.run(iterations=3, callback=lambda i, o: seen.append(i), verbose=False)
    assert seen == [0, 1, 2]
    assert out.evaluate("epot").max() <= 5.0 + 1e-9


def test_markers_dataset_on_common_time_grid(out):
    mk = out.evaluate("markers", nt=50)
    assert isinstance(mk, xr.Dataset)
    assert set(mk.data_vars) == {"x", "vx", "r", "vr", "w"}
    assert mk["x"].dims == ("t", "marker") and mk["x"].shape == (50, 30)
    assert mk["x"].attrs["units"] == "m" and mk.coords["t"].attrs["units"] == "s"
    assert float(mk["x"].isel(t=0).max()) == 0.0
    # Every particle ends at some time: NaN afterwards, finite before.
    assert np.isnan(mk["x"].isel(t=-1)).any() or float(mk["x"].isel(t=-1).min()) > 0.0
    assert not np.isnan(mk["x"].isel(t=0)).any()
    sub = out.evaluate("markers", nt=10, particles=[1, 2])
    assert list(sub.coords["marker"].values) == [1, 2]


def test_output_attrs_follow_plasma_plots_conventions():
    out = lens_simulation(name="lens run").run(verbose=False)
    epot = out.evaluate("epot")
    assert epot.attrs["run"] == "lens run"
    assert epot.attrs["units"] == "V" and epot.attrs["label"]
    assert epot.coords["r"].attrs == {"label": "$r$", "long_name": "$r$", "units": "m"}
    ds = out.diagnostics(ibsimu.AXIS_X, 55e-3, [ibsimu.DIAG_RP, ibsimu.DIAG_CURR])
    assert ds["rp"].dims == ("marker",) and ds["curr"].attrs["units"] == "A"
