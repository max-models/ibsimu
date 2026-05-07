#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "vec3d.hpp"
#include "types.hpp"
#include "geometry.hpp"
#include "solid.hpp"
#include "func_solid.hpp"
#include "epot_solver.hpp"
#include "epot_gssolver.hpp"
#include "epot_mgsolver.hpp"
#include "epot_umfpacksolver.hpp"
#include "epot_bicgstabsolver.hpp"
#include "particledatabase.hpp"
#include "convergence.hpp"
#include "epot_efield.hpp"
#include "meshvectorfield.hpp"
#include "ibsimu.hpp"
#include "particlediagplotter.hpp"
#include "fielddiagplotter.hpp"
#include "geomplotter.hpp"
#include "fieldgraph.hpp"
#include "meshcolormap.hpp"

namespace py = pybind11;

class PySolid : public Solid {
    py::function _func;
public:
    PySolid(py::function func) : _func(func) {}
    virtual bool inside(const Vec3D &x) const override {
        py::gil_scoped_acquire acquire;
        return _func(x[0], x[1], x[2]).cast<bool>();
    }
    virtual void debug_print(std::ostream &os) const override { os << "PySolid"; }
    virtual void save(std::ostream &s) const override { }
};

PYBIND11_MODULE(ibsimu, m) {
    m.doc() = "Python wrapper for IBSimu";

    // Enums
    py::enum_<geom_mode_e>(m, "GeometryMode")
        .value("MODE_1D", MODE_1D)
        .value("MODE_2D", MODE_2D)
        .value("MODE_3D", MODE_3D)
        .value("MODE_CYL", MODE_CYL)
        .export_values();

    py::enum_<bound_e>(m, "BoundType")
        .value("BOUND_NEUMANN", BOUND_NEUMANN)
        .value("BOUND_DIRICHLET", BOUND_DIRICHLET)
        .export_values();

    py::enum_<coordinate_axis_e>(m, "Axis")
        .value("AXIS_X", AXIS_X)
        .value("AXIS_Y", AXIS_Y)
        .value("AXIS_Z", AXIS_Z)
        .value("AXIS_R", AXIS_R)
        .export_values();

    py::enum_<field_extrpl_e>(m, "FieldExtrapolation")
        .value("FIELD_EXTRAPOLATE", FIELD_EXTRAPOLATE)
        .value("FIELD_MIRROR", FIELD_MIRROR)
        .value("FIELD_ANTIMIRROR", FIELD_ANTIMIRROR)
        .value("FIELD_SYMMETRIC_POTENTIAL", FIELD_SYMMETRIC_POTENTIAL)
        .value("FIELD_ZERO", FIELD_ZERO)
        .value("FIELD_NAN", FIELD_NAN)
        .export_values();

    py::enum_<trajectory_diagnostic_e>(m, "TrajectoryDiagnostic")
        .value("DIAG_NONE", DIAG_NONE)
        .value("DIAG_Y", DIAG_Y)
        .value("DIAG_YP", DIAG_YP)
        .export_values();

    py::enum_<field_type_e>(m, "FieldType")
        .value("FIELD_TRAJDENS", FIELD_TRAJDENS)
        .export_values();

    py::enum_<zscale_e>(m, "ZScale")
        .value("ZSCALE_LINEAR", ZSCALE_LINEAR)
        .value("ZSCALE_LOG", ZSCALE_LOG)
        .value("ZSCALE_RELLOG", ZSCALE_RELLOG)
        .export_values();

    py::enum_<particle_diag_plot_type_e>(m, "ParticleDiagPlotType")
        .value("PARTICLE_DIAG_PLOT_SCATTER", PARTICLE_DIAG_PLOT_SCATTER)
        .value("PARTICLE_DIAG_PLOT_HISTO1D", PARTICLE_DIAG_PLOT_HISTO1D)
        .value("PARTICLE_DIAG_PLOT_HISTO2D", PARTICLE_DIAG_PLOT_HISTO2D)
        .export_values();

    // Vec3D
    py::class_<Vec3D>(m, "Vec3D")
        .def(py::init<double, double, double>(), py::arg("x")=0, py::arg("y")=0, py::arg("z")=0)
        .def_property("x", [](Vec3D &v) { return v[0]; }, [](Vec3D &v, double x) { v[0] = x; })
        .def_property("y", [](Vec3D &v) { return v[1]; }, [](Vec3D &v, double y) { v[1] = y; })
        .def_property("z", [](Vec3D &v) { return v[2]; }, [](Vec3D &v, double z) { v[2] = z; })
        .def("__repr__", [](const Vec3D &v) {
            return "Vec3D(" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", " + std::to_string(v[2]) + ")";
        });

    // Int3D
    py::class_<Int3D>(m, "Int3D")
        .def(py::init<int32_t, int32_t, int32_t>(), py::arg("i")=0, py::arg("j")=0, py::arg("k")=0)
        .def("__repr__", [](const Int3D &i) {
            return "Int3D(" + std::to_string(i[0]) + ", " + std::to_string(i[1]) + ", " + std::to_string(i[2]) + ")";
        });

    // Bound
    py::class_<Bound>(m, "Bound")
        .def(py::init<bound_e, double>());

    // Solid
    py::class_<Solid, std::unique_ptr<Solid, py::nodelete>>(m, "Solid");

    // PySolid
    py::class_<PySolid, Solid>(m, "PySolid");

    // FuncSolid
    m.def("FuncSolid", [](py::function func) {
        return new PySolid(func);
    }, py::return_value_policy::take_ownership);

    // Geometry
    py::class_<Geometry>(m, "Geometry")
        .def(py::init<geom_mode_e, Int3D, Vec3D, double>())
        .def("set_solid", &Geometry::set_solid)
        .def("set_boundary", &Geometry::set_boundary)
        .def("build_mesh", &Geometry::build_mesh);

    // EpotSolver
    py::class_<EpotSolver>(m, "EpotSolver");

    // EpotGSSolver
    py::class_<EpotGSSolver, EpotSolver>(m, "EpotGSSolver")
        .def(py::init<Geometry &>())
        .def("solve", &EpotGSSolver::solve)
        .def("set_initial_plasma", &EpotGSSolver::set_initial_plasma)
        .def("set_pexp_plasma", &EpotGSSolver::set_pexp_plasma);

    // InitialPlasma
    py::class_<InitialPlasma>(m, "InitialPlasma")
        .def(py::init<coordinate_axis_e, double>());

    // ScalarField
    py::class_<ScalarField>(m, "ScalarField");

    // EpotField
    py::class_<EpotField, ScalarField>(m, "EpotField")
        .def(py::init<Geometry &>());

    // MeshScalarField
    py::class_<MeshScalarField, ScalarField>(m, "MeshScalarField")
        .def(py::init<Geometry &>());

    // VectorField
    py::class_<VectorField>(m, "VectorField");

    // MeshVectorField
    py::class_<MeshVectorField, VectorField>(m, "MeshVectorField")
        .def(py::init<>());

    // EpotEfield
    py::class_<EpotEfield, VectorField>(m, "EpotEfield")
        .def(py::init<EpotField &>())
        .def("recalculate", &EpotEfield::recalculate)
        .def("set_extrapolation", [](EpotEfield &ef, py::list ext) {
            field_extrpl_e e[6];
            for(int i=0; i<6; i++) e[i] = ext[i].cast<field_extrpl_e>();
            ef.set_extrapolation(e);
        });

    // ParticleDataBase
    py::class_<ParticleDataBase>(m, "ParticleDataBase")
        .def("clear", &ParticleDataBase::clear)
        .def("iterate_trajectories", &ParticleDataBase::iterate_trajectories)
        .def("get_rhosum", &ParticleDataBase::get_rhosum)
        .def("build_trajectory_density_field", &ParticleDataBase::build_trajectory_density_field)
        .def("set_mirror", [](ParticleDataBase &pdb, py::list mirror) {
            bool m[6];
            for(int i=0; i<6; i++) m[i] = mirror[i].cast<bool>();
            pdb.set_mirror(m);
        })
        .def("set_polyint", &ParticleDataBase::set_polyint);

    py::class_<ParticleDataBase2D, ParticleDataBase>(m, "ParticleDataBase2D")
        .def(py::init<Geometry &>())
        .def("add_2d_beam_with_energy", &ParticleDataBase2D::add_2d_beam_with_energy);

    // Convergence
    py::class_<Convergence>(m, "Convergence")
        .def(py::init<>())
        .def("add_epot", &Convergence::add_epot)
        .def("add_scharge", &Convergence::add_scharge)
        .def("add_emittance", &Convergence::add_emittance)
        .def("evaluate_iteration", &Convergence::evaluate_iteration)
        .def("print_history", [](Convergence &c, std::string filename) {
            std::ofstream of(filename);
            c.print_history(of);
        });

    // Emittance
    py::class_<Emittance>(m, "Emittance")
        .def(py::init<>());

    // ParticleDiagPlotter
    py::class_<ParticleDiagPlotter>(m, "ParticleDiagPlotter")
        .def(py::init<const Geometry &, const ParticleDataBase &, coordinate_axis_e, double, particle_diag_plot_type_e, trajectory_diagnostic_e, trajectory_diagnostic_e>())
        .def("calculate_emittance", &ParticleDiagPlotter::calculate_emittance)
        .def("set_font_size", &ParticleDiagPlotter::set_font_size)
        .def("set_size", &ParticleDiagPlotter::set_size)
        .def("plot_png", &ParticleDiagPlotter::plot_png);

    // MeshColormap
    py::class_<MeshColormap>(m, "MeshColormap")
        .def("set_zscale", &MeshColormap::set_zscale);

    // FieldGraph
    py::class_<FieldGraph, MeshColormap>(m, "FieldGraph");

    // GeomPlotter
    py::class_<GeomPlotter>(m, "GeomPlotter")
        .def(py::init<const Geometry &>())
        .def("set_size", &GeomPlotter::set_size)
        .def("set_font_size", &GeomPlotter::set_font_size)
        .def("set_epot", &GeomPlotter::set_epot)
        .def("set_particle_database", &GeomPlotter::set_particle_database)
        .def("set_trajdens", &GeomPlotter::set_trajdens)
        .def("plot_png", &GeomPlotter::plot_png)
        .def("set_eqlines_manual", &GeomPlotter::set_eqlines_manual)
        .def("set_particle_div", &GeomPlotter::set_particle_div)
        .def("set_fieldgraph_plot", &GeomPlotter::set_fieldgraph_plot)
        .def("fieldgraph", (FieldGraph* (GeomPlotter::*)()) &GeomPlotter::fieldgraph, py::return_value_policy::reference_internal);
}
