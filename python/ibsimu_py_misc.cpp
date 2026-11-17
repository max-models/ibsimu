// IBSimu Python bindings: ASCII data files and space charge deposition from
// user-supplied trajectories.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pthread.h>

#include "readascii.hpp"
#include "meshscalarfield.hpp"
#include "particles.hpp"
#include "scharge.hpp"
#include "ibsimu_py_common.hpp"

// The deposition functions lock a mutex around each update.  Python calls
// them one at a time, so one module-wide mutex is enough.
static pthread_mutex_t g_scharge_mutex = PTHREAD_MUTEX_INITIALIZER;

static void require_3d(const MeshScalarField &f) {
    if (f.geom_mode() != MODE_3D || f.size(0) < 2 || f.size(1) < 2 || f.size(2) < 2)
        throw std::invalid_argument("Step deposition needs a 3D mesh with at least 2 nodes per direction");
}

template <typename PP>
static void def_scharge_add(py::module_ &m) {
    m.def("scharge_add_from_trajectory_pic",
          [](MeshScalarField &scharge, double I, const PP &x1, const PP &x2) {
              scharge_add_from_trajectory_pic(scharge, &g_scharge_mutex, I, x1, x2);
          },
          py::arg("scharge"), py::arg("I"), py::arg("x1"), py::arg("x2"),
          "Deposit the charge of a trajectory segment x1 -> x2 carrying current I (A, or A/m in 2D)\n"
          "onto the mesh with the particle-in-cell weighting. Call scharge_finalize_pic() after all\n"
          "segments to divide by the cell volumes.");
}

void init_misc_extra(py::module_ &m) {
    // -----------------------------------------------------------------------
    // ReadAscii
    // -----------------------------------------------------------------------
    py::class_<ReadAscii>(m, "ReadAscii", "Columns of numbers from an ASCII file ('#' comments)")
        .def(py::init<>())
        .def(py::init<const std::string &, int>(), py::arg("filename"), py::arg("columns") = -1)
        .def("read", &ReadAscii::read, py::arg("filename"), py::arg("columns") = -1)
        .def("set_comment_character", &ReadAscii::set_comment_character, py::arg("c"))
        .def("clear", &ReadAscii::clear)
        .def("columns", &ReadAscii::columns)
        .def("rows", &ReadAscii::rows)
        .def("__getitem__", [](const ReadAscii &r, uint32_t i) {
            if (i >= r.columns()) throw py::index_error("Column index out of range");
            const std::vector<double> &c = r[i];
            return py::array_t<double>((py::ssize_t)c.size(), c.data());
        }, py::arg("i"), "Column i as an array")
        .def("numpy", [](const ReadAscii &r) {
            py::array_t<double> a({ (py::ssize_t)r.rows(), (py::ssize_t)r.columns() });
            auto d = a.mutable_unchecked<2>();
            for (uint32_t j = 0; j < r.columns(); j++) {
                const std::vector<double> &c = r[j];
                for (uint32_t i = 0; i < r.rows(); i++) d(i, j) = c[i];
            }
            return a;
        }, "All data as a (rows, columns) array");

    // -----------------------------------------------------------------------
    // Space charge deposition (scharge.hpp)
    // -----------------------------------------------------------------------
    def_scharge_add<ParticleP2D>(m);
    def_scharge_add<ParticlePCyl>(m);
    def_scharge_add<ParticleP3D>(m);
    m.def("scharge_finalize_pic", &scharge_finalize_pic, py::arg("scharge"),
          "Convert the charge deposited with scharge_add_from_trajectory_pic() to charge density");
    // The step deposition always spreads over 2x2x2 nodes, so it is only
    // defined on 3D meshes (on thinner meshes it would write past the data).
    m.def("scharge_add_step_pic", [](MeshScalarField &scharge, double Q, const Vec3D &x) {
              require_3d(scharge);
              scharge_add_step_pic(scharge, Q, x);
          }, py::arg("scharge"), py::arg("Q"), py::arg("x"),
          "Deposit point charge Q (C) at x with the particle-in-cell weighting (3D meshes only)");
    m.def("scharge_finalize_step_pic", [](MeshScalarField &scharge) {
              require_3d(scharge);
              scharge_finalize_step_pic(scharge);
          }, py::arg("scharge"),
          "Convert the charge deposited with scharge_add_step_pic() to charge density (3D meshes only)");
}
