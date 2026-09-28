// IBSimu Python bindings - comprehensive wrapper for the IBSimu C++ library
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <sstream>
#include <mutex>

#include "vec3d.hpp"
#include "vec4d.hpp"
#include "mat3d.hpp"
#include "types.hpp"
#include "constants.hpp"
#include "mesh.hpp"
#include "geometry.hpp"
#include "solid.hpp"
#include "func_solid.hpp"
#include "stl_solid.hpp"
#include "dxf_solid.hpp"
#ifdef CSG
#include "csgobject_solid.hpp"
#endif
#include "mydxffile.hpp"
#include "transformation.hpp"
#include "epot_solver.hpp"
#include "epot_gssolver.hpp"
#include "epot_mgsolver.hpp"
#include "epot_matrixsolver.hpp"
#include "epot_bicgstabsolver.hpp"
#ifdef HAVE_LIBUMFPACK
#include "epot_umfpacksolver.hpp"
#endif
#include "epot_field.hpp"
#include "epot_efield.hpp"
#include "scalarfield.hpp"
#include "vectorfield.hpp"
#include "meshscalarfield.hpp"
#include "meshvectorfield.hpp"
#include "multimeshvectorfield.hpp"
#include "axisymmetricvectorfield.hpp"
#include "particles.hpp"
#include "particlestatistics.hpp"
#include "particledatabase.hpp"
#include "trajectorydiagnostics.hpp"
#include "convergence.hpp"
#include "histogram.hpp"
#include "interpolation.hpp"
#include "random.hpp"
#include "timer.hpp"
#include "statusprint.hpp"
#include "polysolver.hpp"
#include "ibsimu.hpp"
#include "particlediagplotter.hpp"
#include "fielddiagplotter.hpp"
#include "geomplotter.hpp"
#include "fieldgraph.hpp"
#include "meshcolormap.hpp"
#include "callback.hpp"
#include "error.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Python errors raised inside callbacks
//
// The library calls solids and trajectory callbacks from worker threads while
// the binding has released the GIL.  A Python exception cannot propagate
// through those threads, so the first one is parked here and re-raised in
// the calling thread once the C++ call returns (see rethrow_python_error()).
// ---------------------------------------------------------------------------
static std::mutex g_pyerr_mutex;
static std::exception_ptr g_pyerr;

static void store_python_error() {
    std::lock_guard<std::mutex> lock(g_pyerr_mutex);
    if (!g_pyerr) g_pyerr = std::current_exception();
}

static bool python_error_pending() {
    std::lock_guard<std::mutex> lock(g_pyerr_mutex);
    return (bool)g_pyerr;
}

static void rethrow_python_error() {
    std::exception_ptr e;
    {
        std::lock_guard<std::mutex> lock(g_pyerr_mutex);
        std::swap(e, g_pyerr);
    }
    if (e) std::rethrow_exception(e);
}

// Run f() (which calls into Python) and park any exception it raises.
template <typename R, typename F>
static R guarded_call(R fallback, F f) {
    if (python_error_pending()) return fallback;
    try {
        return f();
    } catch (py::error_already_set &) {
        store_python_error();
    } catch (std::exception &) {
        store_python_error();
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// Trampoline: Solid subclassable from Python via a callable
// ---------------------------------------------------------------------------
class PySolid : public Solid {
    py::function _func;
public:
    PySolid(py::function func) : _func(func) {}
    virtual bool inside(const Vec3D &x) const override {
        Vec3D y = _T.transform_point(x);
        py::gil_scoped_acquire acquire;
        return guarded_call<bool>(false, [&]{ return _func(y[0], y[1], y[2]).cast<bool>(); });
    }
    virtual void debug_print(std::ostream &os) const override { os << "PySolid"; }
    virtual void save(std::ostream &s) const override {
        throw std::runtime_error("FuncSolid cannot be saved to a file: "
                                 "the solid is defined by a Python callable. "
                                 "Use Geometry.save(filename, save_solids=False).");
    }
};

// ---------------------------------------------------------------------------
// Trampolines: functors subclassable from Python (override __call__)
// ---------------------------------------------------------------------------
#define IBSIMU_PY_FUNCTOR(Base, Ret, Fallback, Args, CallArgs)                     \
    class Py##Base : public Base {                                                  \
    public:                                                                         \
        Ret operator() Args const override {                                        \
            py::gil_scoped_acquire gil;                                             \
            return guarded_call<Ret>(Fallback, [&]() -> Ret {                        \
                PYBIND11_OVERRIDE_PURE_NAME(Ret, Base, "__call__", operator(), CallArgs); \
            });                                                                     \
        }                                                                           \
    };

#define IBSIMU_ARGS(...) (__VA_ARGS__)
#define IBSIMU_CALL(...) __VA_ARGS__
IBSIMU_PY_FUNCTOR(CallbackFunctorD_3D, double, 0.0,   IBSIMU_ARGS(double x, double y, double z), IBSIMU_CALL(x, y, z))
IBSIMU_PY_FUNCTOR(CallbackFunctorB_3D, bool,   false, IBSIMU_ARGS(double x, double y, double z), IBSIMU_CALL(x, y, z))
IBSIMU_PY_FUNCTOR(CallbackFunctorD_D,  double, 0.0,   IBSIMU_ARGS(double x),                     IBSIMU_CALL(x))
IBSIMU_PY_FUNCTOR(CallbackFunctorB_V,  bool,   false, IBSIMU_ARGS(const Vec3D &x),               IBSIMU_CALL(x))
IBSIMU_PY_FUNCTOR(CallbackFunctorD_V,  double, 0.0,   IBSIMU_ARGS(const Vec3D &x),               IBSIMU_CALL(x))

// ---------------------------------------------------------------------------
// Trampolines: trajectory callbacks
//
// The library hands over ParticleBase / ParticlePBase pointers, which carry
// no run-time type information.  The concrete type follows from the geometry
// mode, which is recorded when the callback is registered on a database.
// ---------------------------------------------------------------------------
static py::object cast_particle(ParticleBase *p, geom_mode_e mode) {
    const auto ref = py::return_value_policy::reference;
    switch (mode) {
    case MODE_2D:  return py::cast(static_cast<Particle2D *>(p), ref);
    case MODE_CYL: return py::cast(static_cast<ParticleCyl *>(p), ref);
    case MODE_3D:  return py::cast(static_cast<Particle3D *>(p), ref);
    default:       return py::cast(p, ref);
    }
}

static py::object cast_particlep(ParticlePBase *x, geom_mode_e mode) {
    const auto ref = py::return_value_policy::reference;
    switch (mode) {
    case MODE_2D:  return py::cast(static_cast<ParticleP2D *>(x), ref);
    case MODE_CYL: return py::cast(static_cast<ParticlePCyl *>(x), ref);
    case MODE_3D:  return py::cast(static_cast<ParticleP3D *>(x), ref);
    default:       return py::cast(x, ref);
    }
}

class PyTrajectoryHandlerCallback : public TrajectoryHandlerCallback {
public:
    geom_mode_e mode = MODE_2D;
    void operator()(ParticleBase *particle, ParticlePBase *xcur, ParticlePBase *xend) override {
        py::gil_scoped_acquire gil;
        guarded_call<int>(0, [&]{
            py::function f = py::get_override(static_cast<const TrajectoryHandlerCallback *>(this), "__call__");
            if (!f) py::pybind11_fail("TrajectoryHandlerCallback subclass must define __call__");
            f(cast_particle(particle, mode), cast_particlep(xcur, mode), cast_particlep(xend, mode));
            return 0;
        });
    }
};

class PyTrajectoryEndCallback : public TrajectoryEndCallback {
public:
    geom_mode_e mode = MODE_2D;
    void operator()(ParticleBase *particle, ParticleDataBase *pdb) override {
        py::gil_scoped_acquire gil;
        guarded_call<int>(0, [&]{
            py::function f = py::get_override(static_cast<const TrajectoryEndCallback *>(this), "__call__");
            if (!f) py::pybind11_fail("TrajectoryEndCallback subclass must define __call__");
            f(cast_particle(particle, pdb->geom_mode()), py::cast(pdb, py::return_value_policy::reference));
            return 0;
        });
    }
};

class PyTrajectorySurfaceCollisionCallback : public TrajectorySurfaceCollisionCallback {
public:
    geom_mode_e mode = MODE_2D;
    void operator()(ParticleBase *particle, ParticlePBase *x, uint32_t tri,
                    double s, double t) override {
        py::gil_scoped_acquire gil;
        guarded_call<int>(0, [&]{
            py::function f = py::get_override(static_cast<const TrajectorySurfaceCollisionCallback *>(this), "__call__");
            if (!f) py::pybind11_fail("TrajectorySurfaceCollisionCallback subclass must define __call__");
            f(cast_particle(particle, mode), cast_particlep(x, mode), tri, s, t);
            return 0;
        });
    }
};

// Register a callback on a database, telling a Python trampoline the geometry mode.
template <typename CB, typename PyCB>
static void register_callback(ParticleDataBase &pdb, CB *cb, void (ParticleDataBase::*setter)(CB *)) {
    if (auto *p = dynamic_cast<PyCB *>(cb)) p->mode = pdb.geom_mode();
    (pdb.*setter)(cb);
}

// ---------------------------------------------------------------------------
// Trampoline: Random_Variate_Transformation subclassable from Python
// ---------------------------------------------------------------------------
class PyRandom_Variate_Transformation : public Random_Variate_Transformation {
public:
    Random_Variate_Transformation *copy() const override {
        PYBIND11_OVERRIDE_PURE(Random_Variate_Transformation*, Random_Variate_Transformation, copy,);
    }
    double transform(double R) override {
        PYBIND11_OVERRIDE_PURE(double, Random_Variate_Transformation, transform, R);
    }
};

// ---------------------------------------------------------------------------
// Helper: convert Python list[6] → field_extrpl_e[6]
// ---------------------------------------------------------------------------
static void list_to_extrpl(const py::list &lst, field_extrpl_e e[6]) {
    for (int i = 0; i < 6; i++) e[i] = lst[i].cast<field_extrpl_e>();
}

// ---------------------------------------------------------------------------
// Helper: convert Python list[6] → bool[6]
// ---------------------------------------------------------------------------
static void list_to_mirror(const py::list &lst, bool m[6]) {
    for (int i = 0; i < 6; i++) m[i] = lst[i].cast<bool>();
}

// ---------------------------------------------------------------------------
// Helper: convert Python list[3] → bool[3]
// ---------------------------------------------------------------------------
static void list_to_fout(const py::list &lst, bool fout[3]) {
    for (int i = 0; i < 3; i++) fout[i] = lst[i].cast<bool>();
}

// ---------------------------------------------------------------------------
// Helpers: numpy views of mesh-shaped data
//
// IBSimu stores node data with the x index varying fastest
// (index = i + (j + k*ny)*nx).  The arrays returned here carry that layout
// as Fortran-order strides, so a[i, j, k] is node (i, j, k) with no copying.
// The trailing dimensions are dropped for 1D and 2D/cylindrical meshes.
// ---------------------------------------------------------------------------
static std::vector<py::ssize_t> mesh_shape(const Mesh &m) {
    switch (m.geom_mode()) {
    case MODE_1D:  return { (py::ssize_t)m.size(0) };
    case MODE_2D:
    case MODE_CYL: return { (py::ssize_t)m.size(0), (py::ssize_t)m.size(1) };
    default:       return { (py::ssize_t)m.size(0), (py::ssize_t)m.size(1),
                            (py::ssize_t)m.size(2) };
    }
}

static std::vector<py::ssize_t> mesh_strides(const Mesh &m, py::ssize_t itemsize,
                                             py::ssize_t ncomp = 0) {
    // With ncomp > 0 a trailing component axis (contiguous) is appended.
    std::vector<py::ssize_t> st;
    py::ssize_t s = itemsize * (ncomp > 0 ? ncomp : 1);
    for (size_t d = 0; d < mesh_shape(m).size(); d++) {
        st.push_back(s);
        s *= m.size((int)d);
    }
    if (ncomp > 0) st.push_back(itemsize);
    return st;
}

template <typename T>
static py::array_t<T> mesh_view(const Mesh &m, T *ptr, py::handle base) {
    if (m.nodecount() == 0 || ptr == nullptr)
        throw std::runtime_error("Field has no mesh data");
    return py::array_t<T>(mesh_shape(m), mesh_strides(m, sizeof(T)), ptr, base);
}

static py::array_t<double> vector_field_to_numpy(const MeshVectorField &f) {
    std::vector<py::ssize_t> shape = mesh_shape(f);
    shape.push_back(3);
    py::array_t<double> arr(shape, mesh_strides(f, sizeof(double), 3));
    double *out = arr.mutable_data();
    for (uint32_t i = 0; i < f.nodecount(); i++) {
        Vec3D v = f(i);
        out[3*i] = v[0]; out[3*i+1] = v[1]; out[3*i+2] = v[2];
    }
    return arr;
}

static void vector_field_from_numpy(MeshVectorField &f,
        py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    std::vector<py::ssize_t> shape = mesh_shape(f);
    shape.push_back(3);
    if ((size_t)arr.ndim() != shape.size())
        throw std::invalid_argument("Array has wrong number of dimensions");
    for (size_t d = 0; d < shape.size(); d++)
        if (arr.shape(d) != shape[d])
            throw std::invalid_argument("Array shape does not match the field mesh");
    // C-contiguous with x as the first axis: node (i,j,k) sits at
    // ((i*ny + j)*nz + k)*3, which is not the field's own ordering.
    const uint32_t nx = f.size(0), ny = f.size(1), nz = f.size(2);
    const double *in = arr.data();
    for (uint32_t k = 0; k < nz; k++)
        for (uint32_t j = 0; j < ny; j++)
            for (uint32_t i = 0; i < nx; i++) {
                const double *v = in + ((size_t)(i*ny + j)*nz + k)*3;
                f.set((int32_t)(i + (j + k*ny)*nx), Vec3D(v[0], v[1], v[2]));
            }
}

// Trajectory of one particle as an (npoints, ncoord) array.
template <typename P, typename PP>
static py::array_t<double> trajectory_array(const P &p) {
    const size_t n = p.traj_size();
    const size_t m = PP().size();
    py::array_t<double> arr({ (py::ssize_t)n, (py::ssize_t)m });
    auto a = arr.template mutable_unchecked<2>();
    for (size_t j = 0; j < n; j++)
        for (size_t k = 0; k < m; k++)
            a(j, k) = p.traj((int)j)((int)k);
    return arr;
}

static py::array_t<double> vector_to_numpy(const std::vector<double> &v) {
    return py::array_t<double>((py::ssize_t)v.size(), v.data());
}

// ===========================================================================
PYBIND11_MODULE(ibsimu, m) {
    m.doc() = "Python wrapper for IBSimu";

    // -----------------------------------------------------------------------
    // Exceptions: map the library's Error hierarchy onto Python exceptions
    // -----------------------------------------------------------------------
    static py::exception<Error> ibsimu_error(m, "IBSimuError", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (ErrorRange &e) {
            PyErr_SetString(PyExc_IndexError, e.get_error_message().c_str());
        } catch (ErrorDim &e) {
            PyErr_SetString(PyExc_ValueError, e.get_error_message().c_str());
        } catch (ErrorUnimplemented &e) {
            PyErr_SetString(PyExc_NotImplementedError, e.get_error_message().c_str());
        } catch (ErrorNoMem &e) {
            PyErr_SetString(PyExc_MemoryError, e.get_error_message().c_str());
        } catch (Error &e) {
            std::string msg = e.get_error_message();
            if (msg.empty()) msg = "IBSimu error";
            ibsimu_error(msg.c_str());
        }
    });

    // -----------------------------------------------------------------------
    // Physical constants
    // -----------------------------------------------------------------------
    m.attr("EPSILON0")  = EPSILON0;
    m.attr("MASS_U")    = MASS_U;
    m.attr("MASS_E")    = MASS_E;
    m.attr("MASS_P")    = MASS_P;
    m.attr("CHARGE_E")  = CHARGE_E;
    m.attr("SPEED_C")   = SPEED_C;
    m.attr("SPEED_C2")  = SPEED_C2;

    // -----------------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------------
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

    py::enum_<field_type_e>(m, "FieldType")
        .value("FIELD_NONE", FIELD_NONE)
        .value("FIELD_EPOT", FIELD_EPOT)
        .value("FIELD_SCHARGE", FIELD_SCHARGE)
        .value("FIELD_TRAJDENS", FIELD_TRAJDENS)
        .value("FIELD_EFIELD", FIELD_EFIELD)
        .value("FIELD_EFIELD_X", FIELD_EFIELD_X)
        .value("FIELD_EFIELD_Y", FIELD_EFIELD_Y)
        .value("FIELD_EFIELD_Z", FIELD_EFIELD_Z)
        .value("FIELD_BFIELD", FIELD_BFIELD)
        .value("FIELD_BFIELD_X", FIELD_BFIELD_X)
        .value("FIELD_BFIELD_Y", FIELD_BFIELD_Y)
        .value("FIELD_BFIELD_Z", FIELD_BFIELD_Z)
        .export_values();

    py::enum_<field_loc_type_e>(m, "FieldLocType")
        .value("FIELDD_LOC_NONE", FIELDD_LOC_NONE)
        .value("FIELDD_LOC_X",    FIELDD_LOC_X)
        .value("FIELDD_LOC_Y",    FIELDD_LOC_Y)
        .value("FIELDD_LOC_Z",    FIELDD_LOC_Z)
        .value("FIELDD_LOC_DIST", FIELDD_LOC_DIST)
        .export_values();

    py::enum_<trajectory_diagnostic_e>(m, "TrajectoryDiagnostic")
        .value("DIAG_NONE",   DIAG_NONE)
        .value("DIAG_T",      DIAG_T)
        .value("DIAG_X",      DIAG_X)
        .value("DIAG_VX",     DIAG_VX)
        .value("DIAG_Y",      DIAG_Y)
        .value("DIAG_R",      DIAG_R)
        .value("DIAG_VY",     DIAG_VY)
        .value("DIAG_VR",     DIAG_VR)
        .value("DIAG_W",      DIAG_W)
        .value("DIAG_VTHETA", DIAG_VTHETA)
        .value("DIAG_Z",      DIAG_Z)
        .value("DIAG_VZ",     DIAG_VZ)
        .value("DIAG_O",      DIAG_O)
        .value("DIAG_VO",     DIAG_VO)
        .value("DIAG_P",      DIAG_P)
        .value("DIAG_VP",     DIAG_VP)
        .value("DIAG_Q",      DIAG_Q)
        .value("DIAG_VQ",     DIAG_VQ)
        .value("DIAG_XP",     DIAG_XP)
        .value("DIAG_YP",     DIAG_YP)
        .value("DIAG_RP",     DIAG_RP)
        .value("DIAG_AP",     DIAG_AP)
        .value("DIAG_ZP",     DIAG_ZP)
        .value("DIAG_OP",     DIAG_OP)
        .value("DIAG_PP",     DIAG_PP)
        .value("DIAG_CURR",   DIAG_CURR)
        .value("DIAG_EK",     DIAG_EK)
        .value("DIAG_QM",     DIAG_QM)
        .value("DIAG_CHARGE", DIAG_CHARGE)
        .value("DIAG_MASS",   DIAG_MASS)
        .value("DIAG_NO",     DIAG_NO)
        .export_values();

    py::enum_<plasma_mode_e>(m, "PlasmaMode")
        .value("PLASMA_NONE",          PLASMA_NONE)
        .value("PLASMA_PEXP_INITIAL",  PLASMA_PEXP_INITIAL)
        .value("PLASMA_NSIMP_INITIAL", PLASMA_NSIMP_INITIAL)
        .value("PLASMA_PEXP",          PLASMA_PEXP)
        .value("PLASMA_NSIMP",         PLASMA_NSIMP)
        .value("PLASMA_SHIELD",        PLASMA_SHIELD)
        .export_values();

    py::enum_<particle_status_e>(m, "ParticleStatus")
        .value("PARTICLE_OK",     PARTICLE_OK)
        .value("PARTICLE_OUT",    PARTICLE_OUT)
        .value("PARTICLE_COLL",   PARTICLE_COLL)
        .value("PARTICLE_BADDEF", PARTICLE_BADDEF)
        .value("PARTICLE_TIME",   PARTICLE_TIME)
        .value("PARTICLE_NSTP",   PARTICLE_NSTP)
        .export_values();

    py::enum_<scharge_deposition_e>(m, "SchargeDeposition")
        .value("SCHARGE_DEPOSITION_PIC",    SCHARGE_DEPOSITION_PIC)
        .value("SCHARGE_DEPOSITION_LINEAR", SCHARGE_DEPOSITION_LINEAR)
        .export_values();

    py::enum_<trajectory_interpolation_e>(m, "TrajectoryInterpolation")
        .value("TRAJECTORY_INTERPOLATION_POLYNOMIAL", TRAJECTORY_INTERPOLATION_POLYNOMIAL)
        .value("TRAJECTORY_INTERPOLATION_LINEAR",     TRAJECTORY_INTERPOLATION_LINEAR)
        .export_values();

    py::enum_<message_type_e>(m, "MessageType")
        .value("MSG_VERBOSE",       MSG_VERBOSE)
        .value("MSG_WARNING",       MSG_WARNING)
        .value("MSG_ERROR",         MSG_ERROR)
        .value("MSG_DEBUG_GENERAL", MSG_DEBUG_GENERAL)
        .value("MSG_DEBUG_DXF",     MSG_DEBUG_DXF)
        .export_values();

    py::enum_<rng_type_e>(m, "RngType")
        .value("RNG_SOBOL", RNG_SOBOL)
        .value("RNG_MT",    RNG_MT)
        .export_values();

    py::enum_<histogram_accumulation_e>(m, "HistogramAccumulation")
        .value("HISTOGRAM_ACCUMULATION_CLOSEST", HISTOGRAM_ACCUMULATION_CLOSEST)
        .value("HISTOGRAM_ACCUMULATION_LINEAR",  HISTOGRAM_ACCUMULATION_LINEAR)
        .export_values();

    py::enum_<interpolation_e>(m, "Interpolation")
        .value("INTERPOLATION_CLOSEST",  INTERPOLATION_CLOSEST)
        .value("INTERPOLATION_BILINEAR", INTERPOLATION_BILINEAR)
        .value("INTERPOLATION_BICUBIC",  INTERPOLATION_BICUBIC)
        .export_values();

    py::enum_<view_e>(m, "View")
        .value("VIEW_XY", VIEW_XY)
        .value("VIEW_XZ", VIEW_XZ)
        .value("VIEW_YX", VIEW_YX)
        .value("VIEW_YZ", VIEW_YZ)
        .value("VIEW_ZX", VIEW_ZX)
        .value("VIEW_ZY", VIEW_ZY)
        .export_values();

    py::enum_<zscale_e>(m, "ZScale")
        .value("ZSCALE_LINEAR", ZSCALE_LINEAR)
        .value("ZSCALE_LOG",    ZSCALE_LOG)
        .value("ZSCALE_RELLOG", ZSCALE_RELLOG)
        .export_values();

    py::enum_<particle_diag_plot_type_e>(m, "ParticleDiagPlotType")
        .value("PARTICLE_DIAG_PLOT_SCATTER", PARTICLE_DIAG_PLOT_SCATTER)
        .value("PARTICLE_DIAG_PLOT_HISTO1D", PARTICLE_DIAG_PLOT_HISTO1D)
        .value("PARTICLE_DIAG_PLOT_HISTO2D", PARTICLE_DIAG_PLOT_HISTO2D)
        .export_values();

    // -----------------------------------------------------------------------
    // Vec3D
    // -----------------------------------------------------------------------
    py::class_<Vec3D>(m, "Vec3D")
        .def(py::init<double, double, double>(), py::arg("x")=0, py::arg("y")=0, py::arg("z")=0)
        .def_property("x", [](const Vec3D &v){ return v[0]; }, [](Vec3D &v, double x){ v[0]=x; })
        .def_property("y", [](const Vec3D &v){ return v[1]; }, [](Vec3D &v, double y){ v[1]=y; })
        .def_property("z", [](const Vec3D &v){ return v[2]; }, [](Vec3D &v, double z){ v[2]=z; })
        .def("__getitem__", [](const Vec3D &v, int i){ return v[i]; })
        .def("__setitem__", [](Vec3D &v, int i, double val){ v[i]=val; })
        .def("__len__",    [](const Vec3D &v){ return 3; })
        .def("__iter__",   [](const Vec3D &v){
            return py::make_iterator(&v[0], &v[0] + 3);
        }, py::keep_alive<0, 1>())
        .def("__add__",  [](const Vec3D &a, const Vec3D &b){ return a+b; })
        .def("__sub__",  [](const Vec3D &a, const Vec3D &b){ return a-b; })
        .def("__mul__",  [](const Vec3D &a, double s){ return a*s; })
        .def("__rmul__", [](const Vec3D &a, double s){ return a*s; })
        .def("__truediv__", [](const Vec3D &a, double s){
            return Vec3D(a[0]/s, a[1]/s, a[2]/s); })
        .def("__neg__",  [](const Vec3D &a){ return -a; })
        .def("__eq__",   [](const Vec3D &a, const Vec3D &b){ return a==b; })
        .def("__ne__",   [](const Vec3D &a, const Vec3D &b){ return a!=b; })
        .def("dot",      [](const Vec3D &a, const Vec3D &b){ return a*b; },
             "Dot product with another Vec3D")
        .def("norm2",    &Vec3D::norm2, "2-norm (magnitude)")
        .def("ssqr",     &Vec3D::ssqr, "Sum of squares")
        .def("normalize",&Vec3D::normalize)
        .def("abs",      [](Vec3D &v){ v.abs(); })
        .def("approx",   &Vec3D::approx, py::arg("x"), py::arg("eps")=1e-6)
        .def("__repr__",  [](const Vec3D &v){
            std::ostringstream oss; oss << v; return "Vec3D(" + oss.str() + ")";
        });

    m.def("dot",   [](const Vec3D &a, const Vec3D &b){ return a*b; });
    m.def("cross", [](const Vec3D &a, const Vec3D &b){ return cross(a,b); });

    // -----------------------------------------------------------------------
    // Int3D
    // -----------------------------------------------------------------------
    py::class_<Int3D>(m, "Int3D")
        .def(py::init<int32_t, int32_t, int32_t>(), py::arg("i")=0, py::arg("j")=0, py::arg("k")=0)
        .def("__getitem__", [](const Int3D &v, int i){ return v[i]; })
        .def("__setitem__", [](Int3D &v, int i, int32_t val){ v[i]=val; })
        .def("__len__",     [](const Int3D &v){ return 3; })
        .def("__iter__",    [](const Int3D &v){
            return py::make_iterator(&v[0], &v[0] + 3);
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Int3D &a, const Int3D &b){ return a==b; })
        .def("__ne__", [](const Int3D &a, const Int3D &b){ return a!=b; })
        .def("__repr__", [](const Int3D &v){
            std::ostringstream oss; oss << v; return "Int3D(" + oss.str() + ")";
        });

    // -----------------------------------------------------------------------
    // Vec4D
    // -----------------------------------------------------------------------
    py::class_<Vec4D>(m, "Vec4D")
        .def(py::init<double,double,double,double>())
        .def("__getitem__", [](const Vec4D &v, int i){ return v[i]; })
        .def("__setitem__", [](Vec4D &v, int i, double val){ v[i]=val; })
        .def("__add__",  [](const Vec4D &a, const Vec4D &b){ return a+b; })
        .def("__sub__",  [](const Vec4D &a, const Vec4D &b){ return a-b; })
        .def("__mul__",  [](const Vec4D &a, double s){ return a*s; })
        .def("dot",      [](const Vec4D &a, const Vec4D &b){ return a*b; })
        .def("norm2",    &Vec4D::norm2)
        .def("__repr__", [](const Vec4D &v){
            return "Vec4D(" + std::to_string(v[0]) + ", "
                            + std::to_string(v[1]) + ", "
                            + std::to_string(v[2]) + ", "
                            + std::to_string(v[3]) + ")";
        });

    // -----------------------------------------------------------------------
    // Mat3D
    // -----------------------------------------------------------------------
    py::class_<Mat3D>(m, "Mat3D")
        .def(py::init<>())
        .def(py::init<double,double,double,double,double,double,double,double,double>())
        .def("__call__", [](const Mat3D &m, int i, int j){ return m(i,j); },
             "Access element at (row, col)")
        .def("mul_vec",  [](const Mat3D &a, const Vec3D &v){ return a*v; },
             "Matrix-vector product")
        .def("determinant", &Mat3D::determinant)
        .def("inverse",     [](const Mat3D &m){ return m.inverse(); });

    // -----------------------------------------------------------------------
    // Transformation
    // -----------------------------------------------------------------------
    py::class_<Transformation>(m, "Transformation")
        .def(py::init<>())
        .def("__getitem__", [](const Transformation &t, int i){ return t[i]; })
        .def("__setitem__", [](Transformation &t, int i, double v){ t[i]=v; })
        .def("__mul__",     [](const Transformation &a, const Transformation &b){ return a*b; })
        .def("mul_vec4",    [](const Transformation &t, const Vec4D &v){ return t*v; })
        .def("transpose",   &Transformation::transpose)
        .def("determinant", &Transformation::determinant)
        .def("inverse",     &Transformation::inverse)
        .def("transform",         &Transformation::transform)
        .def("transform_point",   &Transformation::transform_point)
        .def("inv_transform_point",&Transformation::inv_transform_point)
        .def("transform_vector",  &Transformation::transform_vector)
        .def("inv_transform_vector",&Transformation::inv_transform_vector)
        .def("reset",           &Transformation::reset)
        .def("translate",       [](Transformation &t, const Vec3D &d){ t.translate(d); })
        .def("translate_before",[](Transformation &t, const Vec3D &d){ t.translate_before(d); })
        .def("scale",           [](Transformation &t, const Vec3D &s){ t.scale(s); })
        .def("scale_before",    [](Transformation &t, const Vec3D &s){ t.scale_before(s); })
        .def("rotate_x",        [](Transformation &t, double a){ t.rotate_x(a); })
        .def("rotate_x_before", [](Transformation &t, double a){ t.rotate_x_before(a); })
        .def("rotate_y",        [](Transformation &t, double a){ t.rotate_y(a); })
        .def("rotate_y_before", [](Transformation &t, double a){ t.rotate_y_before(a); })
        .def("rotate_z",        [](Transformation &t, double a){ t.rotate_z(a); })
        .def("rotate_z_before", [](Transformation &t, double a){ t.rotate_z_before(a); })
        .def("save", [](const Transformation &t, const std::string &f){ t.save(f); })
        .def_static("unity",       &Transformation::unity)
        .def_static("translation", &Transformation::translation)
        .def_static("scaling",     &Transformation::scaling)
        .def_static("rotation_x",  &Transformation::rotation_x)
        .def_static("rotation_y",  &Transformation::rotation_y)
        .def_static("rotation_z",  &Transformation::rotation_z);

    // -----------------------------------------------------------------------
    // Mesh (base class, read-only queries)
    // -----------------------------------------------------------------------
    py::class_<Mesh>(m, "Mesh")
        .def("geom_mode",  &Mesh::geom_mode)
        .def("size",       [](const Mesh &me, int i){ return me.size(i); })
        .def("nodecount",  &Mesh::nodecount)
        .def("h",          &Mesh::h)
        .def("origo",      [](const Mesh &me, int i){ return me.origo(i); })
        .def("origo_vec",  [](const Mesh &me){ return me.origo(); })
        .def("max",        [](const Mesh &me, int i){ return me.max(i); })
        .def("max_vec",    [](const Mesh &me){ return me.max(); })
        .def("size_vec",   [](const Mesh &me){ return me.size(); })
        .def("shape",      [](const Mesh &me){
            py::tuple t(mesh_shape(me).size());
            size_t d = 0;
            for (auto n : mesh_shape(me)) t[d++] = n;
            return t;
        }, "Shape of the node arrays returned by numpy(): (nx,), (nx, ny) or (nx, ny, nz)")
        .def("closest_node",  &Mesh::closest_node)
        .def("mesh_number",   &Mesh::mesh_number)
        .def("coord_of_node", &Mesh::coord_of_node)
        .def("node_coordinates", [](const Mesh &me){
            // Coordinate axes of the mesh nodes, one 1D array per dimension.
            py::list axes;
            for (size_t d = 0; d < mesh_shape(me).size(); d++) {
                py::array_t<double> ax((py::ssize_t)me.size((int)d));
                double *p = ax.mutable_data();
                for (uint32_t i = 0; i < me.size((int)d); i++) p[i] = me.origo((int)d) + i*me.h();
                axes.append(ax);
            }
            return py::tuple(axes);
        }, "Node coordinate axes as a tuple of 1D arrays (x, [y|r, [z]])");

    // -----------------------------------------------------------------------
    // Bound
    // -----------------------------------------------------------------------
    py::class_<Bound>(m, "Bound")
        .def(py::init<bound_e, double>(), py::arg("type"), py::arg("value"))
        .def(py::init<bound_e, const CallbackFunctorD_V *>(), py::arg("type"), py::arg("functor"),
             py::keep_alive<1, 3>(), "Boundary whose value varies with position via a functor")
        .def("type",  &Bound::type)
        .def("value", [](const Bound &b){ return b.value(); })
        .def("value_at", [](const Bound &b, const Vec3D &x){
            double v = b.value(x); rethrow_python_error(); return v;
        })
        .def("set_value", &Bound::set_value)
        .def("is_constant", &Bound::is_constant)
        .def("__repr__", [](const Bound &b){
            std::ostringstream oss; oss << b; return "Bound(" + oss.str() + ")";
        });

    // -----------------------------------------------------------------------
    // Geometry
    // -----------------------------------------------------------------------
    py::class_<Geometry, Mesh>(m, "Geometry")
        .def(py::init<geom_mode_e, Int3D, Vec3D, double>(),
             py::arg("geom_mode"), py::arg("size"), py::arg("origo"), py::arg("h"))
        .def(py::init([](const std::string &filename){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new Geometry(is);
        }), py::arg("filename"), "Load a geometry saved with Geometry.save()")
        // Geometry takes ownership of the solid (it deletes the old one when a
        // solid number is redefined), so the Python side must stop owning it.
        // keep_alive additionally ties the wrapper's lifetime to the geometry,
        // which matters for FuncSolid, where the wrapper holds the callable.
        .def("set_solid",   [](Geometry &g, uint32_t n, py::object solid){
            const Solid *s = solid.cast<const Solid *>();
            reinterpret_cast<py::detail::instance *>(solid.ptr())->owned = false;
            g.set_solid(n, s);
        }, py::keep_alive<1, 3>(), py::arg("n"), py::arg("solid"),
             "Assign solid number n (>= 7); the geometry takes ownership of the solid")
        .def("get_solid",   &Geometry::get_solid, py::return_value_policy::reference, py::arg("n"))
        .def("set_boundary",&Geometry::set_boundary, py::arg("n"), py::arg("b"))
        .def("get_boundary",&Geometry::get_boundary, py::arg("n"))
        // Geometry::get_boundaries() is declared but not implemented in the library.
        .def("get_boundaries", [](const Geometry &g){
            std::vector<Bound> b;
            for (uint32_t i = 1; i <= g.number_of_boundaries(); i++) b.push_back(g.get_boundary(i));
            return b;
        }, "List of boundary conditions, index 0 is boundary number 1")
        .def("number_of_solids",     &Geometry::number_of_solids)
        .def("number_of_boundaries", &Geometry::number_of_boundaries)
        .def("have_solid_data",      &Geometry::have_solid_data)
        .def("build_mesh",  [](Geometry &g){
            { py::gil_scoped_release release; g.build_mesh(); }
            rethrow_python_error();
        })
        .def("build_surface",[](Geometry &g){
            { py::gil_scoped_release release; g.build_surface(); }
            rethrow_python_error();
        })
        .def("surface_built",&Geometry::surface_built)
        .def("inside",      [](const Geometry &g, const Vec3D &x){
            uint32_t r = g.inside(x); rethrow_python_error(); return r;
        })
        .def("inside_n",    [](const Geometry &g, uint32_t n, const Vec3D &x){
            bool r = g.inside(n, x); rethrow_python_error(); return r;
        })
        .def("bracket_surface", [](const Geometry &g, uint32_t n, const Vec3D &xin, const Vec3D &xout){
            Vec3D xsurf;
            double s = g.bracket_surface(n, xin, xout, xsurf);
            rethrow_python_error();
            return py::make_tuple(s, xsurf);
        })
        .def("surface_normal", &Geometry::surface_normal)
        .def("built",          &Geometry::built)
        .def("mesh",           [](const Geometry &g, uint32_t i){ return g.mesh(i); })
        .def("mesh3",          [](const Geometry &g, uint32_t i, uint32_t j, uint32_t k){ return g.mesh(i,j,k); })
        .def("mesh_check1",    [](const Geometry &g, int32_t i){ return g.mesh_check(i); })
        .def("mesh_check2",    [](const Geometry &g, int32_t i, int32_t j){ return g.mesh_check(i,j); })
        .def("mesh_check3",    [](const Geometry &g, int32_t i, int32_t j, int32_t k){ return g.mesh_check(i,j,k); })
        .def("is_near_solid",  &Geometry::is_near_solid)
        .def("mesh_numpy", [](py::object self){
            const Geometry &g = self.cast<const Geometry &>();
            if (!g.built()) throw std::runtime_error("Mesh not built");
            py::array_t<uint32_t> a = mesh_view<uint32_t>(g, const_cast<uint32_t *>(&g.mesh(0)), self);
            py::detail::array_proxy(a.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
            return a;
        }, "Read-only view of the solid mesh (node type flags | boundary number, see geometry.hpp)")
        .def("solid_numpy", [](const Geometry &g){
            // Boundary/solid number of each node: 0 vacuum, 1-6 mesh boundaries,
            // >= 7 solids.  Vacuum nodes near a solid are 0 as well.
            if (!g.built()) throw std::runtime_error("Mesh not built");
            py::array_t<uint32_t> a(mesh_shape(g), mesh_strides(g, sizeof(uint32_t)));
            uint32_t *out = a.mutable_data();
            for (uint32_t i = 0; i < g.nodecount(); i++) {
                uint32_t v = g.mesh((int32_t)i);
                uint32_t id = v & SMESH_NODE_ID_MASK;
                out[i] = (id == SMESH_NODE_ID_DIRICHLET || id == SMESH_NODE_ID_NEUMANN)
                         ? (v & SMESH_BOUNDARY_NUMBER_MASK) : 0;
            }
            return a;
        }, "Boundary/solid number of every node as an array (0 = vacuum)")
        .def("surface_vertexc",&Geometry::surface_vertexc)
        .def("surface_vertex", &Geometry::surface_vertex, py::return_value_policy::reference_internal)
        .def("surface_triangle_normal", (Vec3D (Geometry::*)(const Vec3D &) const) &Geometry::surface_triangle_normal)
        .def("surface_triangle_normal_idx", (Vec3D (Geometry::*)(int32_t) const) &Geometry::surface_triangle_normal)
        .def("surface_triangle", &Geometry::surface_triangle, py::return_value_policy::reference_internal)
        .def("surface_triangle_ptr", &Geometry::surface_triangle_ptr)
        .def("surface_trianglec",      [](const Geometry &g){ return g.surface_trianglec(); })
        .def("surface_trianglec3",     [](const Geometry &g, int32_t i, int32_t j, int32_t k){ return g.surface_trianglec(i,j,k); })
        .def("save", [](const Geometry &g, const std::string &f, bool save_solids){
            g.save(f, save_solids);
        }, py::arg("filename"), py::arg("save_solids")=false);

    // -----------------------------------------------------------------------
    // Solid hierarchy
    // -----------------------------------------------------------------------
    py::class_<Solid, std::unique_ptr<Solid, py::nodelete>>(m, "Solid")
        .def("inside", [](const Solid &s, const Vec3D &x){
            bool r = s.inside(x); rethrow_python_error(); return r;
        })
        .def("reset_transformation", &Solid::reset_transformation)
        .def("set_transformation",   &Solid::set_transformation)
        .def("translate", &Solid::translate, py::arg("dx"))
        .def("scale",     (void (Solid::*)(double)) &Solid::scale)
        .def("scale_v",   (void (Solid::*)(const Vec3D &)) &Solid::scale)
        .def("rotate_x",  &Solid::rotate_x, py::arg("a"))
        .def("rotate_y",  &Solid::rotate_y, py::arg("a"))
        .def("rotate_z",  &Solid::rotate_z, py::arg("a"));

    py::class_<PySolid, Solid>(m, "PySolid");

    m.def("FuncSolid", [](py::function func) -> Solid* {
        return new PySolid(func);
    }, py::return_value_policy::take_ownership,
       "Create a Solid whose inside() is defined by a Python callable f(x,y,z)->bool");

    py::class_<STLSolid, Solid>(m, "STLSolid")
        .def(py::init<const std::string &>())
        .def("inside",    &STLSolid::inside)
        .def("save",      [](const STLSolid &s, const std::string &fn){
            std::ofstream os(fn); s.save(os);
        });

    py::class_<MyDXFFile>(m, "MyDXFFile")
        .def(py::init<>())
        .def(py::init<const std::string &>())
        .def("read",  &MyDXFFile::read)
        .def("write", &MyDXFFile::write)
        .def("set_warning_level", &MyDXFFile::set_warning_level);

    py::class_<DXFSolid, Solid>(m, "DXFSolid")
        .def(py::init([](const std::string &filename){
            std::ifstream is(filename); return new DXFSolid(is);
        }))
        .def(py::init<MyDXFFile*, const std::string &>())
        .def("inside",    &DXFSolid::inside)
        .def("define_2x3_mapping", [](DXFSolid &s, const std::string &mname){
            if(mname == "unity") s.define_2x3_mapping(DXFSolid::unity);
            else if(mname == "rotx") s.define_2x3_mapping(DXFSolid::rotx);
            else if(mname == "roty") s.define_2x3_mapping(DXFSolid::roty);
            else if(mname == "rotz") s.define_2x3_mapping(DXFSolid::rotz);
            else throw std::runtime_error("Unknown mapping name: " + mname);
        });

#ifdef CSG
    py::class_<CSGObject>(m, "CSGObject");
    py::class_<CSGObjectSolid, Solid>(m, "CSGObjectSolid")
        .def(py::init<CSGObject*>())
        .def("inside", &CSGObjectSolid::inside);
#endif

    // -----------------------------------------------------------------------
    // Field base classes
    // -----------------------------------------------------------------------
    py::class_<Field>(m, "Field");

    py::class_<ScalarField, Field>(m, "ScalarField")
        .def("__call__", [](const ScalarField &f, const Vec3D &x){ return f(x); });

    py::class_<VectorField, Field>(m, "VectorField")
        .def("__call__", [](const VectorField &f, const Vec3D &x){ return f(x); });

    // -----------------------------------------------------------------------
    // MeshScalarField
    // -----------------------------------------------------------------------
    py::class_<MeshScalarField, ScalarField, Mesh>(m, "MeshScalarField")
        .def(py::init<>())
        .def(py::init<const Mesh &>())
        .def(py::init<const MeshScalarField &>())
        .def(py::init([](const std::string &filename){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new MeshScalarField(is);
        }), py::arg("filename"))
        .def("__call__",   [](const MeshScalarField &f, const Vec3D &x){ return f(x); })
        .def("numpy", [](py::object self){
            MeshScalarField &f = self.cast<MeshScalarField &>();
            return mesh_view<double>(f, f.nodecount() ? &f(0) : nullptr, self);
        }, "Writable numpy view of the node data, shape (nx,), (nx, ny) or (nx, ny, nz)")
        .def("__array__", [](py::object self, py::object dtype, py::object copy){
            MeshScalarField &f = self.cast<MeshScalarField &>();
            py::array a = mesh_view<double>(f, f.nodecount() ? &f(0) : nullptr, self);
            if (!dtype.is_none()) a = a.attr("astype")(dtype);
            else if (copy.is(py::bool_(true))) a = a.attr("copy")();
            return a;
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none())
        .def("set_numpy", [](MeshScalarField &f, py::array_t<double> arr){
            py::array_t<double> view = mesh_view<double>(f, f.nodecount() ? &f(0) : nullptr, py::none());
            view.attr("__setitem__")(py::ellipsis(), arr);
        }, py::arg("array"), "Copy array values into the field (broadcasting allowed)")
        .def("get",        [](const MeshScalarField &f, uint32_t i){ return f(i); })
        .def("set",        [](MeshScalarField &f, uint32_t i, double v){ f(i) = v; })
        .def("get3",       [](const MeshScalarField &f, uint32_t i, uint32_t j, uint32_t k){ return f(i,j,k); })
        .def("set3",       [](MeshScalarField &f, uint32_t i, uint32_t j, uint32_t k, double v){ f(i,j,k) = v; })
        .def("clear",      &MeshScalarField::clear)
        .def("reset",      [](MeshScalarField &f, geom_mode_e mode, Int3D size, Vec3D origo, double h){ f.reset(mode, size, origo, h); })
        .def("get_minmax",  [](const MeshScalarField &f){ double mn, mx; f.get_minmax(mn, mx); return py::make_tuple(mn, mx); })
        .def("save", [](const MeshScalarField &f, const std::string &fn){ f.save(fn); })
        .def("__add__", [](MeshScalarField &a, const MeshScalarField &b){
            MeshScalarField res(a); res += b; return res;
        })
        .def("__iadd__", [](MeshScalarField &a, const MeshScalarField &b){ a += b; return a; })
        .def("__sub__", [](MeshScalarField &a, const MeshScalarField &b){
            MeshScalarField res(a); res -= b; return res;
        })
        .def("__isub__", [](MeshScalarField &a, const MeshScalarField &b){ a -= b; return a; })
        .def("__mul__", [](MeshScalarField &a, double s){
            MeshScalarField res(a); res *= s; return res;
        })
        .def("__rmul__", [](MeshScalarField &a, double s){
            MeshScalarField res(a); res *= s; return res;
        })
        .def("__imul__", [](MeshScalarField &a, double s){ a *= s; return a; })
        .def("__truediv__", [](MeshScalarField &a, double s){
            MeshScalarField res(a); res /= s; return res;
        })
        .def("__itruediv__", [](MeshScalarField &a, double s){ a /= s; return a; });

    // -----------------------------------------------------------------------
    // EpotField
    // -----------------------------------------------------------------------
    py::class_<EpotField, MeshScalarField>(m, "EpotField")
        .def(py::init<const Geometry &>(), py::arg("geom"))
        .def(py::init<const EpotField &>())
        .def(py::init([](const std::string &filename, const Geometry &geom){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new EpotField(is, geom);
        }), py::arg("filename"), py::arg("geom"), "Load a potential saved with save()");

    // -----------------------------------------------------------------------
    // MeshVectorField
    // -----------------------------------------------------------------------
    py::class_<MeshVectorField, VectorField, Mesh>(m, "MeshVectorField")
        .def(py::init<>())
        .def(py::init([](const Mesh &mesh, py::list fout){
            bool fo[3]; list_to_fout(fout, fo);
            return new MeshVectorField(mesh, fo);
        }))
        .def(py::init([](geom_mode_e mode, py::list fout, Int3D size, Vec3D origo, double h){
            bool fo[3]; list_to_fout(fout, fo);
            return new MeshVectorField(mode, fo, size, origo, h);
        }))
        .def(py::init([](geom_mode_e mode, py::list fout, double xscale, double fscale,
                         const std::string &filename){
            bool fo[3]; list_to_fout(fout, fo);
            return new MeshVectorField(mode, fo, xscale, fscale, filename);
        }))
        .def(py::init([](geom_mode_e mode, py::list fout, Int3D size, Vec3D origo, double h,
                         const MeshVectorField &fin){
            bool fo[3]; list_to_fout(fout, fo);
            return new MeshVectorField(mode, fo, size, origo, h, fin);
        }), py::arg("mode"), py::arg("fout"), py::arg("size"), py::arg("origo"), py::arg("h"),
            py::arg("fin"), "Resample an existing field onto a new mesh")
        .def(py::init<const MeshVectorField &>())
        .def(py::init([](const std::string &filename){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new MeshVectorField(is);
        }), py::arg("filename"))
        .def("__call__", [](const MeshVectorField &f, const Vec3D &x){ return f(x); })
        .def("numpy", &vector_field_to_numpy,
             "Copy of the node vectors, shape (..., 3) matching Mesh.shape()")
        .def("set_numpy", &vector_field_from_numpy, py::arg("array"),
             "Set node vectors from an array of shape (..., 3)")
        .def("reset", [](MeshVectorField &f, geom_mode_e mode, py::list fout, Int3D size,
                         Vec3D origo, double h){
            bool fo[3]; list_to_fout(fout, fo); f.reset(mode, fo, size, origo, h);
        })
        .def("__iadd__", [](MeshVectorField &a, const MeshVectorField &b){ a += b; return a; })
        .def("__imul__", [](MeshVectorField &a, double s){ a *= s; return a; })
        .def("__itruediv__", [](MeshVectorField &a, double s){ a /= s; return a; })
        .def("get1",     [](const MeshVectorField &f, uint32_t i){ return f(i); })
        .def("get2",     [](const MeshVectorField &f, uint32_t i, uint32_t j){ return f(i,j); })
        .def("get3",     [](const MeshVectorField &f, uint32_t i, uint32_t j, uint32_t k){ return f(i,j,k); })
        .def("set1",     (void (MeshVectorField::*)(int32_t, const Vec3D &)) &MeshVectorField::set)
        .def("set2",     (void (MeshVectorField::*)(int32_t, int32_t, const Vec3D &)) &MeshVectorField::set)
        .def("set3",     (void (MeshVectorField::*)(int32_t, int32_t, int32_t, const Vec3D &)) &MeshVectorField::set)
        .def("set_extrapolation", [](MeshVectorField &f, py::list ext){
            field_extrpl_e e[6]; list_to_extrpl(ext, e); f.set_extrapolation(e);
        })
        .def("reset_transformation", &MeshVectorField::reset_transformation)
        .def("set_transformation",   &MeshVectorField::set_transformation)
        .def("translate",   &MeshVectorField::translate)
        .def("scale",       [](MeshVectorField &f, const Vec3D &s){ f.scale(s); })
        .def("rotate_x",    &MeshVectorField::rotate_x)
        .def("rotate_y",    &MeshVectorField::rotate_y)
        .def("rotate_z",    &MeshVectorField::rotate_z)
        .def("clear",       &MeshVectorField::clear)
        .def("get_minmax_scalar", [](const MeshVectorField &f){
            double mn, mx; f.get_minmax(mn, mx); return py::make_tuple(mn, mx);
        })
        .def("get_minmax_vec", [](const MeshVectorField &f){
            Vec3D mn, mx; f.get_minmax(mn, mx); return py::make_tuple(mn, mx);
        })
        .def("get_defined_components", [](const MeshVectorField &f){
            bool fo[3]; f.get_defined_components(fo);
            return py::make_tuple(fo[0], fo[1], fo[2]);
        })
        .def("save", [](const MeshVectorField &f, const std::string &fn){ f.save(fn); });

    // -----------------------------------------------------------------------
    // MultiMeshVectorField
    // -----------------------------------------------------------------------
    py::class_<MultiMeshVectorField, VectorField>(m, "MultiMeshVectorField")
        .def(py::init<>())
        .def("add_mesh",   [](MultiMeshVectorField &f, const MeshVectorField &mf){ f.add_mesh(mf); })
        .def("__call__",   [](const MultiMeshVectorField &f, const Vec3D &x){ return f(x); });

    // -----------------------------------------------------------------------
    // AxisymmetricVectorField
    // -----------------------------------------------------------------------
    py::class_<AxisymmetricVectorField, VectorField>(m, "AxisymmetricVectorField")
        .def(py::init<geom_mode_e, const std::vector<double> &, const std::vector<double> &>(),
             "Construct from z and Bz arrays")
        .def(py::init([](geom_mode_e mode, double origo, double h,
                         const std::vector<double> &Bz, uint32_t order){
            return new AxisymmetricVectorField(mode, origo, h, Bz, order);
        }), py::arg("mode"), py::arg("origo"), py::arg("h"), py::arg("Bz"), py::arg("order")=6u,
            "Construct from uniform grid Bz data")
        .def("__call__", [](const AxisymmetricVectorField &f, const Vec3D &x){ return f(x); });

    // -----------------------------------------------------------------------
    // EpotEfield
    // -----------------------------------------------------------------------
    py::class_<EpotEfield, VectorField>(m, "EpotEfield")
        .def(py::init<EpotField &>())
        .def("recalculate", &EpotEfield::recalculate)
        .def("__call__", [](const EpotEfield &f, const Vec3D &x){ return f(x); })
        .def("set_extrapolation", [](EpotEfield &ef, py::list ext){
            field_extrpl_e e[6]; list_to_extrpl(ext, e); ef.set_extrapolation(e);
        });

    // -----------------------------------------------------------------------
    // Callbacks / Functors
    // -----------------------------------------------------------------------
    py::class_<CallbackFunctor>(m, "CallbackFunctor");

    py::class_<CallbackFunctorD_3D, PyCallbackFunctorD_3D>(m, "CallbackFunctorD_3D")
        .def(py::init<>())
        .def("__call__", [](const CallbackFunctorD_3D &f, double x, double y, double z){
            auto r = f(x, y, z); rethrow_python_error(); return r;
        });

    py::class_<CallbackFunctorD_V, PyCallbackFunctorD_V>(m, "CallbackFunctorD_V")
        .def(py::init<>())
        .def("__call__", [](const CallbackFunctorD_V &f, const Vec3D &x){
            auto r = f(x); rethrow_python_error(); return r;
        });

    py::class_<CallbackFunctorB_3D, PyCallbackFunctorB_3D>(m, "CallbackFunctorB_3D")
        .def(py::init<>())
        .def("__call__", [](const CallbackFunctorB_3D &f, double x, double y, double z){
            auto r = f(x, y, z); rethrow_python_error(); return r;
        });

    py::class_<CallbackFunctorB_V, PyCallbackFunctorB_V>(m, "CallbackFunctorB_V")
        .def(py::init<>())
        .def("__call__", [](const CallbackFunctorB_V &f, const Vec3D &x){
            auto r = f(x); rethrow_python_error(); return r;
        });

    py::class_<CallbackFunctorD_D, PyCallbackFunctorD_D>(m, "CallbackFunctorD_D")
        .def(py::init<>())
        .def("__call__", [](const CallbackFunctorD_D &f, double x){
            auto r = f(x); rethrow_python_error(); return r;
        });

    py::class_<InitialPlasma, CallbackFunctorB_V>(m, "InitialPlasma")
        .def(py::init<coordinate_axis_e, double>());

    py::class_<PPlasmaBfieldSuppression, CallbackFunctorD_V>(m, "PPlasmaBfieldSuppression")
        .def(py::init<const MeshScalarField &, double>());

    py::class_<NPlasmaBfieldSuppression, CallbackFunctorD_V>(m, "NPlasmaBfieldSuppression")
        .def(py::init<const MeshScalarField &, double>());

    // -----------------------------------------------------------------------
    // EpotSolver (abstract base)
    // -----------------------------------------------------------------------
    py::class_<EpotSolver>(m, "EpotSolver")
        .def("solve", [](EpotSolver &s, MeshScalarField &epot, const ScalarField &sc){
            { py::gil_scoped_release release; s.solve(epot, sc); }
            rethrow_python_error();
        }, py::arg("epot"), py::arg("scharge"))
        .def("set_pexp_plasma",   &EpotSolver::set_pexp_plasma, py::arg("rhoe"), py::arg("Te"), py::arg("Up"))
        .def("set_nsimp_plasma",  &EpotSolver::set_nsimp_plasma, py::arg("rhop"), py::arg("Ep"), py::arg("rhoi"), py::arg("Ei"))
        .def("set_shield_plasma", &EpotSolver::set_shield_plasma, py::arg("Tm"), py::arg("Um"))
        .def("set_initial_plasma",&EpotSolver::set_initial_plasma, py::keep_alive<1, 3>())
        .def("set_forced_potential_volume", (void (EpotSolver::*)(double, CallbackFunctorB_V*)) &EpotSolver::set_forced_potential_volume, py::keep_alive<1, 3>())
        .def("set_forced_potential_volume_func", (void (EpotSolver::*)(CallbackFunctorD_V*)) &EpotSolver::set_forced_potential_volume, py::keep_alive<1, 2>())
        .def("set_plasma_calc_region", &EpotSolver::set_plasma_calc_region)
        .def("linear",            &EpotSolver::linear)
        .def("geometry", &EpotSolver::geometry, py::return_value_policy::reference);

    // -----------------------------------------------------------------------
    // EpotGSSolver
    // -----------------------------------------------------------------------
    py::class_<EpotGSSolver, EpotSolver>(m, "EpotGSSolver")
        .def(py::init<Geometry &>())
        .def("set_eps",   &EpotGSSolver::set_eps, py::arg("eps"))
        .def("set_imax",  &EpotGSSolver::set_imax, py::arg("imax"))
        .def("set_w",     &EpotGSSolver::set_w, py::arg("w"))
        .def("set_plasma_solver_parameters", &EpotGSSolver::set_plasma_solver_parameters, py::arg("Ulim_fac"), py::arg("imax"), py::arg("eps"))
        .def("get_potential_change_norm", &EpotGSSolver::get_potential_change_norm)
        .def("get_error_estimate",        &EpotGSSolver::get_error_estimate)
        .def("get_iter",                  &EpotGSSolver::get_iter);

    // -----------------------------------------------------------------------
    // EpotMGSolver
    // -----------------------------------------------------------------------
    py::class_<EpotMGSolver, EpotSolver>(m, "EpotMGSolver")
        .def(py::init<Geometry &>())
        .def("set_eps",    &EpotMGSolver::set_eps, py::arg("eps"))
        .def("set_imax",   &EpotMGSolver::set_imax, py::arg("imax"))
        .def("set_levels", &EpotMGSolver::set_levels, py::arg("levels"))
        .def("set_npre",   &EpotMGSolver::set_npre, py::arg("npre"))
        .def("set_npost",  &EpotMGSolver::set_npost, py::arg("npost"))
        .def("set_gamma",  &EpotMGSolver::set_gamma, py::arg("gamma"));

    // EpotMatrixSolver intermediate base (needed for BiCGSTAB / UMFPACK)
    py::class_<EpotMatrixSolver, EpotSolver>(m, "EpotMatrixSolver");

    // -----------------------------------------------------------------------
    // EpotBiCGSTABSolver
    // -----------------------------------------------------------------------
    py::class_<EpotBiCGSTABSolver, EpotMatrixSolver>(m, "EpotBiCGSTABSolver")
        .def(py::init<Geometry &>())
        .def("set_eps",  &EpotBiCGSTABSolver::set_eps, py::arg("eps"))
        .def("set_imax", &EpotBiCGSTABSolver::set_imax, py::arg("imax"));

#ifdef HAVE_LIBUMFPACK
    // EpotUMFPACKSolver (only available when UMFPACK is installed)
    py::class_<EpotUMFPACKSolver, EpotMatrixSolver>(m, "EpotUMFPACKSolver")
        .def(py::init<Geometry &>());
#endif

    // -----------------------------------------------------------------------
    // ParticleStatistics
    // -----------------------------------------------------------------------
    py::class_<ParticleStatistics>(m, "ParticleStatistics")
        .def(py::init<>())
        .def(py::init<uint32_t>(), py::arg("nboundaries"))
        .def("clear",               &ParticleStatistics::clear)
        .def("reset",               &ParticleStatistics::reset)
        .def("end_time",            &ParticleStatistics::end_time)
        .def("end_step",            &ParticleStatistics::end_step)
        .def("end_baddef",          &ParticleStatistics::end_baddef)
        .def("sum_steps",           &ParticleStatistics::sum_steps)
        .def("number_of_boundaries",&ParticleStatistics::number_of_boundaries)
        .def("bound_collisions",    [](const ParticleStatistics &ps, uint32_t b){ return ps.bound_collisions(b); })
        .def("total_collisions",    [](const ParticleStatistics &ps){ return ps.bound_collisions(); })
        .def("bound_current",       [](const ParticleStatistics &ps, uint32_t b){ return ps.bound_current(b); })
        .def("total_current",       [](const ParticleStatistics &ps){ return ps.bound_current(); });

    // -----------------------------------------------------------------------
    // ParticlePBase (abstract base for particle coordinates)
    // -----------------------------------------------------------------------
    py::class_<ParticlePBase>(m, "ParticlePBase");

    // -----------------------------------------------------------------------
    // ParticleP2D
    // -----------------------------------------------------------------------
    py::class_<ParticleP2D, ParticlePBase>(m, "ParticleP2D")
        .def(py::init<>())
        .def(py::init<double,double,double,double,double>())
        .def("__call__", [](const ParticleP2D &p, int i){ return p(i); })
        .def("size",     &ParticleP2D::size)
        .def("location", &ParticleP2D::location)
        .def("velocity", &ParticleP2D::velocity)
        .def("t",  [](const ParticleP2D &p){ return p(PARTICLE_T); })
        .def("x",  [](const ParticleP2D &p){ return p(PARTICLE_X); })
        .def("vx", [](const ParticleP2D &p){ return p(PARTICLE_VX); })
        .def("y",  [](const ParticleP2D &p){ return p(PARTICLE_Y); })
        .def("vy", [](const ParticleP2D &p){ return p(PARTICLE_VY); });

    // -----------------------------------------------------------------------
    // ParticlePCyl
    // -----------------------------------------------------------------------
    py::class_<ParticlePCyl, ParticlePBase>(m, "ParticlePCyl")
        .def(py::init<>())
        .def(py::init<double,double,double,double,double,double>())
        .def("__call__", [](const ParticlePCyl &p, int i){ return p(i); })
        .def("size",     &ParticlePCyl::size)
        .def("location", &ParticlePCyl::location)
        .def("velocity", &ParticlePCyl::velocity)
        .def("t",  [](const ParticlePCyl &p){ return p(PARTICLE_T); })
        .def("x",  [](const ParticlePCyl &p){ return p(PARTICLE_X); })
        .def("vx", [](const ParticlePCyl &p){ return p(PARTICLE_VX); })
        .def("r",  [](const ParticlePCyl &p){ return p(PARTICLE_R); })
        .def("vr", [](const ParticlePCyl &p){ return p(PARTICLE_VR); })
        .def("w",  [](const ParticlePCyl &p){ return p(PARTICLE_W); });

    // -----------------------------------------------------------------------
    // ParticleP3D
    // -----------------------------------------------------------------------
    py::class_<ParticleP3D, ParticlePBase>(m, "ParticleP3D")
        .def(py::init<>())
        .def(py::init<double,double,double,double,double,double,double>())
        .def("__call__", [](const ParticleP3D &p, int i){ return p(i); })
        .def("size",     &ParticleP3D::size)
        .def("location", &ParticleP3D::location)
        .def("velocity", &ParticleP3D::velocity)
        .def("t",  [](const ParticleP3D &p){ return p(PARTICLE_T); })
        .def("x",  [](const ParticleP3D &p){ return p(PARTICLE_X); })
        .def("vx", [](const ParticleP3D &p){ return p(PARTICLE_VX); })
        .def("y",  [](const ParticleP3D &p){ return p(PARTICLE_Y); })
        .def("vy", [](const ParticleP3D &p){ return p(PARTICLE_VY); })
        .def("z",  [](const ParticleP3D &p){ return p(PARTICLE_Z); })
        .def("vz", [](const ParticleP3D &p){ return p(PARTICLE_VZ); });

    // -----------------------------------------------------------------------
    // ParticleBase
    // -----------------------------------------------------------------------
    py::class_<ParticleBase, std::unique_ptr<ParticleBase, py::nodelete>>(m, "ParticleBase")
        .def("get_status", &ParticleBase::get_status)
        .def("set_status", &ParticleBase::set_status)
        .def("IQ",  &ParticleBase::IQ)
        .def("q",   &ParticleBase::q)
        .def("m",   &ParticleBase::m)
        .def("qm",  &ParticleBase::qm);

    // -----------------------------------------------------------------------
    // Particle2D
    // -----------------------------------------------------------------------
    py::class_<Particle2D, ParticleBase>(m, "Particle2D")
        .def(py::init<double, double, double, const ParticleP2D &>())
        .def("location",  &Particle2D::location)
        .def("velocity",  &Particle2D::velocity)
        .def("x",         [](const Particle2D &p) -> const ParticleP2D& { return p.x(); },
             py::return_value_policy::reference_internal)
        .def("traj_size", &Particle2D::traj_size)
        .def("traj",      [](const Particle2D &p, int i) -> const ParticleP2D& { return p.traj(i); },
             py::return_value_policy::reference_internal)
        .def("trajectory", &trajectory_array<Particle2D, ParticleP2D>,
             "Saved trajectory as an (npoints, ncoord) array; columns as in ParticleP2D")
        .def("clear_trajectory", &Particle2D::clear_trajectory)
        .def("reset_trajectory", &Particle2D::reset_trajectory);

    // -----------------------------------------------------------------------
    // ParticleCyl
    // -----------------------------------------------------------------------
    py::class_<ParticleCyl, ParticleBase>(m, "ParticleCyl")
        .def(py::init<double, double, double, const ParticlePCyl &>())
        .def("location",  &ParticleCyl::location)
        .def("velocity",  &ParticleCyl::velocity)
        .def("x",         [](const ParticleCyl &p) -> const ParticlePCyl& { return p.x(); },
             py::return_value_policy::reference_internal)
        .def("traj_size", &ParticleCyl::traj_size)
        .def("traj",      [](const ParticleCyl &p, int i) -> const ParticlePCyl& { return p.traj(i); },
             py::return_value_policy::reference_internal)
        .def("trajectory", &trajectory_array<ParticleCyl, ParticlePCyl>,
             "Saved trajectory as an (npoints, ncoord) array; columns as in ParticlePCyl")
        .def("clear_trajectory", &ParticleCyl::clear_trajectory)
        .def("reset_trajectory", &ParticleCyl::reset_trajectory);

    // -----------------------------------------------------------------------
    // Particle3D
    // -----------------------------------------------------------------------
    py::class_<Particle3D, ParticleBase>(m, "Particle3D")
        .def(py::init<double, double, double, const ParticleP3D &>())
        .def("location",  &Particle3D::location)
        .def("velocity",  &Particle3D::velocity)
        .def("x",         [](const Particle3D &p) -> const ParticleP3D& { return p.x(); },
             py::return_value_policy::reference_internal)
        .def("traj_size", &Particle3D::traj_size)
        .def("traj",      [](const Particle3D &p, int i) -> const ParticleP3D& { return p.traj(i); },
             py::return_value_policy::reference_internal)
        .def("trajectory", &trajectory_array<Particle3D, ParticleP3D>,
             "Saved trajectory as an (npoints, ncoord) array; columns as in ParticleP3D")
        .def("clear_trajectory", &Particle3D::clear_trajectory)
        .def("reset_trajectory", &Particle3D::reset_trajectory);

    // -----------------------------------------------------------------------
    // Trajectory callbacks
    // -----------------------------------------------------------------------
    py::class_<TrajectoryHandlerCallback, PyTrajectoryHandlerCallback>(m, "TrajectoryHandlerCallback")
        .def(py::init<>());

    py::class_<TrajectoryEndCallback, PyTrajectoryEndCallback>(m, "TrajectoryEndCallback")
        .def(py::init<>());

    py::class_<TrajectorySurfaceCollisionCallback, PyTrajectorySurfaceCollisionCallback>(
        m, "TrajectorySurfaceCollisionCallback")
        .def(py::init<>());

    // -----------------------------------------------------------------------
    // TrajectoryDiagnosticColumn
    // -----------------------------------------------------------------------
    py::class_<TrajectoryDiagnosticColumn>(m, "TrajectoryDiagnosticColumn")
        .def(py::init<trajectory_diagnostic_e>())
        .def("data",       [](const TrajectoryDiagnosticColumn &c){ return vector_to_numpy(c.data()); },
             "Column values as a numpy array")
        .def("size",       &TrajectoryDiagnosticColumn::size)
        .def("diagnostic", &TrajectoryDiagnosticColumn::diagnostic)
        .def("__call__",   [](const TrajectoryDiagnosticColumn &c, size_t i){ return c(i); })
        .def("__getitem__",[](const TrajectoryDiagnosticColumn &c, size_t i){ return c[i]; });

    // -----------------------------------------------------------------------
    // TrajectoryDiagnosticData
    // -----------------------------------------------------------------------
    py::class_<TrajectoryDiagnosticData>(m, "TrajectoryDiagnosticData")
        .def(py::init<>())
        .def(py::init<std::vector<trajectory_diagnostic_e>>())
        .def("clear",           &TrajectoryDiagnosticData::clear)
        .def("add_data_column", &TrajectoryDiagnosticData::add_data_column)
        .def("diag_size",       &TrajectoryDiagnosticData::diag_size)
        .def("traj_size",       &TrajectoryDiagnosticData::traj_size)
        .def("diagnostic",      &TrajectoryDiagnosticData::diagnostic)
        .def("column",          [](const TrajectoryDiagnosticData &d, size_t i) -> const TrajectoryDiagnosticColumn& {
            return d(i);
        }, py::return_value_policy::reference_internal)
        .def("get",             [](const TrajectoryDiagnosticData &d, size_t j, size_t i){ return d(j,i); })
        .def("add_data",        &TrajectoryDiagnosticData::add_data)
        .def("export_data",     &TrajectoryDiagnosticData::export_data);

    // -----------------------------------------------------------------------
    // Emittance
    // -----------------------------------------------------------------------
    py::class_<Emittance>(m, "Emittance")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>())
        .def(py::init<const std::vector<double>&, const std::vector<double>&>())
        .def("current", &Emittance::current)
        .def("xave",    &Emittance::xave)
        .def("xpave",   &Emittance::xpave)
        .def("alpha",   &Emittance::alpha)
        .def("beta",    &Emittance::beta)
        .def("gamma",   &Emittance::gamma)
        .def("epsilon", &Emittance::epsilon)
        .def("angle",   &Emittance::angle)
        .def("rmajor",  &Emittance::rmajor)
        .def("rminor",  &Emittance::rminor);

    // -----------------------------------------------------------------------
    // EmittanceConv
    // -----------------------------------------------------------------------
    py::class_<EmittanceConv, Emittance>(m, "EmittanceConv")
        .def(py::init([](uint32_t n, uint32_t mm,
                         const std::vector<double> &r,
                         const std::vector<double> &rp,
                         const std::vector<double> &ap,
                         const std::vector<double> &I,
                         uint32_t rotn,
                         double xmin, double xpmin, double xmax, double xpmax){
            return new EmittanceConv(n, mm, r, rp, ap, I, rotn, xmin, xpmin, xmax, xpmax);
        }), py::arg("n"), py::arg("m"),
            py::arg("r"), py::arg("rp"), py::arg("ap"), py::arg("I"),
            py::arg("rotn")=100,
            py::arg("xmin")=std::numeric_limits<double>::quiet_NaN(),
            py::arg("xpmin")=std::numeric_limits<double>::quiet_NaN(),
            py::arg("xmax")=std::numeric_limits<double>::quiet_NaN(),
            py::arg("xpmax")=std::numeric_limits<double>::quiet_NaN())
        .def("histogram",      &EmittanceConv::histogram, py::return_value_policy::reference_internal)
        .def("free_histogram", &EmittanceConv::free_histogram);

    // -----------------------------------------------------------------------
    // Convergence
    // -----------------------------------------------------------------------
    py::class_<Convergence>(m, "Convergence")
        .def(py::init<>())
        .def("add_epot",          &Convergence::add_epot)
        .def("add_scharge",       &Convergence::add_scharge)
        .def("add_emittance",     &Convergence::add_emittance)
        .def("evaluate_iteration",&Convergence::evaluate_iteration)
        .def("clear",             &Convergence::clear)
        .def("print_history",     [](const Convergence &c, const std::string &filename){
            std::ofstream of(filename); c.print_history(of);
        });

    // -----------------------------------------------------------------------
    // Histogram1D
    // -----------------------------------------------------------------------
    py::class_<Histogram>(m, "Histogram");

    py::class_<Histogram1D, Histogram>(m, "Histogram1D")
        .def(py::init([](uint32_t n, py::list range){
            double r[2]; r[0]=range[0].cast<double>(); r[1]=range[1].cast<double>();
            return new Histogram1D(n, r);
        }))
        .def(py::init([](uint32_t n, const std::vector<double> &xdata,
                         histogram_accumulation_e type){
            return new Histogram1D(n, xdata, type);
        }), py::arg("n"), py::arg("xdata"),
            py::arg("type")=HISTOGRAM_ACCUMULATION_CLOSEST)
        .def(py::init([](uint32_t n, const std::vector<double> &xdata,
                         const std::vector<double> &wdata,
                         histogram_accumulation_e type){
            return new Histogram1D(n, xdata, wdata, type);
        }), py::arg("n"), py::arg("xdata"), py::arg("wdata"),
            py::arg("type")=HISTOGRAM_ACCUMULATION_CLOSEST)
        .def("n",    &Histogram1D::n)
        .def("step", &Histogram1D::step)
        .def("coord",&Histogram1D::coord)
        .def("accumulate_closest", [](Histogram1D &h, double x, double w){ h.accumulate_closest(x,w); },
             py::arg("x"), py::arg("weight")=1.0)
        .def("accumulate_linear",  [](Histogram1D &h, double x, double w){ h.accumulate_linear(x,w); },
             py::arg("x"), py::arg("weight")=1.0)
        .def("convert_to_density", &Histogram1D::convert_to_density)
        .def("get_range", [](const Histogram1D &h){
            double r[2]; h.get_range(r); return py::make_tuple(r[0],r[1]);
        })
        .def("get_bin_range", [](const Histogram1D &h){
            double mn, mx; h.get_bin_range(mn,mx); return py::make_tuple(mn,mx);
        })
        .def("get_data", [](const Histogram1D &h){ return vector_to_numpy(h.get_data()); })
        .def("__call__", [](const Histogram1D &h, uint32_t i){ return h(i); });

    // -----------------------------------------------------------------------
    // Histogram2D
    // -----------------------------------------------------------------------
    py::class_<Histogram2D, Histogram>(m, "Histogram2D")
        .def(py::init([](uint32_t n, uint32_t mm, py::list range){
            double r[4]; for(int i=0;i<4;i++) r[i]=range[i].cast<double>();
            return new Histogram2D(n, mm, r);
        }), py::arg("n"), py::arg("m"), py::arg("range"),
            "n x m histogram (n, m >= 4) with range [xmin, ymin, xmax, ymax]")
        .def(py::init([](uint32_t n, uint32_t mm,
                         const std::vector<double> &xd, const std::vector<double> &yd,
                         histogram_accumulation_e type){
            return new Histogram2D(n, mm, xd, yd, type);
        }), py::arg("n"), py::arg("m"), py::arg("xdata"), py::arg("ydata"),
            py::arg("type")=HISTOGRAM_ACCUMULATION_CLOSEST)
        .def(py::init([](uint32_t n, uint32_t mm,
                         const std::vector<double> &xd, const std::vector<double> &yd,
                         const std::vector<double> &wd, histogram_accumulation_e type){
            return new Histogram2D(n, mm, xd, yd, wd, type);
        }), py::arg("n"), py::arg("m"), py::arg("xdata"), py::arg("ydata"), py::arg("wdata"),
            py::arg("type")=HISTOGRAM_ACCUMULATION_CLOSEST)
        .def("n",      &Histogram2D::n)
        .def("m",      &Histogram2D::m)
        .def("nstep",  &Histogram2D::nstep)
        .def("mstep",  &Histogram2D::mstep)
        .def("icoord", &Histogram2D::icoord)
        .def("jcoord", &Histogram2D::jcoord)
        .def("accumulate_closest", [](Histogram2D &h, double x, double y, double w){
            h.accumulate_closest(x,y,w);
        })
        .def("accumulate_linear",  [](Histogram2D &h, double x, double y, double w){
            h.accumulate_linear(x,y,w);
        })
        .def("convert_to_density", &Histogram2D::convert_to_density)
        .def("get_range", [](const Histogram2D &h){
            double r[4]; h.get_range(r); return py::make_tuple(r[0],r[1],r[2],r[3]);
        })
        .def("get_bin_range", [](const Histogram2D &h){
            double mn, mx; h.get_bin_range(mn,mx); return py::make_tuple(mn,mx);
        })
        .def("get_data", [](const Histogram2D &h){
            // Stored as data[i + j*n]; expose as an (n, m) array with h[i, j].
            return py::array_t<double>({ (py::ssize_t)h.n(), (py::ssize_t)h.m() },
                                       { (py::ssize_t)sizeof(double), (py::ssize_t)(sizeof(double)*h.n()) },
                                       h.get_data().data());
        }, "Bin values as an (n, m) array")
        .def("__call__", [](const Histogram2D &h, uint32_t i, uint32_t j){ return h(i,j); });

    // -----------------------------------------------------------------------
    // Interpolation2D
    // -----------------------------------------------------------------------
    py::class_<Interpolation2D>(m, "Interpolation2D");

    py::class_<ClosestInterpolation2D, Interpolation2D>(m, "ClosestInterpolation2D")
        .def(py::init<size_t, size_t, const std::vector<double>&>())
        .def("__call__", [](const ClosestInterpolation2D &f, double x, double y){ return f(x,y); });

    py::class_<BiLinearInterpolation2D, Interpolation2D>(m, "BiLinearInterpolation2D")
        .def(py::init<size_t, size_t, const std::vector<double>&>())
        .def("__call__", [](const BiLinearInterpolation2D &f, double x, double y){ return f(x,y); });

    py::class_<BiCubicInterpolation2D, Interpolation2D>(m, "BiCubicInterpolation2D")
        .def(py::init<size_t, size_t, const std::vector<double>&>())
        .def("__call__", [](const BiCubicInterpolation2D &f, double x, double y){ return f(x,y); });

    // -----------------------------------------------------------------------
    // Random_Variate_Transformation
    // -----------------------------------------------------------------------
    py::class_<Random_Variate_Transformation, PyRandom_Variate_Transformation>(
        m, "Random_Variate_Transformation")
        .def(py::init<>())
        .def("transform", &Random_Variate_Transformation::transform);

    py::class_<Uniform_Transformation, Random_Variate_Transformation>(m, "Uniform_Transformation")
        .def(py::init<>())
        .def("transform", &Uniform_Transformation::transform);

    py::class_<Gaussian_Transformation, Random_Variate_Transformation>(m, "Gaussian_Transformation")
        .def(py::init<>())
        .def("transform", &Gaussian_Transformation::transform);

    py::class_<Cosine_Transformation, Random_Variate_Transformation>(m, "Cosine_Transformation")
        .def(py::init<>())
        .def("transform", &Cosine_Transformation::transform);

    py::class_<Gamma_Transformation, Random_Variate_Transformation>(m, "Gamma_Transformation")
        .def(py::init<double, double>())
        .def("transform", &Gamma_Transformation::transform);

    // -----------------------------------------------------------------------
    // Random (dimension N set in constructor of QRandom/MTRandom)
    // -----------------------------------------------------------------------
    py::class_<Random>(m, "Random")
        .def("set_transformation", &Random::set_transformation)
        .def("get", [](const Random &r, size_t N) -> std::vector<double> {
            std::vector<double> v(N);
            r.get(v.data());
            return v;
        }, "Get N random values (N must match dimension set at construction)");

    py::class_<QRandom, Random>(m, "QRandom")
        .def(py::init<size_t>(), py::arg("N"), "Quasi-random (Sobol) generator with N dimensions");

    py::class_<MTRandom, Random>(m, "MTRandom")
        .def(py::init<size_t>(), py::arg("N"), "Mersenne Twister random generator with N dimensions");

    // -----------------------------------------------------------------------
    // Timer
    // -----------------------------------------------------------------------
    py::class_<Timer>(m, "Timer")
        .def(py::init<>())
        .def("start",         &Timer::start)
        .def("stop",          &Timer::stop)
        .def("get_real_time", &Timer::get_real_time)
        .def("get_cpu_time",  &Timer::get_cpu_time);

    // -----------------------------------------------------------------------
    // StatusPrint
    // -----------------------------------------------------------------------
    py::class_<StatusPrint>(m, "StatusPrint")
        .def(py::init<>())
        .def("print", &StatusPrint::print, py::arg("str"), py::arg("force")=false);

    // -----------------------------------------------------------------------
    // IBSimu global singleton
    // -----------------------------------------------------------------------
    py::class_<IBSimu>(m, "IBSimu")
        .def("set_message_output", [](IBSimu &ib, const std::string &fn){ ib.set_message_output(fn); })
        .def("set_message_threshold", &IBSimu::set_message_threshold, py::arg("type"), py::arg("level"))
        .def("get_message_threshold", &IBSimu::get_message_threshold)
        .def("set_thread_count",      &IBSimu::set_thread_count, py::arg("n"))
        .def("set_rng_type",          &IBSimu::set_rng_type, py::arg("type"))
        .def("inc_indent",            &IBSimu::inc_indent)
        .def("dec_indent",            &IBSimu::dec_indent)
        .def("output_is_cout",        &IBSimu::output_is_cout);

    m.attr("ibsimu") = py::cast(&ibsimu, py::return_value_policy::reference);

    // -----------------------------------------------------------------------
    // ParticleDataBase (abstract base with iteration settings)
    // -----------------------------------------------------------------------
    py::class_<ParticleDataBase>(m, "ParticleDataBase")
        .def("set_accuracy",         &ParticleDataBase::set_accuracy, py::arg("epsabs"), py::arg("epsrel"))
        .def("set_bfield_suppression",&ParticleDataBase::set_bfield_suppression, py::keep_alive<1, 2>())
        .def("set_trajectory_handler_callback", [](ParticleDataBase &pdb, TrajectoryHandlerCallback *cb){
            register_callback<TrajectoryHandlerCallback, PyTrajectoryHandlerCallback>(
                pdb, cb, &ParticleDataBase::set_trajectory_handler_callback);
        }, py::keep_alive<1, 2>())
        .def("set_trajectory_end_callback", [](ParticleDataBase &pdb, TrajectoryEndCallback *cb){
            register_callback<TrajectoryEndCallback, PyTrajectoryEndCallback>(
                pdb, cb, &ParticleDataBase::set_trajectory_end_callback);
        }, py::keep_alive<1, 2>())
        .def("set_trajectory_surface_collision_callback", [](ParticleDataBase &pdb, TrajectorySurfaceCollisionCallback *cb){
            register_callback<TrajectorySurfaceCollisionCallback, PyTrajectorySurfaceCollisionCallback>(
                pdb, cb, &ParticleDataBase::set_trajectory_surface_collision_callback);
        }, py::keep_alive<1, 2>())
        .def("set_relativistic",       &ParticleDataBase::set_relativistic, py::arg("enable"))
        .def("set_surface_collision",  &ParticleDataBase::set_surface_collision, py::arg("enable"))
        .def("set_polyint",            &ParticleDataBase::set_polyint, py::arg("enable"))
        .def("get_polyint",            &ParticleDataBase::get_polyint)
        .def("set_trajectory_interpolation", &ParticleDataBase::set_trajectory_interpolation, py::arg("intrp"))
        .def("get_trajectory_interpolation", &ParticleDataBase::get_trajectory_interpolation)
        .def("set_scharge_deposition", &ParticleDataBase::set_scharge_deposition, py::arg("type"))
        .def("get_scharge_deposition", &ParticleDataBase::get_scharge_deposition)
        .def("set_max_steps",          &ParticleDataBase::set_max_steps, py::arg("maxsteps"))
        .def("set_max_time",           &ParticleDataBase::set_max_time, py::arg("maxt"))
        .def("set_save_all_points",    &ParticleDataBase::set_save_all_points, py::arg("enable"))
        .def("set_save_trajectories",  [](ParticleDataBase &pdb, py::object div){
            // Accept True/False as well as an integer divisor.
            pdb.set_save_trajectories(py::isinstance<py::bool_>(div) ? (div.cast<bool>() ? 1 : 0)
                                                                     : div.cast<uint32_t>());
        }, py::arg("div"), "Save every div-th trajectory (0 or False: none, 1 or True: all)")
        .def("get_save_trajectories",  &ParticleDataBase::get_save_trajectories)
        .def("set_mirror", [](ParticleDataBase &pdb, py::list mirror){
            bool m[6]; list_to_mirror(mirror, m); pdb.set_mirror(m);
        })
        .def("get_mirror", [](const ParticleDataBase &pdb){
            bool m[6]; pdb.get_mirror(m);
            return py::make_tuple(m[0],m[1],m[2],m[3],m[4],m[5]);
        })
        .def("get_iteration_number",  &ParticleDataBase::get_iteration_number)
        .def("get_rhosum",            &ParticleDataBase::get_rhosum)
        .def("set_rhosum",            &ParticleDataBase::set_rhosum, py::arg("rhosum"))
        .def("get_statistics",        &ParticleDataBase::get_statistics,
             py::return_value_policy::reference_internal)
        .def("geom_mode",             &ParticleDataBase::geom_mode)
        .def("size",                  &ParticleDataBase::size)
        .def("particle_base",         [](ParticleDataBase &pdb, uint32_t i) -> ParticleBase& {
            return pdb.particle(i);
        }, py::return_value_policy::reference_internal)
        .def("traj_length",           &ParticleDataBase::traj_length)
        .def("traj_size",             &ParticleDataBase::traj_size)
        .def("trajectory_point_tlv",  [](const ParticleDataBase &pdb, uint32_t i, uint32_t j){
            double t; Vec3D loc, vel;
            pdb.trajectory_point(t, loc, vel, i, j);
            return py::make_tuple(t, loc, vel);
        })
        .def("trajectories_at_plane", [](const ParticleDataBase &pdb,
                                         TrajectoryDiagnosticData &tdata,
                                         coordinate_axis_e axis, double val,
                                         const std::vector<trajectory_diagnostic_e> &diag){
            pdb.trajectories_at_plane(tdata, axis, val, diag);
        }, py::arg("tdata"), py::arg("axis"), py::arg("val"), py::arg("diagnostics"),
           "Collect diagnostics where trajectories cross the plane axis = val")
        .def("build_trajectory_density_field", &ParticleDataBase::build_trajectory_density_field, py::arg("tdens"))
        .def("clear",                 &ParticleDataBase::clear)
        .def("clear_trajectories",    &ParticleDataBase::clear_trajectories)
        .def("clear_trajectory",      &ParticleDataBase::clear_trajectory)
        .def("reset_trajectories",    &ParticleDataBase::reset_trajectories)
        .def("reset_trajectory",      &ParticleDataBase::reset_trajectory)
        .def("reserve",               &ParticleDataBase::reserve, py::arg("size"))
        .def("iterate_trajectories",  [](ParticleDataBase &pdb, MeshScalarField &sc,
                                         const VectorField &ef, const VectorField &bf){
            { py::gil_scoped_release release; pdb.iterate_trajectories(sc, ef, bf); }
            rethrow_python_error();
        }, py::arg("scharge"), py::arg("efield"), py::arg("bfield"))
        .def("step_particles",        [](ParticleDataBase &pdb, MeshScalarField &sc,
                                         const VectorField &ef, const VectorField &bf, double dt){
            { py::gil_scoped_release release; pdb.step_particles(sc, ef, bf, dt); }
            rethrow_python_error();
        }, py::arg("scharge"), py::arg("efield"), py::arg("bfield"), py::arg("dt"))
        .def("save", [](const ParticleDataBase &pdb, const std::string &fn){ pdb.save(fn); });

    // -----------------------------------------------------------------------
    // ParticleDataBase2D
    // -----------------------------------------------------------------------
    py::class_<ParticleDataBase2D, ParticleDataBase>(m, "ParticleDataBase2D")
        .def(py::init<const Geometry &>(), py::arg("geom"))
        .def(py::init<const ParticleDataBase2D &>())
        .def(py::init([](const std::string &filename, const Geometry &geom){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new ParticleDataBase2D(is, geom);
        }), py::arg("filename"), py::arg("geom"), "Load a particle database saved with save()")
        .def("particle", [](ParticleDataBase2D &pdb, uint32_t i) -> Particle2D& {
            return pdb.particle(i);
        }, py::return_value_policy::reference_internal)
        .def("trajectory_point", [](const ParticleDataBase2D &pdb, uint32_t i, uint32_t j)
            -> const ParticleP2D& { return pdb.trajectory_point(i,j); },
            py::return_value_policy::reference_internal)
        .def("add_particle", [](ParticleDataBase2D &pdb, double IQ, double q, double m,
                                 const ParticleP2D &x){ pdb.add_particle(IQ, q, m, x); },
             py::arg("IQ"), py::arg("q"), py::arg("m"), py::arg("x"))
        .def("add_particle", [](ParticleDataBase2D &pdb, const Particle2D &p){ pdb.add_particle(p); }, py::arg("p"))
        .def("add_2d_beam_with_energy",       &ParticleDataBase2D::add_2d_beam_with_energy,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("E"), py::arg("Tp"), py::arg("Tt"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             "N particles, current density J [A/m], charge q [e], mass m [u], energy E [eV], "
             "parallel/transverse temperatures Tp, Tt [eV], from (x1, y1) to (x2, y2)")
        .def("add_2d_beam_with_velocity",     &ParticleDataBase2D::add_2d_beam_with_velocity,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("v"), py::arg("dvp"), py::arg("dvt"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"))
        .def("add_2d_gaussian_beam_with_emittance",
             &ParticleDataBase2D::add_2d_gaussian_beam_with_emittance,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("a"), py::arg("b"), py::arg("e"), py::arg("Ex"), py::arg("x0"), py::arg("y0"))
        .def("add_2d_KV_beam_with_emittance", &ParticleDataBase2D::add_2d_KV_beam_with_emittance,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("a"), py::arg("b"), py::arg("e"), py::arg("Ex"), py::arg("x0"), py::arg("y0"));

    // -----------------------------------------------------------------------
    // ParticleDataBaseCyl
    // -----------------------------------------------------------------------
    py::class_<ParticleDataBaseCyl, ParticleDataBase>(m, "ParticleDataBaseCyl")
        .def(py::init<const Geometry &>(), py::arg("geom"))
        .def(py::init<const ParticleDataBaseCyl &>())
        .def(py::init([](const std::string &filename, const Geometry &geom){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new ParticleDataBaseCyl(is, geom);
        }), py::arg("filename"), py::arg("geom"), "Load a particle database saved with save()")
        .def("particle", [](ParticleDataBaseCyl &pdb, uint32_t i) -> ParticleCyl& {
            return pdb.particle(i);
        }, py::return_value_policy::reference_internal)
        .def("trajectory_point", [](const ParticleDataBaseCyl &pdb, uint32_t i, uint32_t j)
            -> const ParticlePCyl& { return pdb.trajectory_point(i,j); },
            py::return_value_policy::reference_internal)
        .def("add_particle", [](ParticleDataBaseCyl &pdb, double IQ, double q, double m,
                                 const ParticlePCyl &x){ pdb.add_particle(IQ, q, m, x); },
             py::arg("IQ"), py::arg("q"), py::arg("m"), py::arg("x"))
        .def("add_particle", [](ParticleDataBaseCyl &pdb, const ParticleCyl &p){ pdb.add_particle(p); }, py::arg("p"))
        .def("add_2d_beam_with_energy",       &ParticleDataBaseCyl::add_2d_beam_with_energy,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("E"), py::arg("Tp"), py::arg("Tt"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             "N particles, current density J [A/m^2], charge q [e], mass m [u], energy E [eV], "
             "parallel/transverse temperatures Tp, Tt [eV], from (x1, r1) to (x2, r2)")
        .def("add_2d_beam_with_total_energy", &ParticleDataBaseCyl::add_2d_beam_with_total_energy,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("Etot"), py::arg("epot"), py::arg("Tp"), py::arg("Tt"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"))
        .def("add_2d_beam_with_velocity",     &ParticleDataBaseCyl::add_2d_beam_with_velocity,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("v"), py::arg("dvp"), py::arg("dvt"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"))
        .def("add_2d_full_gaussian_beam",     &ParticleDataBaseCyl::add_2d_full_gaussian_beam,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("Ex"), py::arg("Tp"), py::arg("Tt"), py::arg("x0"), py::arg("dr"))
        .def("add_2d_gaussian_beam_with_emittance",
             &ParticleDataBaseCyl::add_2d_gaussian_beam_with_emittance,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("a"), py::arg("b"), py::arg("e"), py::arg("Ex"), py::arg("x0"))
        .def("export_path_manager_data",      &ParticleDataBaseCyl::export_path_manager_data);

    // -----------------------------------------------------------------------
    // ParticleDataBase3D
    // -----------------------------------------------------------------------
    py::class_<ParticleDataBase3D, ParticleDataBase>(m, "ParticleDataBase3D")
        .def(py::init<const Geometry &>(), py::arg("geom"))
        .def(py::init<const ParticleDataBase3D &>())
        .def(py::init([](const std::string &filename, const Geometry &geom){
            std::ifstream is(filename);
            if (!is) throw std::runtime_error("Cannot open file: " + filename);
            return new ParticleDataBase3D(is, geom);
        }), py::arg("filename"), py::arg("geom"), "Load a particle database saved with save()")
        .def("particle", [](ParticleDataBase3D &pdb, uint32_t i) -> Particle3D& {
            return pdb.particle(i);
        }, py::return_value_policy::reference_internal)
        .def("trajectory_point", [](const ParticleDataBase3D &pdb, uint32_t i, uint32_t j)
            -> const ParticleP3D& { return pdb.trajectory_point(i,j); },
            py::return_value_policy::reference_internal)
        .def("add_particle", [](ParticleDataBase3D &pdb, double IQ, double q, double m,
                                 const ParticleP3D &x){ pdb.add_particle(IQ, q, m, x); },
             py::arg("IQ"), py::arg("q"), py::arg("m"), py::arg("x"))
        .def("add_particle", [](ParticleDataBase3D &pdb, const Particle3D &p){ pdb.add_particle(p); }, py::arg("p"))
        .def("add_cylindrical_beam_with_total_energy",
             &ParticleDataBase3D::add_cylindrical_beam_with_total_energy,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("Etot"), py::arg("epot"), py::arg("Tp"), py::arg("Tt"), py::arg("c"), py::arg("dir1"), py::arg("dir2"), py::arg("r"))
        .def("add_cylindrical_beam_with_energy",
             &ParticleDataBase3D::add_cylindrical_beam_with_energy,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("E"), py::arg("Tp"), py::arg("Tt"), py::arg("c"), py::arg("dir1"), py::arg("dir2"), py::arg("r"),
             "N particles, current density J [A/m^2], charge q [e], mass m [u], energy E [eV], "
             "temperatures Tp, Tt [eV], disc of radius r centred at c spanned by dir1, dir2")
        .def("add_cylindrical_beam_with_velocity",
             &ParticleDataBase3D::add_cylindrical_beam_with_velocity,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("v"), py::arg("dvp"), py::arg("dvt"), py::arg("c"), py::arg("dir1"), py::arg("dir2"), py::arg("r"))
        .def("add_rectangular_beam_with_energy",
             &ParticleDataBase3D::add_rectangular_beam_with_energy,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("E"), py::arg("Tp"), py::arg("Tt"), py::arg("c"), py::arg("dir1"), py::arg("dir2"), py::arg("size1"), py::arg("size2"))
        .def("add_rectangular_beam_with_velocity",
             &ParticleDataBase3D::add_rectangular_beam_with_velocity,
             py::arg("N"), py::arg("J"), py::arg("q"), py::arg("m"), py::arg("v"), py::arg("dvp"), py::arg("dvt"), py::arg("c"), py::arg("dir1"), py::arg("dir2"), py::arg("size1"), py::arg("size2"))
        .def("add_3d_KV_beam_with_emittance",
             &ParticleDataBase3D::add_3d_KV_beam_with_emittance,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("E0"), py::arg("a1"), py::arg("b1"), py::arg("e1"), py::arg("a2"), py::arg("b2"), py::arg("e2"), py::arg("c"), py::arg("dir1"), py::arg("dir2"))
        .def("add_3d_waterbag_beam_with_emittance",
             &ParticleDataBase3D::add_3d_waterbag_beam_with_emittance,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("E0"), py::arg("a1"), py::arg("b1"), py::arg("e1"), py::arg("a2"), py::arg("b2"), py::arg("e2"), py::arg("c"), py::arg("dir1"), py::arg("dir2"))
        .def("add_3d_gaussian_beam_with_emittance",
             &ParticleDataBase3D::add_3d_gaussian_beam_with_emittance,
             py::arg("N"), py::arg("I"), py::arg("q"), py::arg("m"), py::arg("E0"), py::arg("a1"), py::arg("b1"), py::arg("e1"), py::arg("a2"), py::arg("b2"), py::arg("e2"), py::arg("c"), py::arg("dir1"), py::arg("dir2"))
        .def("trajectories_at_free_plane", [](const ParticleDataBase3D &pdb,
                                              TrajectoryDiagnosticData &tdata,
                                              const Vec3D &c, const Vec3D &o, const Vec3D &p,
                                              const std::vector<trajectory_diagnostic_e> &diag){
            pdb.trajectories_at_free_plane(tdata, c, o, p, diag);
        })
        .def("export_path_manager_data", [](const ParticleDataBase3D &pdb,
                                            const std::string &fn,
                                            double ref_E, double ref_q, double ref_m,
                                            const Vec3D &c, const Vec3D &o, const Vec3D &p){
            pdb.export_path_manager_data(fn, ref_E, ref_q, ref_m, c, o, p);
        });

    // -----------------------------------------------------------------------
    // Plotting / diagnostics
    // -----------------------------------------------------------------------
    py::class_<Plotter, std::unique_ptr<Plotter, py::nodelete>>(m, "Plotter")
        .def("set_size",      &Plotter::set_size, py::arg("width"), py::arg("height"))
        .def("set_font_size", &Plotter::set_font_size, py::arg("size"))
        .def("set_ranges",    &Plotter::set_ranges, py::arg("xmin"), py::arg("ymin"), py::arg("xmax"), py::arg("ymax"))
        .def("get_ranges",    [](Plotter &p){
            double xmin, ymin, xmax, ymax; p.get_ranges(xmin, ymin, xmax, ymax);
            return py::make_tuple(xmin, ymin, xmax, ymax);
        })
        .def("plot_png",      &Plotter::plot_png, py::arg("filename"))
#ifdef CAIRO_HAS_PS_SURFACE
        .def("plot_eps",      &Plotter::plot_eps, py::arg("filename"))
#endif
#ifdef CAIRO_HAS_PDF_SURFACE
        .def("plot_pdf",      &Plotter::plot_pdf, py::arg("filename"))
#endif
#ifdef CAIRO_HAS_SVG_SURFACE
        .def("plot_svg",      &Plotter::plot_svg, py::arg("filename"))
#endif
        ;

    py::class_<MeshColormap>(m, "MeshColormap")
        .def("set_zscale", &MeshColormap::set_zscale, py::arg("zscale"))
        .def("get_zscale", &MeshColormap::get_zscale)
        .def("set_interpolation", &MeshColormap::set_interpolation, py::arg("interpolation"))
        .def("get_interpolation", &MeshColormap::get_interpolation)
        .def("set_zrange", &MeshColormap::set_zrange, py::arg("min"), py::arg("max"))
        .def("get_zrange", [](const MeshColormap &c){
            double mn, mx; c.get_zrange(mn, mx); return py::make_tuple(mn, mx);
        });

    py::class_<FieldGraph, MeshColormap>(m, "FieldGraph")
        .def("set_zrange", &FieldGraph::set_zrange);

    py::class_<GeomPlotter, Plotter>(m, "GeomPlotter")
        .def(py::init<const Geometry &>())
        .def("set_epot",              &GeomPlotter::set_epot, py::keep_alive<1, 2>())
        .def("set_bfield",            &GeomPlotter::set_bfield, py::keep_alive<1, 2>())
        .def("set_efield",            &GeomPlotter::set_efield, py::keep_alive<1, 2>())
        .def("set_scharge",           &GeomPlotter::set_scharge, py::keep_alive<1, 2>())
        .def("set_trajdens",          &GeomPlotter::set_trajdens, py::keep_alive<1, 2>())
        .def("set_particle_database", &GeomPlotter::set_particle_database, py::keep_alive<1, 2>())
        .def("set_particledatabase",  &GeomPlotter::set_particledatabase, py::keep_alive<1, 2>())
        .def("set_eqlines_manual",    &GeomPlotter::set_eqlines_manual, py::arg("pot"))
        .def("set_eqlines_auto",      &GeomPlotter::set_eqlines_auto, py::arg("N"))
        .def("enable_colormap_legend", &GeomPlotter::enable_colormap_legend, py::arg("enable"))
        .def("set_particle_div",      [](GeomPlotter &g, uint32_t div, uint32_t offset){
            g.set_particle_div(div, offset);
        }, py::arg("div"), py::arg("offset") = 0)
        .def("set_qm_discretation",   &GeomPlotter::set_qm_discretation, py::arg("enable"))
        .def("set_mesh",              &GeomPlotter::set_mesh, py::arg("enable"))
        .def("set_view",              &GeomPlotter::set_view, py::arg("view"), py::arg("level") = -1)
        .def("set_view_si",           &GeomPlotter::set_view_si, py::arg("view"), py::arg("level"))
        .def("set_fieldgraph_plot",   &GeomPlotter::set_fieldgraph_plot, py::arg("fieldplot"))
        .def("fieldgraph", (FieldGraph* (GeomPlotter::*)()) &GeomPlotter::fieldgraph,
             py::return_value_policy::reference_internal);

    py::class_<ParticleDiagPlotter, Plotter>(m, "ParticleDiagPlotter")
        .def(py::init<const Geometry &, const ParticleDataBase &, coordinate_axis_e, double,
                      particle_diag_plot_type_e, trajectory_diagnostic_e, trajectory_diagnostic_e>(),
             py::arg("geom"), py::arg("pdb"), py::arg("axis"), py::arg("level"),
             py::arg("type"), py::arg("diagx"), py::arg("diagy")=DIAG_NONE)
        .def(py::init<const Geometry &, const ParticleDataBase &, const Vec3D &, const Vec3D &, const Vec3D &,
                      particle_diag_plot_type_e, trajectory_diagnostic_e, trajectory_diagnostic_e>(),
             py::arg("geom"), py::arg("pdb"), py::arg("c"), py::arg("o"), py::arg("p"),
             py::arg("type"), py::arg("diagx"), py::arg("diagy")=DIAG_NONE)
        .def("set_emittance_ellipse", &ParticleDiagPlotter::set_emittance_ellipse, py::arg("enable"))
        .def("set_view", &ParticleDiagPlotter::set_view, py::arg("axis"), py::arg("level"))
        .def("set_plot", &ParticleDiagPlotter::set_plot, py::arg("type"), py::arg("diagx"), py::arg("diagy"))
        .def("set_histogram_n", &ParticleDiagPlotter::set_histogram_n, py::arg("n"))
        .def("set_histogram_m", &ParticleDiagPlotter::set_histogram_m, py::arg("m"))
        .def("set_histogram_accumulation", &ParticleDiagPlotter::set_histogram_accumulation, py::arg("accumulation"))
        .def("set_histogram_style", &ParticleDiagPlotter::set_histogram_style, py::arg("style"))
        .def("set_colormap_interpolation", &ParticleDiagPlotter::set_colormap_interpolation, py::arg("interpolation"))
        .def("set_dot_size", &ParticleDiagPlotter::set_dot_size, py::arg("size"))
        .def("get_isum", &ParticleDiagPlotter::get_isum, "Total current of the plotted particles")
        .def("export_data", &ParticleDiagPlotter::export_data, py::arg("filename"))
        .def("calculate_emittance", &ParticleDiagPlotter::calculate_emittance,
             py::return_value_policy::reference_internal);

    py::class_<FieldDiagPlotter, Plotter>(m, "FieldDiagPlotter")
        .def(py::init<const Geometry &>())
        .def("set_epot",     &FieldDiagPlotter::set_epot)
        .def("set_efield",   &FieldDiagPlotter::set_efield)
        .def("set_scharge",  &FieldDiagPlotter::set_scharge)
        .def("set_trajdens", &FieldDiagPlotter::set_trajdens)
        .def("set_bfield",   &FieldDiagPlotter::set_bfield)
        .def("set_coordinates", &FieldDiagPlotter::set_coordinates, py::arg("N"), py::arg("x1"), py::arg("x2"))
        .def("set_diagnostic", [](FieldDiagPlotter &p, py::list diag, py::list loc){
             field_diag_type_e d[2] = { diag[0].cast<field_diag_type_e>(), diag[1].cast<field_diag_type_e>() };
             field_loc_type_e l[2] = { loc[0].cast<field_loc_type_e>(), loc[1].cast<field_loc_type_e>() };
             p.set_diagnostic(d, l);
        })
        .def("export_data",  &FieldDiagPlotter::export_data, py::arg("filename"));

    // -----------------------------------------------------------------------
    // Free functions - polynomial solvers
    // -----------------------------------------------------------------------
    m.def("solve_quadratic", [](double a, double b, double c){
        double x0, x1; uint32_t n = solve_quadratic(a,b,c,&x0,&x1);
        std::vector<double> roots;
        if(n>=1) roots.push_back(x0); if(n>=2) roots.push_back(x1);
        return roots;
    }, "Solve a*x^2 + b*x + c = 0, returns list of real roots");

    m.def("solve_cubic", [](double a, double b, double c, double d){
        double x0, x1, x2; uint32_t n = solve_cubic(a,b,c,d,&x0,&x1,&x2);
        std::vector<double> roots;
        if(n>=1) roots.push_back(x0); if(n>=2) roots.push_back(x1);
        if(n>=3) roots.push_back(x2);
        return roots;
    }, "Solve a*x^3 + b*x^2 + c*x + d = 0, returns list of real roots");

    m.def("solve_quartic", [](double a, double b, double c, double d){
        double x0, x1, x2, x3; uint32_t n = solve_quartic(a,b,c,d,&x0,&x1,&x2,&x3);
        std::vector<double> roots;
        if(n>=1) roots.push_back(x0); if(n>=2) roots.push_back(x1);
        if(n>=3) roots.push_back(x2); if(n>=4) roots.push_back(x3);
        return roots;
    }, "Solve x^4+a*x^3+b*x^2+c*x+d=0 (monic), returns list of real roots");

    // -----------------------------------------------------------------------
    // Particle coordinate index constants
    // -----------------------------------------------------------------------
    m.attr("PARTICLE_T")  = PARTICLE_T;
    m.attr("PARTICLE_X")  = PARTICLE_X;
    m.attr("PARTICLE_VX") = PARTICLE_VX;
    m.attr("PARTICLE_Y")  = PARTICLE_Y;
    m.attr("PARTICLE_VY") = PARTICLE_VY;
    m.attr("PARTICLE_R")  = PARTICLE_R;
    m.attr("PARTICLE_VR") = PARTICLE_VR;
    m.attr("PARTICLE_W")  = PARTICLE_W;
    m.attr("PARTICLE_Z")  = PARTICLE_Z;
    m.attr("PARTICLE_VZ") = PARTICLE_VZ;
}
