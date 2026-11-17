// Shared helpers of the IBSimu Python bindings (ibsimu_py*.cpp).
#ifndef IBSIMU_PY_COMMON_HPP
#define IBSIMU_PY_COMMON_HPP

#include <pybind11/pybind11.h>
#include <exception>
#include <memory>
#include <typeinfo>

namespace py = pybind11;

// Python errors raised inside callbacks that the library calls with the GIL
// released are parked and re-raised in the calling thread (ibsimu_py.cpp).
void store_python_error();
bool python_error_pending();
void rethrow_python_error();

// Run f() (which calls into Python) and park any exception it raises.
template <typename R, typename F>
R guarded_call(R fallback, F f) {
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

// Hand a Python-created object over to C++, which will delete it: the
// default std::unique_ptr holder must let go of the pointer (clearing the
// instance's "owned" flag alone does not stop the holder from deleting it).
template <typename T>
T *release_to_cpp(py::object obj) {
    T *ptr = obj.cast<T *>();
    auto *inst = reinterpret_cast<py::detail::instance *>(obj.ptr());
    auto vh = inst->get_value_and_holder(py::detail::get_type_info(typeid(T)));
    if (vh.holder_constructed()) vh.template holder<std::unique_ptr<T>>().release();
    inst->owned = false;
    return ptr;
}

// Binding groups defined in the other translation units.
void init_linalg(py::module_ &m);
void init_geometry_extra(py::module_ &m);
void init_misc_extra(py::module_ &m);
void init_plotting_extra(py::module_ &m);

#endif
