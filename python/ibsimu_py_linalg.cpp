// IBSimu Python bindings: sparse matrices, vectors, preconditioners and the
// Harwell-Boeing matrix file format.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sstream>

#include "mvector.hpp"
#include "matrix.hpp"
#include "crowmatrix.hpp"
#include "ccolmatrix.hpp"
#include "coordmatrix.hpp"
#include "precond.hpp"
#include "ilu0_precond.hpp"
#include "ilu1_precond.hpp"
#include "diag_precond.hpp"
#include "empty_precond.hpp"
#include "hbio.hpp"
#include "ibsimu_py_common.hpp"

using darray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarray = py::array_t<int, py::array::c_style | py::array::forcecast>;

static Vector vector_from_array(darray a) {
    if (a.ndim() != 1) throw std::invalid_argument("Expected a one-dimensional array");
    return Vector((int)a.shape(0), a.data());
}

static void check_len(const py::array &a, py::ssize_t n, const char *what) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string("Array '") + what + "' has wrong length");
}

// The library checks indices only when built with SPM_RANGE_CHECK.
static void check_index(const Matrix &a, int i, int j) {
    if (i < 0 || i >= a.rows() || j < 0 || j >= a.columns())
        throw py::index_error("Matrix index out of range");
}

static std::string debug_string(const Matrix &mat) {
    std::ostringstream os;
    os << mat;
    return os.str();
}

// Compressed row (CRS) or column (CCS) matrix from scipy-style arrays.
template <typename M, typename Idx>
static M *compressed_from_arrays(int n, int m, iarray ptr, iarray idx, darray val, bool rowmajor,
                                 Idx idxref) {
    const int nmajor = rowmajor ? n : m;
    check_len(ptr, nmajor + 1, "ptr");
    const int nz = ptr.data()[nmajor];
    check_len(idx, nz, "indices");
    check_len(val, nz, "data");
    M *mat = new M(n, m);
    mat->set_nz(nz);
    for (int i = 0; i <= nmajor; i++) mat->ptr(i) = ptr.data()[i];
    for (int i = 0; i < nz; i++) {
        idxref(*mat, i) = idx.data()[i];
        mat->val(i) = val.data()[i];
    }
    return mat;
}

template <typename M, typename Idx>
static py::tuple compressed_to_arrays(const M &mat, bool rowmajor, Idx idxref) {
    const int nmajor = rowmajor ? mat.rows() : mat.columns();
    const int nz = mat.nz_elements();
    py::array_t<int> ptr(nmajor + 1), idx(nz);
    py::array_t<double> val(nz);
    for (int i = 0; i <= nmajor; i++) ptr.mutable_data()[i] = mat.ptr(i);
    for (int i = 0; i < nz; i++) {
        idx.mutable_data()[i] = idxref(mat, i);
        val.mutable_data()[i] = mat.val(i);
    }
    return py::make_tuple(ptr, idx, val);
}

void init_linalg(py::module_ &m) {
    // -----------------------------------------------------------------------
    // Vector
    // -----------------------------------------------------------------------
    py::class_<Vector>(m, "Vector", py::buffer_protocol(),
                       "Dense vector of the library's linear algebra (supports the buffer protocol)")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("n"), "Vector of n zeros")
        .def(py::init<int, double>(), py::arg("n"), py::arg("value"))
        .def(py::init(&vector_from_array), py::arg("array"), "Copy of a 1D array")
        .def(py::init<const Vector &>())
        .def_buffer([](Vector &v) -> py::buffer_info {
            return py::buffer_info(v.get_data(), sizeof(double),
                                   py::format_descriptor<double>::format(), 1,
                                   { (py::ssize_t)v.size() }, { (py::ssize_t)sizeof(double) });
        })
        .def("size", &Vector::size)
        .def("__len__", &Vector::size)
        .def("resize", &Vector::resize, py::arg("n"))
        .def("clear", &Vector::clear)
        .def("numpy", [](py::object self) {
            Vector &v = self.cast<Vector &>();
            return py::array_t<double>({ (py::ssize_t)v.size() }, { (py::ssize_t)sizeof(double) },
                                       v.get_data(), self);
        }, "Writable numpy view of the vector data")
        .def("__getitem__", [](const Vector &v, int i) {
            if (i < 0) i += v.size();
            if (i < 0 || i >= v.size()) throw py::index_error("Vector index out of range");
            return v[i];
        })
        .def("__setitem__", [](Vector &v, int i, double x) {
            if (i < 0) i += v.size();
            if (i < 0 || i >= v.size()) throw py::index_error("Vector index out of range");
            v[i] = x;
        })
        .def("__add__", [](const Vector &a, const Vector &b) { return Vector(a + b); })
        .def("__sub__", [](const Vector &a, const Vector &b) { return Vector(a - b); })
        .def("__neg__", [](const Vector &a) { return Vector(-a); })
        .def("__mul__", [](const Vector &a, double x) { return Vector(a * x); })
        .def("__rmul__", [](const Vector &a, double x) { return Vector(a * x); })
        .def("__iadd__", [](Vector &a, const Vector &b) -> Vector & { a += b; return a; })
        .def("__isub__", [](Vector &a, const Vector &b) -> Vector & { a -= b; return a; })
        .def("__imul__", [](Vector &a, double x) -> Vector & { a *= x; return a; })
        .def("__eq__", [](const Vector &a, const Vector &b) { return a == b; })
        .def("__ne__", [](const Vector &a, const Vector &b) { return a != b; })
        .def("fill", [](Vector &a, double x) { a = x; }, py::arg("value"))
        .def("dot", [](const Vector &a, const Vector &b) { return dot_prod(a, b); })
        .def("norm1", [](const Vector &a) { return norm1(a); })
        .def("norm2", [](const Vector &a) { return norm2(a); })
        .def("ssqr", [](const Vector &a) { return ssqr(a); })
        .def("min", [](const Vector &a) { return min(a); })
        .def("max", [](const Vector &a) { return max(a); })
        .def("min_abs", [](const Vector &a) { return min_abs(a); })
        .def("max_abs", [](const Vector &a) { return max_abs(a); })
        .def("__repr__", [](const Vector &v) {
            std::ostringstream os;
            os << "Vector(" << v.size() << ")";
            return os.str();
        });
    py::implicitly_convertible<py::array, Vector>();

    // -----------------------------------------------------------------------
    // Matrix (abstract base)
    // -----------------------------------------------------------------------
    py::class_<Matrix>(m, "Matrix")
        .def("rows", &Matrix::rows)
        .def("columns", &Matrix::columns)
        .def("shape", [](const Matrix &a) { return py::make_tuple(a.rows(), a.columns()); })
        .def("resize", &Matrix::resize, py::arg("n"), py::arg("m"))
        .def("clear", (void (Matrix::*)(void)) &Matrix::clear)
        .def("get", [](const Matrix &a, int i, int j) {
            check_index(a, i, j);
            return a.get(i, j);
        }, py::arg("i"), py::arg("j"))
        .def("set", [](Matrix &a, int i, int j, double x) {
            check_index(a, i, j);
            a.set(i, j) = x;
        }, py::arg("i"), py::arg("j"), py::arg("value"))
        .def("__getitem__", [](const Matrix &a, std::pair<int, int> ij) {
            check_index(a, ij.first, ij.second);
            return a.get(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix &a, std::pair<int, int> ij, double x) {
            check_index(a, ij.first, ij.second);
            a.set(ij.first, ij.second) = x;
        })
        .def("multiply_by_vector", [](const Matrix &a, const Vector &x) {
            Vector res;
            a.multiply_by_vector(res, x);
            return res;
        }, py::arg("x"), "Matrix-vector product A x")
        .def("__matmul__", [](const Matrix &a, const Vector &x) {
            Vector res;
            a.multiply_by_vector(res, x);
            return res;
        })
        .def("lower_unit_solve", [](const Matrix &a, const Vector &b) {
            Vector x;
            a.lower_unit_solve(x, b);
            return x;
        }, py::arg("b"), "Solve L x = b, L the strictly lower part with unit diagonal")
        .def("upper_diag_solve", [](const Matrix &a, const Vector &b) {
            Vector x;
            a.upper_diag_solve(x, b);
            return x;
        }, py::arg("b"), "Solve U x = b, U the upper part including the diagonal")
        .def("toarray", [](const Matrix &a) {
            py::array_t<double> d({ (py::ssize_t)a.rows(), (py::ssize_t)a.columns() });
            auto r = d.mutable_unchecked<2>();
            for (int i = 0; i < a.rows(); i++)
                for (int j = 0; j < a.columns(); j++) r(i, j) = a.get(i, j);
            return d;
        }, "Dense copy as a 2D array")
        .def("__str__", &debug_string);

    // -----------------------------------------------------------------------
    // CRowMatrix (compressed row storage)
    // -----------------------------------------------------------------------
    py::class_<CRowMatrix, Matrix>(m, "CRowMatrix")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("n"), py::arg("m"))
        .def(py::init<const CRowMatrix &>())
        .def(py::init<const CColMatrix &>())
        .def(py::init<const CoordMatrix &>())
        .def(py::init([](int n, int mm, iarray ptr, iarray col, darray val) {
            return compressed_from_arrays<CRowMatrix>(n, mm, ptr, col, val, true,
                [](CRowMatrix &a, int i) -> int & { return a.col(i); });
        }), py::arg("n"), py::arg("m"), py::arg("ptr"), py::arg("col"), py::arg("val"),
            "From CSR arrays (scipy.sparse.csr_matrix: indptr, indices, data)")
        .def("nz_elements", &CRowMatrix::nz_elements)
        .def("capacity", &CRowMatrix::capacity)
        .def("reserve", &CRowMatrix::reserve, py::arg("size"))
        .def("clear_element", (void (CRowMatrix::*)(int, int)) &CRowMatrix::clear, py::arg("i"), py::arg("j"))
        .def("order_ascending", &CRowMatrix::order_ascending)
        .def("check_ascending", &CRowMatrix::check_ascending)
        .def("construct_add", &CRowMatrix::construct_add, py::arg("i"), py::arg("j"), py::arg("val"),
             "Append element (i, j); rows must be built in increasing order")
        .def("set_row", [](CRowMatrix &a, int i, iarray col, darray val) {
            check_len(val, col.size(), "val");
            a.set_row(i, (int)col.size(), col.data(), val.data());
        }, py::arg("i"), py::arg("col"), py::arg("val"))
        .def("LU_solve", [](const CRowMatrix &a, const Vector &b) {
            Vector x;
            a.LU_solve(x, b);
            return x;
        }, py::arg("b"), "Solve (LU) x = b for a matrix holding an LU factorisation")
        .def("arrays", [](const CRowMatrix &a) {
            return compressed_to_arrays<CRowMatrix>(a, true,
                [](const CRowMatrix &b, int i) { return b.col(i); });
        }, "CSR arrays (ptr, col, val), e.g. for scipy.sparse.csr_matrix((val, col, ptr))");

    // -----------------------------------------------------------------------
    // CColMatrix (compressed column storage)
    // -----------------------------------------------------------------------
    py::class_<CColMatrix, Matrix>(m, "CColMatrix")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("n"), py::arg("m"))
        .def(py::init<const CColMatrix &>())
        .def(py::init<const CRowMatrix &>())
        .def(py::init<const CoordMatrix &>())
        .def(py::init([](int n, int mm, iarray ptr, iarray row, darray val) {
            return compressed_from_arrays<CColMatrix>(n, mm, ptr, row, val, false,
                [](CColMatrix &a, int i) -> int & { return a.row(i); });
        }), py::arg("n"), py::arg("m"), py::arg("ptr"), py::arg("row"), py::arg("val"),
            "From CSC arrays (scipy.sparse.csc_matrix: indptr, indices, data)")
        .def("nz_elements", &CColMatrix::nz_elements)
        .def("capacity", &CColMatrix::capacity)
        .def("reserve", &CColMatrix::reserve, py::arg("size"))
        .def("clear_element", (void (CColMatrix::*)(int, int)) &CColMatrix::clear, py::arg("i"), py::arg("j"))
        .def("order_ascending", &CColMatrix::order_ascending)
        .def("check_ascending", &CColMatrix::check_ascending)
        .def("construct_add", &CColMatrix::construct_add, py::arg("i"), py::arg("j"), py::arg("val"),
             "Append element (i, j); columns must be built in increasing order")
        .def("set_column", [](CColMatrix &a, int j, iarray row, darray val) {
            check_len(val, row.size(), "val");
            a.set_column(j, (int)row.size(), row.data(), val.data());
        }, py::arg("j"), py::arg("row"), py::arg("val"))
        .def("arrays", [](const CColMatrix &a) {
            return compressed_to_arrays<CColMatrix>(a, false,
                [](const CColMatrix &b, int i) { return b.row(i); });
        }, "CSC arrays (ptr, row, val), e.g. for scipy.sparse.csc_matrix((val, row, ptr))");

    // -----------------------------------------------------------------------
    // CoordMatrix (coordinate storage)
    // -----------------------------------------------------------------------
    py::class_<CoordMatrix, Matrix>(m, "CoordMatrix")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("n"), py::arg("m"))
        .def(py::init<const CoordMatrix &>())
        .def(py::init<const CRowMatrix &>())
        .def(py::init<const CColMatrix &>())
        .def(py::init([](int n, int mm, iarray row, iarray col, darray val) {
            check_len(col, row.size(), "col");
            check_len(val, row.size(), "val");
            const int nz = (int)row.size();
            auto *a = new CoordMatrix(n, mm);
            a->set_nz(nz);
            for (int i = 0; i < nz; i++) {
                a->row(i) = row.data()[i];
                a->col(i) = col.data()[i];
                a->val(i) = val.data()[i];
            }
            return a;
        }), py::arg("n"), py::arg("m"), py::arg("row"), py::arg("col"), py::arg("val"),
            "From COO arrays (scipy.sparse.coo_matrix: row, col, data)")
        .def("nz_elements", &CoordMatrix::nz_elements)
        .def("capacity", &CoordMatrix::capacity)
        .def("reserve", &CoordMatrix::reserve, py::arg("size"))
        .def("clear_element", (void (CoordMatrix::*)(int, int)) &CoordMatrix::clear, py::arg("i"), py::arg("j"))
        .def("order_ascending_row_column", &CoordMatrix::order_ascending_row_column)
        .def("order_ascending_column_row", &CoordMatrix::order_ascending_column_row)
        .def("arrays", [](const CoordMatrix &a) {
            const int nz = a.nz_elements();
            py::array_t<int> row(nz), col(nz);
            py::array_t<double> val(nz);
            for (int i = 0; i < nz; i++) {
                row.mutable_data()[i] = a.row(i);
                col.mutable_data()[i] = a.col(i);
                val.mutable_data()[i] = a.val(i);
            }
            return py::make_tuple(row, col, val);
        }, "COO arrays (row, col, val)");

    // -----------------------------------------------------------------------
    // Preconditioners (for EpotBiCGSTABSolver.set_preconditioner)
    // -----------------------------------------------------------------------
    py::class_<Precond>(m, "Precond")
        .def("prepare", &Precond::prepare, py::arg("A"), "Prepare the structure for matrix A")
        .def("construct", &Precond::construct, py::arg("A"), "Construct the preconditioner for matrix A")
        .def("clear", &Precond::clear)
        .def("is_prepared", &Precond::is_prepared)
        .def("typestring", &Precond::typestring)
        .def("solve", [](const Precond &p, const Vector &b) {
            Vector x(b.size());
            p.solve(x, b);
            return x;
        }, py::arg("b"), "Apply the preconditioner: solve M x = b")
        .def("__repr__", [](const Precond &p) { return "<Precond " + p.typestring() + ">"; });

    py::class_<ILU0_Precond, Precond>(m, "ILU0_Precond", "Incomplete LU factorisation with no fill-in")
        .def(py::init<>())
        .def("get_matrix", &ILU0_Precond::get_matrix, py::return_value_policy::reference_internal);
    py::class_<ILU1_Precond, Precond>(m, "ILU1_Precond", "Incomplete LU factorisation with fill level 1")
        .def(py::init<>())
        .def("get_matrix", &ILU1_Precond::get_matrix, py::return_value_policy::reference_internal);
    py::class_<Diag_Precond, Precond>(m, "Diag_Precond", "Diagonal (Jacobi) preconditioner")
        .def(py::init<>());
    py::class_<Empty_Precond, Precond>(m, "Empty_Precond", "No preconditioning")
        .def(py::init<>());

    // -----------------------------------------------------------------------
    // HBIO: Harwell-Boeing sparse matrix files
    // -----------------------------------------------------------------------
    py::class_<HBIO>(m, "HBIO", "Harwell-Boeing sparse matrix file reader/writer")
        .def(py::init<>())
        .def(py::init([](const std::string &fn) {
            auto *h = new HBIO();
            h->read(fn);
            return h;
        }), py::arg("filename"))
        .def("read", &HBIO::read, py::arg("filename"))
        .def("write", &HBIO::write, py::arg("filename"))
        .def("get_title", &HBIO::get_title)
        .def("set_title", &HBIO::set_title, py::arg("title"))
        .def("get_key", &HBIO::get_key)
        .def("set_key", &HBIO::set_key, py::arg("key"))
        .def("get_valacc", &HBIO::get_valacc)
        .def("set_valacc", &HBIO::set_valacc, py::arg("valacc"))
        .def("get_rhsacc", &HBIO::get_rhsacc)
        .def("set_rhsacc", &HBIO::set_rhsacc, py::arg("rhsacc"))
        .def("get_matrix", [](const HBIO &h) {
            CColMatrix a;
            h.get_matrix(a);
            return a;
        })
        .def("set_matrix", &HBIO::set_matrix, py::arg("matrix"))
        .def("get_rhs_vector", [](const HBIO &h) {
            Vector v;
            h.get_rhs_vector(v);
            return v;
        })
        .def("set_rhs_vector", &HBIO::set_rhs_vector, py::arg("rhs"))
        .def("get_solution_vector", [](const HBIO &h) {
            Vector v;
            h.get_solution_vector(v);
            return v;
        })
        .def("set_solution_vector", &HBIO::set_solution_vector, py::arg("solution"));
}
