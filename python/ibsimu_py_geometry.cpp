// IBSimu Python bindings: triangulated surfaces, STL files and the contents
// of DXF files (entities, blocks and layers).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "vec3d.hpp"
#include "transformation.hpp"
#include "vtriangle.hpp"
#include "stlfile.hpp"
#include "mydxffile.hpp"
#include "mydxfentities.hpp"
#include "mydxfline.hpp"
#include "mydxfarc.hpp"
#include "mydxfcircle.hpp"
#include "mydxflwpolyline.hpp"
#include "mydxfspline.hpp"
#include "mydxfinsert.hpp"
#include "mydxfmtext.hpp"
#include "mydxfblocks.hpp"
#include "mydxftables.hpp"
#include "ibsimu_py_common.hpp"

template <typename T>
static std::string debug_string(const T &obj) {
    std::ostringstream os;
    obj.debug_print(os);
    return os.str();
}

// DXF entities carry no RTTI-friendly Python type; downcast from get_type().
static py::object cast_entity(MyDXFEntity *e, py::handle parent) {
    const auto pol = py::return_value_policy::reference_internal;
    if (!e) return py::none();
    switch (e->get_type()) {
    case ENTITY_LINE:       return py::cast(static_cast<MyDXFLine *>(e), pol, parent);
    case ENTITY_LWPOLYLINE: return py::cast(static_cast<MyDXFLWPolyline *>(e), pol, parent);
    case ENTITY_SPLINE:     return py::cast(static_cast<MyDXFSpline *>(e), pol, parent);
    case ENTITY_ARC:        return py::cast(static_cast<MyDXFArc *>(e), pol, parent);
    case ENTITY_CIRCLE:     return py::cast(static_cast<MyDXFCircle *>(e), pol, parent);
    case ENTITY_MTEXT:      return py::cast(static_cast<MyDXFMText *>(e), pol, parent);
    case ENTITY_INSERT:     return py::cast(static_cast<MyDXFInsert *>(e), pol, parent);
    default:                return py::cast(e, pol, parent);
    }
}

static py::tuple bbox_tuple(const Vec3D &a, const Vec3D &b) { return py::make_tuple(a, b); }

void init_geometry_extra(py::module_ &m) {
    // -----------------------------------------------------------------------
    // Triangulated surfaces
    // -----------------------------------------------------------------------
    py::class_<VTriangle>(m, "VTriangle", "Triangle given by three vertex indices")
        .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("v1"), py::arg("v2"), py::arg("v3"))
        .def("__getitem__", [](const VTriangle &t, uint32_t i) {
            if (i > 2) throw py::index_error();
            return t[i];
        })
        .def("__setitem__", [](VTriangle &t, uint32_t i, uint32_t v) {
            if (i > 2) throw py::index_error();
            t[i] = v;
        })
        .def("__len__", [](const VTriangle &) { return 3; })
        .def("__repr__", [](const VTriangle &t) {
            std::ostringstream os;
            os << "VTriangle(" << t[0] << ", " << t[1] << ", " << t[2] << ")";
            return os.str();
        });

    py::class_<VTriangleSurface>(m, "VTriangleSurface", "Surface of vertices and triangles")
        .def(py::init<double>(), py::arg("vertex_matching_eps") = 1.0e-9)
        .def("set_vertex_matching_eps", &VTriangleSurface::set_vertex_matching_eps, py::arg("eps"))
        .def("vertexc", &VTriangleSurface::vertexc)
        .def("vertex", [](const VTriangleSurface &s, uint32_t i) {
            if (i >= s.vertexc()) throw py::index_error();
            return s.vertex(i);
        }, py::arg("i"))
        .def("trianglec", &VTriangleSurface::trianglec)
        .def("triangle", [](const VTriangleSurface &s, uint32_t i) {
            if (i >= s.trianglec()) throw py::index_error();
            return s.triangle(i);
        }, py::arg("i"))
        .def("add_vertex", &VTriangleSurface::add_vertex, py::arg("x"),
             "Add a vertex, reusing an existing one within vertex_matching_eps; returns its index")
        .def("add_vertex_no_check", &VTriangleSurface::add_vertex_no_check, py::arg("x"))
        .def("add_triangle", (uint32_t (VTriangleSurface::*)(const Vec3D &, const Vec3D &, const Vec3D &))
             &VTriangleSurface::add_triangle, py::arg("x1"), py::arg("x2"), py::arg("x3"))
        .def("add_triangle_idx", (uint32_t (VTriangleSurface::*)(uint32_t, uint32_t, uint32_t))
             &VTriangleSurface::add_triangle, py::arg("v1"), py::arg("v2"), py::arg("v3"))
        .def("clear", &VTriangleSurface::clear)
        .def("__str__", &debug_string<VTriangleSurface>);

    py::class_<VTriangleSurfaceSolid, VTriangleSurface>(m, "VTriangleSurfaceSolid",
            "Closed triangulated surface with an inside test")
        .def(py::init<double, double>(), py::arg("vertex_matching_eps") = 1.0e-9,
             py::arg("signed_volume_eps") = 1.0e-15)
        .def("set_signed_volume_eps", &VTriangleSurfaceSolid::set_signed_volume_eps, py::arg("eps"))
        .def("prepare_for_inside", &VTriangleSurfaceSolid::prepare_for_inside,
             "Build the search structures; call after adding triangles and before inside()")
        .def("inside", &VTriangleSurfaceSolid::inside, py::arg("x"))
        .def("get_bbox", [](const VTriangleSurfaceSolid &s) {
            Vec3D a, b;
            s.get_bbox(a, b);
            return bbox_tuple(a, b);
        }, "Bounding box as (min, max)")
        .def("remove_duplicate_triangles", &VTriangleSurfaceSolid::remove_duplicate_triangles)
        .def("check_data", &VTriangleSurfaceSolid::check_data,
             "Check that the surface is closed and consistently oriented (raises on failure)")
        .def("clear", &VTriangleSurfaceSolid::clear)
        .def("__str__", &debug_string<VTriangleSurfaceSolid>);

    // -----------------------------------------------------------------------
    // STLFile
    // -----------------------------------------------------------------------
    py::class_<STLFile>(m, "STLFile", "Triangulated solid read from (or written to) an STL file")
        .def(py::init<const std::string &, double, double>(), py::arg("filename"),
             py::arg("vertex_matching_eps") = 1.0e-9, py::arg("signed_volume_eps") = 1.0e-15)
        .def(py::init<const std::vector<Vec3D> &, const std::vector<VTriangle> &>(),
             py::arg("vertices"), py::arg("triangles"))
        .def("save", &STLFile::save, py::arg("filename"), py::arg("ascii") = true)
        .def("inside", &STLFile::inside, py::arg("x"))
        .def("get_bbox", [](const STLFile &s) {
            Vec3D a, b;
            s.get_bbox(a, b);
            return bbox_tuple(a, b);
        }, "Bounding box as (min, max)")
        .def("__str__", &debug_string<STLFile>);

    // -----------------------------------------------------------------------
    // DXF contents
    // -----------------------------------------------------------------------
    py::enum_<EntityType>(m, "DXFEntityType")
        .value("ENTITY_UNKNOWN", ENTITY_UNKNOWN)
        .value("ENTITY_LINE", ENTITY_LINE)
        .value("ENTITY_LWPOLYLINE", ENTITY_LWPOLYLINE)
        .value("ENTITY_SPLINE", ENTITY_SPLINE)
        .value("ENTITY_ARC", ENTITY_ARC)
        .value("ENTITY_CIRCLE", ENTITY_CIRCLE)
        .value("ENTITY_MTEXT", ENTITY_MTEXT)
        .value("ENTITY_INSERT", ENTITY_INSERT)
        .export_values();

    py::class_<MyDXFEntitySelection>(m, "MyDXFEntitySelection", "List of entity indices")
        .def(py::init<>())
        .def("size", &MyDXFEntitySelection::size)
        .def("__len__", &MyDXFEntitySelection::size)
        .def("add_entity", &MyDXFEntitySelection::add_entity, py::arg("a"))
        .def("__getitem__", [](const MyDXFEntitySelection &s, int a) {
            if (a < 0 || a >= (int)s.size()) throw py::index_error();
            return s(a);
        })
        .def("tolist", [](const MyDXFEntitySelection &s) {
            std::vector<uint32_t> v;
            for (uint32_t a = 0; a < s.size(); a++) v.push_back(s((int)a));
            return v;
        });

    py::class_<MyDXFEntity, std::unique_ptr<MyDXFEntity, py::nodelete>>(m, "MyDXFEntity")
        .def("get_type", &MyDXFEntity::get_type)
        .def("get_layer", &MyDXFEntity::get_layer)
        .def("set_layer", &MyDXFEntity::set_layer, py::arg("layer"))
        .def("get_handle", &MyDXFEntity::get_handle)
        .def("set_handle", &MyDXFEntity::set_handle, py::arg("handle"))
        .def("scale", &MyDXFEntity::scale, py::arg("dxf"), py::arg("s"))
        .def("translate", &MyDXFEntity::translate, py::arg("dxf"), py::arg("dx"))
        .def("rotate_z", &MyDXFEntity::rotate_z, py::arg("dxf"), py::arg("a"))
        .def("get_bbox", [](const MyDXFEntity &e, const MyDXFFile *dxf) {
            Vec3D a, b;
            Transformation t;
            e.get_bbox(a, b, dxf, &t);
            return bbox_tuple(a, b);
        }, py::arg("dxf"), "Bounding box as (min, max)")
        .def("__str__", &debug_string<MyDXFEntity>);

    py::class_<MyDXFPathEntity, MyDXFEntity, std::unique_ptr<MyDXFPathEntity, py::nodelete>>(m, "MyDXFPathEntity")
        .def("start", &MyDXFPathEntity::start)
        .def("end", &MyDXFPathEntity::end)
        .def("set_start", &MyDXFPathEntity::set_start, py::arg("s"))
        .def("set_end", &MyDXFPathEntity::set_end, py::arg("e"))
        .def("ray_cross", &MyDXFPathEntity::ray_cross, py::arg("x"), py::arg("y"),
             "Number of crossings of the ray from (x, y) towards +x with the path");

    py::class_<MyDXFLine, MyDXFPathEntity, std::unique_ptr<MyDXFLine, py::nodelete>>(m, "MyDXFLine");

    py::class_<MyDXFArc, MyDXFPathEntity, std::unique_ptr<MyDXFArc, py::nodelete>>(m, "MyDXFArc")
        .def("center", &MyDXFArc::center)
        .def("radius", &MyDXFArc::radius)
        .def("get_ang1", &MyDXFArc::get_ang1)
        .def("get_ang2", &MyDXFArc::get_ang2)
        .def("set_pc", &MyDXFArc::set_pc, py::arg("pc"))
        .def("set_r", &MyDXFArc::set_r, py::arg("r"))
        .def("set_ang1", &MyDXFArc::set_ang1, py::arg("ang1"))
        .def("set_ang2", &MyDXFArc::set_ang2, py::arg("ang2"));

    py::class_<MyDXFCircle, MyDXFPathEntity, std::unique_ptr<MyDXFCircle, py::nodelete>>(m, "MyDXFCircle")
        .def("center", &MyDXFCircle::center)
        .def("radius", &MyDXFCircle::radius)
        .def("set_center", &MyDXFCircle::set_center, py::arg("c"))
        .def("set_radius", &MyDXFCircle::set_radius, py::arg("r"));

    py::class_<MyDXFLWPolyline, MyDXFPathEntity, std::unique_ptr<MyDXFLWPolyline, py::nodelete>>(m, "MyDXFLWPolyline")
        .def("size", &MyDXFLWPolyline::size)
        .def("vertex", [](const MyDXFLWPolyline &p, uint32_t i) {
            if (i >= p.size()) throw py::index_error();
            return p.vertex(i);
        }, py::arg("i"))
        .def("vertices", [](const MyDXFLWPolyline &p) {
            std::vector<Vec3D> v;
            for (uint32_t i = 0; i < p.size(); i++) v.push_back(p.vertex(i));
            return v;
        })
        .def("closed", &MyDXFLWPolyline::closed);

    py::class_<MyDXFSpline, MyDXFPathEntity, std::unique_ptr<MyDXFSpline, py::nodelete>>(m, "MyDXFSpline")
        .def("closed", &MyDXFSpline::closed)
        .def("periodic", &MyDXFSpline::periodic)
        .def("rational", &MyDXFSpline::rational);

    py::class_<MyDXFMText, MyDXFEntity, std::unique_ptr<MyDXFMText, py::nodelete>>(m, "MyDXFMText");

    py::class_<MyDXFInsert, MyDXFEntity, std::unique_ptr<MyDXFInsert, py::nodelete>>(m, "MyDXFInsert")
        .def("block_name", &MyDXFInsert::block_name);

    py::class_<MyDXFEntities, std::unique_ptr<MyDXFEntities, py::nodelete>>(m, "MyDXFEntities")
        .def("size", &MyDXFEntities::size)
        .def("__len__", &MyDXFEntities::size)
        .def("get_entity", [](py::object self, uint32_t a) {
            MyDXFEntities &e = self.cast<MyDXFEntities &>();
            if (a >= e.size()) throw py::index_error();
            return cast_entity(e.get_entity(a), self);
        }, py::arg("a"), "Entity a, as its concrete type (MyDXFLine, MyDXFArc, ...)")
        .def("__getitem__", [](py::object self, uint32_t a) {
            MyDXFEntities &e = self.cast<MyDXFEntities &>();
            if (a >= e.size()) throw py::index_error();
            return cast_entity(e.get_entity(a), self);
        })
        .def("selection_all", &MyDXFEntities::selection_all, py::return_value_policy::take_ownership)
        .def("selection_layer", &MyDXFEntities::selection_layer, py::arg("layername"),
             py::return_value_policy::take_ownership)
        .def("selection_type", &MyDXFEntities::selection_type, py::arg("type"),
             py::return_value_policy::take_ownership)
        .def("selection_path_loop", &MyDXFEntities::selection_path_loop, py::arg("selection"),
             py::arg("eps") = 1.0e-6, py::return_value_policy::take_ownership,
             "Entities of the selection that form a closed path loop")
        .def("geom_same", &MyDXFEntities::geom_same, py::arg("a"), py::arg("b"), py::arg("eps") = 1.0e-6)
        .def("inside_loop", &MyDXFEntities::inside_loop, py::arg("selection"), py::arg("x"), py::arg("y"),
             py::arg("eps") = 1.0e-6)
        .def("get_bbox", [](const MyDXFEntities &e, const MyDXFEntitySelection *sel, const MyDXFFile *dxf) {
            Vec3D a, b;
            Transformation t;
            e.get_bbox(sel, a, b, dxf, &t);
            return bbox_tuple(a, b);
        }, py::arg("selection"), py::arg("dxf"), "Bounding box of the selection as (min, max)")
        .def("scale", &MyDXFEntities::scale, py::arg("selection"), py::arg("dxf"), py::arg("s"))
        .def("translate", &MyDXFEntities::translate, py::arg("selection"), py::arg("dxf"), py::arg("dx"))
        .def("rotate_z", &MyDXFEntities::rotate_z, py::arg("selection"), py::arg("a"))
        .def("remove", &MyDXFEntities::remove, py::arg("selection"))
        .def("explode", (void (MyDXFEntities::*)(MyDXFEntitySelection *, MyDXFFile *)) &MyDXFEntities::explode,
             py::arg("selection"), py::arg("dxf"), "Replace the selected INSERTs by the entities of their blocks")
        .def("__str__", &debug_string<MyDXFEntities>);

    py::class_<MyDXFBlock, std::unique_ptr<MyDXFBlock, py::nodelete>>(m, "MyDXFBlock")
        .def("name", &MyDXFBlock::name)
        .def("get_entities", (MyDXFEntities *(MyDXFBlock::*)(void)) &MyDXFBlock::get_entities,
             py::return_value_policy::reference_internal)
        .def("scale", &MyDXFBlock::scale, py::arg("dxf"), py::arg("s"))
        .def("translate", &MyDXFBlock::translate, py::arg("dxf"), py::arg("dx"))
        .def("__str__", &debug_string<MyDXFBlock>);

    py::class_<MyDXFBlocks, std::unique_ptr<MyDXFBlocks, py::nodelete>>(m, "MyDXFBlocks")
        .def("size", &MyDXFBlocks::size)
        .def("__len__", &MyDXFBlocks::size)
        .def("__getitem__", [](MyDXFBlocks &b, int a) {
            if (a < 0 || a >= (int)b.size()) throw py::index_error();
            return b(a);
        }, py::return_value_policy::reference_internal)
        .def("get_by_name", (MyDXFBlock *(MyDXFBlocks::*)(const std::string &)) &MyDXFBlocks::get_by_name,
             py::arg("name"), py::return_value_policy::reference_internal)
        .def("clear", &MyDXFBlocks::clear)
        .def("__str__", &debug_string<MyDXFBlocks>);

    py::class_<MyDXFTableEntry, std::unique_ptr<MyDXFTableEntry, py::nodelete>>(m, "MyDXFTableEntry")
        .def("__str__", &debug_string<MyDXFTableEntry>);
    py::class_<MyDXFTableEntryLayer, MyDXFTableEntry, std::unique_ptr<MyDXFTableEntryLayer, py::nodelete>>(
            m, "MyDXFTableEntryLayer")
        .def("name", &MyDXFTableEntryLayer::name)
        .def("set_name", &MyDXFTableEntryLayer::set_name, py::arg("name"));

    py::class_<MyDXFTable, std::unique_ptr<MyDXFTable, py::nodelete>>(m, "MyDXFTable")
        .def("size", &MyDXFTable::size)
        .def("__len__", &MyDXFTable::size)
        .def("get_entry", [](MyDXFTable &t, uint32_t a) {
            if (a >= t.size()) throw py::index_error();
            return t.get_entry(a);
        }, py::arg("a"), py::return_value_policy::reference_internal)
        .def("__str__", &debug_string<MyDXFTable>);

    py::class_<MyDXFTables, std::unique_ptr<MyDXFTables, py::nodelete>>(m, "MyDXFTables")
        .def("get_layers_table", (MyDXFTable *(MyDXFTables::*)(void)) &MyDXFTables::get_layers_table,
             py::return_value_policy::reference_internal)
        .def("layer_names", [](MyDXFTables &t) {
            // Layer table entries are all MyDXFTableEntryLayer.
            std::vector<std::string> names;
            MyDXFTable *lt = t.get_layers_table();
            for (uint32_t a = 0; lt && a < lt->size(); a++)
                names.push_back(static_cast<MyDXFTableEntryLayer *>(lt->get_entry(a))->name());
            return names;
        })
        .def("__str__", &debug_string<MyDXFTables>);
}
