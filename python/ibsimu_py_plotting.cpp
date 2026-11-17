// IBSimu Python bindings: the building blocks of the plots (frames, graphs,
// colormaps, palettes and legends), the 3D geometry view and the interactive
// GTK plotter.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cairo.h>
#ifdef CAIRO_HAS_PDF_SURFACE
#include <cairo-pdf.h>
#endif
#ifdef CAIRO_HAS_SVG_SURFACE
#include <cairo-svg.h>
#endif
#include <sstream>

#include "error.hpp"
#include "palette.hpp"
#include "coordmapper.hpp"
#include "graph.hpp"
#include "colormap.hpp"
#include "graph3d.hpp"
#include "xygraph.hpp"
#include "meshcolormap.hpp"
#include "fieldgraph.hpp"
#include "particlegraph.hpp"
#include "solidgraph.hpp"
#include "eqpotgraph.hpp"
#include "meshgraph.hpp"
#include "legend.hpp"
#include "frame.hpp"
#include "geom3dplot.hpp"
#ifdef GTK3
#include "softwarerenderer.hpp"
#include "gtkplotter.hpp"
#include "gtkwindow.hpp"
#endif
#include "ibsimu_py_common.hpp"

static py::tuple bbox_tuple(Graph &g) {
    double b[4];
    g.get_bbox(b);
    return py::make_tuple(b[0], b[1], b[2], b[3]);
}

// ---------------------------------------------------------------------------
// Drawing a Frame to a file (Frame::draw needs a cairo context)
// ---------------------------------------------------------------------------
static void check_cairo(cairo_surface_t *surface, cairo_t *cairo) {
    if (cairo_surface_status(surface))
        throw std::runtime_error("Error creating cairo surface");
    if (cairo_status(cairo))
        throw std::runtime_error(std::string("Error creating cairo context: ") +
                                 cairo_status_to_string(cairo_status(cairo)));
}

static void frame_to_surface(Frame &frame, cairo_surface_t *surface, int width, int height,
                             bool clear) {
    cairo_t *cairo = cairo_create(surface);
    try {
        check_cairo(surface, cairo);
        if (clear) {
            cairo_rectangle(cairo, 0.0, 0.0, width, height);
            cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 1.0);
            cairo_fill(cairo);
        }
        frame.set_geometry(width, height, 0, 0);
        frame.draw(cairo);
    } catch (...) {
        cairo_destroy(cairo);
        throw;
    }
    cairo_destroy(cairo);
}

static void frame_plot_png(Frame &frame, const std::string &fn, int width, int height) {
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    try {
        frame_to_surface(frame, surface, width, height, true);
        if (cairo_surface_write_to_png(surface, fn.c_str()) != CAIRO_STATUS_SUCCESS)
            throw std::runtime_error("Error writing PNG file " + fn);
    } catch (...) {
        cairo_surface_destroy(surface);
        throw;
    }
    cairo_surface_destroy(surface);
}

template <cairo_surface_t *(*Create)(const char *, double, double)>
static void frame_plot_vector(Frame &frame, const std::string &fn, int width, int height) {
    cairo_surface_t *surface = Create(fn.c_str(), width, height);
    try {
        frame_to_surface(frame, surface, width, height, false);
    } catch (...) {
        cairo_surface_destroy(surface);
        throw;
    }
    cairo_surface_finish(surface);
    cairo_surface_destroy(surface);
}

#ifdef GTK3
// Argument storage for gtk_init(): must stay valid for the program lifetime.
static int g_gtk_argc = 1;
static char g_gtk_arg0[] = "ibsimu";
static char *g_gtk_argv_storage[] = { g_gtk_arg0, nullptr };
static char **g_gtk_argv = g_gtk_argv_storage;
#endif

void init_plotting_extra(py::module_ &m) {
    // -----------------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------------
    py::enum_<PlotAxis>(m, "PlotAxis")
        .value("PLOT_AXIS_X1", PLOT_AXIS_X1)
        .value("PLOT_AXIS_Y1", PLOT_AXIS_Y1)
        .value("PLOT_AXIS_X2", PLOT_AXIS_X2)
        .value("PLOT_AXIS_Y2", PLOT_AXIS_Y2)
        .value("PLOT_AXIS_Z", PLOT_AXIS_Z)
        .export_values();

    py::enum_<PlotFixedMode>(m, "PlotFixedMode")
        .value("PLOT_FIXED_ASPECT_DISABLED", PLOT_FIXED_ASPECT_DISABLED)
        .value("PLOT_FIXED_ASPECT_EXTEND_RANGE", PLOT_FIXED_ASPECT_EXTEND_RANGE)
        .value("PLOT_FIXED_ASPECT_INCREASE_MARGIN", PLOT_FIXED_ASPECT_INCREASE_MARGIN)
        .export_values();

    py::enum_<legend_position_e>(m, "LegendPosition")
        .value("LEGEND_POS_BOTTOM_LEFT", LEGEND_POS_BOTTOM_LEFT)
        .value("LEGEND_POS_BOTTOM_CENTER", LEGEND_POS_BOTTOM_CENTER)
        .value("LEGEND_POS_BOTTOM_RIGHT", LEGEND_POS_BOTTOM_RIGHT)
        .value("LEGEND_POS_MIDDLE_LEFT", LEGEND_POS_MIDDLE_LEFT)
        .value("LEGEND_POS_MIDDLE_CENTER", LEGEND_POS_MIDDLE_CENTER)
        .value("LEGEND_POS_MIDDLE_RIGHT", LEGEND_POS_MIDDLE_RIGHT)
        .value("LEGEND_POS_TOP_LEFT", LEGEND_POS_TOP_LEFT)
        .value("LEGEND_POS_TOP_CENTER", LEGEND_POS_TOP_CENTER)
        .value("LEGEND_POS_TOP_RIGHT", LEGEND_POS_TOP_RIGHT)
        .export_values();

    py::enum_<line_style_e>(m, "LineStyle")
        .value("XYGRAPH_LINE_DISABLE", XYGRAPH_LINE_DISABLE)
        .value("XYGRAPH_LINE_SOLID", XYGRAPH_LINE_SOLID)
        .export_values();

    py::enum_<point_style_e>(m, "PointStyle")
        .value("XYGRAPH_POINT_DISABLE", XYGRAPH_POINT_DISABLE)
        .value("XYGRAPH_POINT_CIRCLE", XYGRAPH_POINT_CIRCLE)
        .value("XYGRAPH_POINT_BOX", XYGRAPH_POINT_BOX)
        .export_values();

    // -----------------------------------------------------------------------
    // Palette
    // -----------------------------------------------------------------------
    py::class_<Palette>(m, "Palette", "Colour palette: colours at values, interpolated linearly")
        .def(py::init<>(), "White (0) to black (1)")
        .def(py::init([](const std::vector<std::pair<Vec3D, double>> &entries) {
            std::vector<Palette::Entry> e;
            for (const auto &p : entries) e.push_back(Palette::Entry(p.first, p.second));
            return new Palette(e);
        }), py::arg("entries"), "From a list of (Vec3D rgb colour, value) pairs; values are normalised")
        .def("__call__", &Palette::operator(), py::arg("x"), "RGB colour (Vec3D) at x in [0, 1]")
        .def("clear", &Palette::clear)
        .def("push_back", &Palette::push_back, py::arg("color"), py::arg("val"))
        .def("normalize", &Palette::normalize, "Scale the entry values to [0, 1]")
        .def("get_steps", &Palette::get_steps)
        .def("set_steps", &Palette::set_steps, py::arg("steps"),
             "Use a stepped palette with this many steps (0 for continuous)")
        .def("__str__", [](const Palette &p) {
            std::ostringstream os;
            p.debug_print(os);
            return os.str();
        });

    // -----------------------------------------------------------------------
    // Coordmapper
    // -----------------------------------------------------------------------
    py::class_<Coordmapper>(m, "Coordmapper", "Mapping from plot coordinates to canvas coordinates")
        .def(py::init<>())
        .def(py::init<double, double, double, double>(), py::arg("xx"), py::arg("x0"), py::arg("yy"),
             py::arg("y0"))
        .def("set_transformation", &Coordmapper::set_transformation, py::arg("xx"), py::arg("x0"),
             py::arg("yy"), py::arg("y0"))
        .def("transform", [](const Coordmapper &c, double x, double y) {
            c.transform(x, y);
            return py::make_tuple(x, y);
        }, py::arg("x"), py::arg("y"))
        .def("inv_transform", [](const Coordmapper &c, double x, double y) {
            c.inv_transform(x, y);
            return py::make_tuple(x, y);
        }, py::arg("x"), py::arg("y"));

    // -----------------------------------------------------------------------
    // Graphs
    // -----------------------------------------------------------------------
    py::class_<Graph>(m, "Graph", "Something drawable in a Frame")
        .def("get_bbox", &bbox_tuple, "Bounding box (xmin, ymin, xmax, ymax) in plot coordinates");

    py::class_<Graph3D, Graph>(m, "Graph3D")
        .def("set_view", &Graph3D::set_view, py::arg("view"), py::arg("level"),
             "Plane of the 3D mesh to show (level is the node index along the normal)");

    py::class_<XYGraph, Graph>(m, "XYGraph", "Line/point graph of (x, y) data")
        .def(py::init<>())
        .def(py::init<const std::vector<double> &, const std::vector<double> &>(), py::arg("xdata"),
             py::arg("ydata"))
        .def("set_data", &XYGraph::set_data, py::arg("xdata"), py::arg("ydata"))
        .def("set_line_width", &XYGraph::set_line_width, py::arg("linewidth"))
        .def("set_color", &XYGraph::set_color, py::arg("color"))
        .def("set_line_style", &XYGraph::set_line_style, py::arg("linestyle"), py::arg("linewidth") = 1.0)
        .def("set_point_style", &XYGraph::set_point_style, py::arg("pointstyle"), py::arg("filled") = true,
             py::arg("scale") = 1.0)
        .def("set_histogram", &XYGraph::set_histogram, py::arg("histo"))
        .def("extend_histogram", &XYGraph::extend_histogram, py::arg("extend"));

    py::class_<Colormap, Graph>(m, "Colormap")
        .def("palette", (Palette & (Colormap::*)(void)) &Colormap::palette,
             py::return_value_policy::reference_internal, "The palette (modifiable in place)")
        .def("zscale", &Colormap::zscale, py::arg("val"))
        .def("zscale_inv", &Colormap::zscale_inv, py::arg("val"));

    py::class_<MeshColormap, Colormap>(m, "MeshColormap", "Colour map of data on a regular 2D grid")
        .def(py::init<>())
        .def(py::init([](std::vector<double> range, size_t n, size_t mm, const std::vector<double> &data) {
            if (range.size() != 4) throw std::invalid_argument("range must be (xmin, ymin, xmax, ymax)");
            return new MeshColormap(range.data(), n, mm, data);
        }), py::arg("range"), py::arg("n"), py::arg("m"), py::arg("data"),
            "range = (xmin, ymin, xmax, ymax); data has n*m values, x index fastest")
        .def("set_data", [](MeshColormap &c, std::vector<double> range, size_t n, size_t mm,
                            const std::vector<double> &data) {
            if (range.size() != 4) throw std::invalid_argument("range must be (xmin, ymin, xmax, ymax)");
            c.set_data(range.data(), n, mm, data);
        }, py::arg("range"), py::arg("n"), py::arg("m"), py::arg("data"))
        .def("clear_data", &MeshColormap::clear_data)
        .def("has_data", &MeshColormap::has_data)
        .def("get_value", &MeshColormap::get_value, py::arg("x"), py::arg("y"),
             "Data value at (x, y) with the current interpolation")
        .def("set_palette", &MeshColormap::set_palette, py::arg("palette"))
        .def("set_zscale", &MeshColormap::set_zscale, py::arg("zscale"))
        .def("get_zscale", &MeshColormap::get_zscale)
        .def("set_interpolation", &MeshColormap::set_interpolation, py::arg("interpolation"))
        .def("get_interpolation", &MeshColormap::get_interpolation)
        .def("set_zrange", &MeshColormap::set_zrange, py::arg("min"), py::arg("max"))
        .def("get_zrange", [](const MeshColormap &c) {
            double mn, mx;
            c.get_zrange(mn, mx);
            return py::make_tuple(mn, mx);
        });

    py::class_<FieldGraph, Graph3D, MeshColormap>(m, "FieldGraph", "Colour map of a field on the geometry mesh")
        .def(py::init<const Geometry &, const Field *, field_type_e>(), py::arg("geom"), py::arg("field"),
             py::arg("field_type"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("field_type", &FieldGraph::field_type)
        .def("enable", &FieldGraph::enable, py::arg("enable"))
        .def("set_zrange", &FieldGraph::set_zrange, py::arg("zmin"), py::arg("zmax"))
        .def("build_plot", &FieldGraph::build_plot)
        .def("get_bbox", [](FieldGraph &g) { return bbox_tuple(static_cast<Graph3D &>(g)); });

    py::class_<ParticleGraph, Graph3D>(m, "ParticleGraph", "Particle trajectories")
        .def(py::init<const Geometry &, const ParticleDataBase &, uint32_t, uint32_t, bool>(), py::arg("geom"),
             py::arg("pdb"), py::arg("particle_div") = 11, py::arg("particle_offset") = 0,
             py::arg("qm_discr") = true, py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("set_particle_div", &ParticleGraph::set_particle_div, py::arg("particle_div"),
             py::arg("particle_offset") = 0)
        .def("set_qm_discretation", &ParticleGraph::set_qm_discretation, py::arg("qm_discr"))
        .def("add_color", &ParticleGraph::add_color, py::arg("color"))
        .def("clear_colors", &ParticleGraph::clear_colors);

    py::class_<SolidGraph, Graph3D>(m, "SolidGraph", "Solids of the geometry")
        .def(py::init<const Geometry &>(), py::arg("geom"), py::keep_alive<1, 2>())
        .def("disable_cache", &SolidGraph::disable_cache);

    py::class_<EqPotGraph, Graph3D>(m, "EqPotGraph", "Equipotential lines")
        .def(py::init<const MeshScalarField &, const Geometry &>(), py::arg("epot"), py::arg("geom"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("disable_cache", &EqPotGraph::disable_cache)
        .def("set_eqlines_manual", &EqPotGraph::set_eqlines_manual, py::arg("pot"))
        .def("set_eqlines_auto", &EqPotGraph::set_eqlines_auto, py::arg("N"));

    py::class_<MeshGraph, Graph3D>(m, "MeshGraph", "Mesh lines")
        .def(py::init<const Geometry &>(), py::arg("geom"), py::keep_alive<1, 2>());

    // -----------------------------------------------------------------------
    // Legends and Frame
    // -----------------------------------------------------------------------
    py::class_<LegendEntry>(m, "LegendEntry", "Legend line showing a sample of a graph and a label")
        .def(py::init<Graph &, const std::string &>(), py::arg("graph"), py::arg("label"), py::keep_alive<1, 2>())
        .def("set_font_size", &LegendEntry::set_font_size, py::arg("fontsize"));

    py::class_<Frame>(m, "Frame", "Plot frame with rulers, labels, graphs and legends")
        .def(py::init<>())
        .def("set_font_size", &Frame::set_font_size, py::arg("size"))
        .def("get_font_size", &Frame::get_font_size)
        .def("set_background", [](Frame &f, Vec3D c) { f.set_background(c); }, py::arg("color"))
        .def("set_foreground", [](Frame &f, Vec3D c) { f.set_foreground(c); }, py::arg("color"))
        .def("get_coordmapper", &Frame::get_coordmapper, py::arg("xaxis"), py::arg("yaxis"),
             "Plot-to-canvas mapping of the last drawing")
        .def("get_margins", [](const Frame &f) {
            double mg[4];
            f.get_margins(mg);
            return py::make_tuple(mg[0], mg[1], mg[2], mg[3]);
        })
        .def("get_frame_edges", [](const Frame &f) {
            double e[4];
            f.get_frame_edges(e);
            return py::make_tuple(e[0], e[1], e[2], e[3]);
        })
        .def("set_title", &Frame::set_title, py::arg("title"))
        .def("set_axis_label", &Frame::set_axis_label, py::arg("axis"), py::arg("label"))
        .def("force_enable_ruler", &Frame::force_enable_ruler, py::arg("axis"), py::arg("force"))
        .def("ruler_autorange_enable", &Frame::ruler_autorange_enable, py::arg("axis"), py::arg("min"),
             py::arg("max"))
        .def("set_ranges", &Frame::set_ranges, py::arg("axis"), py::arg("min"), py::arg("max"))
        .def("get_ranges", [](const Frame &f, PlotAxis axis) {
            double mn, mx;
            f.get_ranges(axis, mn, mx);
            return py::make_tuple(mn, mx);
        }, py::arg("axis"))
        .def("set_fixed_aspect", &Frame::set_fixed_aspect, py::arg("mode"))
        .def("set_automargin", &Frame::set_automargin, py::arg("enable"))
        .def("add_graph", &Frame::add_graph, py::arg("xaxis"), py::arg("yaxis"), py::arg("graph"),
             py::arg("legend") = nullptr, py::keep_alive<1, 4>(), py::keep_alive<1, 5>(),
             "Add a graph (and optionally its legend entry); the frame keeps them alive")
        .def("clear_graphs", &Frame::clear_graphs)
        .def("enable_colormap_legend", &Frame::enable_colormap_legend, py::arg("enable"))
        .def("enable_legend", &Frame::enable_legend, py::arg("enable"))
        .def("set_legend_position", &Frame::set_legend_position, py::arg("pos"))
        .def("plot_png", &frame_plot_png, py::arg("filename"), py::arg("width") = 800, py::arg("height") = 600)
#ifdef CAIRO_HAS_PDF_SURFACE
        .def("plot_pdf", &frame_plot_vector<cairo_pdf_surface_create>, py::arg("filename"),
             py::arg("width") = 800, py::arg("height") = 600)
#endif
#ifdef CAIRO_HAS_SVG_SURFACE
        .def("plot_svg", &frame_plot_vector<cairo_svg_surface_create>, py::arg("filename"),
             py::arg("width") = 800, py::arg("height") = 600)
#endif
        ;

    // -----------------------------------------------------------------------
    // Geom3DPlot: 3D view of a geometry (and beam)
    // -----------------------------------------------------------------------
    py::class_<Geom3DPlot>(m, "Geom3DPlot", "3D view of the solids, cut planes and particle trajectories")
        .def(py::init<const Geometry &, const ParticleDataBase *>(), py::arg("geom"), py::arg("pdb") = nullptr,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("set_size", &Geom3DPlot::set_size, py::arg("width"), py::arg("height"))
        .def("set_model_transformation", &Geom3DPlot::set_model_transformation, py::arg("modeltrans"))
        .def("get_model_transformation", &Geom3DPlot::get_model_transformation)
        .def("set_projection_frustum", &Geom3DPlot::set_projection_frustum, py::arg("xnear"), py::arg("xfar"),
             py::arg("zoom"))
        .def("get_projection_frustum", [](const Geom3DPlot &p) {
            double n, f, z;
            p.get_projection_frustum(n, f, z);
            return py::make_tuple(n, f, z);
        })
        .def("set_projection_zoom", &Geom3DPlot::set_projection_zoom, py::arg("zoom"))
        .def("get_projection_zoom", &Geom3DPlot::get_projection_zoom)
        .def("set_view_look_at", &Geom3DPlot::set_view_look_at, py::arg("camera"), py::arg("target"),
             py::arg("up"))
        .def("get_view_look_at", [](const Geom3DPlot &p) {
            Vec3D c, t, u;
            p.get_view_look_at(c, t, u);
            return py::make_tuple(c, t, u);
        }, "(camera, target, up)")
        .def("rebuild_model", &Geom3DPlot::rebuild_model)
        .def("reset_camera_and_rotation", &Geom3DPlot::reset_camera_and_rotation)
        .def("set_surface_triangle_color_range", &Geom3DPlot::set_surface_triangle_color_range, py::arg("min"),
             py::arg("max"))
        .def("get_surface_triangle_color_range", [](const Geom3DPlot &p) {
            double mn, mx;
            p.get_surface_triangle_color_range(mn, mx);
            return py::make_tuple(mn, mx);
        })
        .def("set_surface_triangle_data", [](py::object self, py::object data) {
            // The plot keeps a pointer: store a copy on the Python object.
            Geom3DPlot &p = self.cast<Geom3DPlot &>();
            if (data.is_none()) {
                p.set_surface_triangle_data(nullptr);
                self.attr("_surface_triangle_data") = py::none();
                return;
            }
            auto *v = new std::vector<double>(data.cast<std::vector<double>>());
            py::capsule owner(v, [](void *ptr) { delete static_cast<std::vector<double> *>(ptr); });
            self.attr("_surface_triangle_data") = owner;
            p.set_surface_triangle_data(v);
        }, py::arg("data"), "Values on the surface triangles (Geometry.surface_trianglec() values)")
        .def("set_surface_triangle_data_plot", &Geom3DPlot::set_surface_triangle_data_plot, py::arg("enable"))
        .def("get_surface_triangle_data_plot", &Geom3DPlot::get_surface_triangle_data_plot)
        .def("set_particle_div", &Geom3DPlot::set_particle_div, py::arg("particle_div"),
             py::arg("particle_offset") = 0)
        .def("get_particle_div", &Geom3DPlot::get_particle_div)
        .def("get_particle_offset", &Geom3DPlot::get_particle_offset)
        .def("set_bbox", &Geom3DPlot::set_bbox, py::arg("bbox"))
        .def("get_bbox", &Geom3DPlot::get_bbox)
        .def("set_clevel", &Geom3DPlot::set_clevel, py::arg("direction"), py::arg("level"),
             "Cut plane level (node index) in direction 0-5 (-x, +x, -y, +y, -z, +z)")
        .def("get_clevel", &Geom3DPlot::get_clevel, py::arg("direction"))
#ifdef GTK3
        .def("plot_png", [](Geom3DPlot &p, const std::string &fn, int width, int height) {
            // Render with the software renderer into an image surface.
            cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
            if (cairo_surface_status(surface)) {
                cairo_surface_destroy(surface);
                throw std::runtime_error("Error creating cairo surface");
            }
            cairo_t *cairo = cairo_create(surface);
            try {
                SoftwareRenderer r(surface);
                p.set_size(width, height);
                p.draw(cairo, &r);
                if (cairo_surface_write_to_png(surface, fn.c_str()) != CAIRO_STATUS_SUCCESS)
                    throw std::runtime_error("Error writing PNG file " + fn);
            } catch (...) {
                cairo_destroy(cairo);
                cairo_surface_destroy(surface);
                throw;
            }
            cairo_destroy(cairo);
            cairo_surface_destroy(surface);
        }, py::arg("filename"), py::arg("width") = 800, py::arg("height") = 600,
           "Render the view with the software renderer to a PNG file")
#endif
        ;

#ifdef GTK3
    // -----------------------------------------------------------------------
    // GTKPlotter: the interactive viewer (needs a display)
    // -----------------------------------------------------------------------
    py::class_<GTKWindow, std::unique_ptr<GTKWindow, py::nodelete>>(m, "GTKWindow");

    py::class_<GTKPlotter>(m, "GTKPlotter",
            "Interactive GTK viewer. Set the data, open windows, then call run(), which returns\n"
            "when the last window is closed. Needs a display.")
        .def(py::init([]() { return new GTKPlotter(&g_gtk_argc, &g_gtk_argv); }))
        .def("run", &GTKPlotter::run, py::call_guard<py::gil_scoped_release>())
        .def("set_geometry", &GTKPlotter::set_geometry, py::arg("geom"), py::keep_alive<1, 2>())
        .def("set_epot", &GTKPlotter::set_epot, py::arg("epot"), py::keep_alive<1, 2>())
        .def("set_efield", &GTKPlotter::set_efield, py::arg("efield"), py::keep_alive<1, 2>())
        .def("set_scharge", &GTKPlotter::set_scharge, py::arg("scharge"), py::keep_alive<1, 2>())
        .def("set_trajdens", &GTKPlotter::set_trajdens, py::arg("tdens"), py::keep_alive<1, 2>())
        .def("set_bfield", &GTKPlotter::set_bfield, py::arg("bfield"), py::keep_alive<1, 2>())
        .def("set_particledatabase", &GTKPlotter::set_particledatabase, py::arg("pdb"), py::keep_alive<1, 2>())
        .def("set_surface_triangle_data", [](py::object self, py::object data) {
            GTKPlotter &p = self.cast<GTKPlotter &>();
            if (data.is_none()) {
                p.set_surface_triangle_data(nullptr);
                self.attr("_surface_triangle_data") = py::none();
                return;
            }
            auto *v = new std::vector<double>(data.cast<std::vector<double>>());
            py::capsule owner(v, [](void *ptr) { delete static_cast<std::vector<double> *>(ptr); });
            self.attr("_surface_triangle_data") = owner;
            p.set_surface_triangle_data(v);
        }, py::arg("data"))
        .def("get_geometry", &GTKPlotter::get_geometry, py::return_value_policy::reference)
        .def("get_epot", &GTKPlotter::get_epot, py::return_value_policy::reference)
        .def("get_efield", &GTKPlotter::get_efield, py::return_value_policy::reference)
        .def("get_scharge", &GTKPlotter::get_scharge, py::return_value_policy::reference)
        .def("get_trajdens", &GTKPlotter::get_trajdens, py::return_value_policy::reference)
        .def("get_bfield", &GTKPlotter::get_bfield, py::return_value_policy::reference)
        .def("get_particledatabase", &GTKPlotter::get_particledatabase, py::return_value_policy::reference)
        .def("opengl", &GTKPlotter::opengl)
        .def("force_software_renderer", &GTKPlotter::force_software_renderer)
        .def("new_geometry_plot_window", &GTKPlotter::new_geometry_plot_window, py::return_value_policy::reference)
        .def("new_geometry_3d_plot_window", &GTKPlotter::new_geometry_3d_plot_window,
             py::return_value_policy::reference)
        .def("new_particle_plot_window", &GTKPlotter::new_particle_plot_window, py::arg("axis"), py::arg("level"),
             py::arg("type"), py::arg("diagx"), py::arg("diagy"), py::return_value_policy::reference)
        .def("new_field_plot_window", [](GTKPlotter &p, size_t N, const Vec3D &x1, const Vec3D &x2,
                                         std::vector<field_type_e> diag, std::vector<field_loc_type_e> loc) {
            if (diag.size() != 2 || loc.size() != 2)
                throw std::invalid_argument("diag and loc need two entries each");
            field_diag_type_e d[2] = { diag[0], diag[1] };
            field_loc_type_e l[2] = { loc[0], loc[1] };
            return p.new_field_plot_window(N, x1, x2, d, l);
        }, py::arg("N"), py::arg("x1"), py::arg("x2"), py::arg("diag"), py::arg("loc"),
           py::return_value_policy::reference)
        .def("delete_window", &GTKPlotter::delete_window, py::arg("window"));
    m.attr("HAVE_GTK") = true;
#else
    m.attr("HAVE_GTK") = false;
#endif
}
