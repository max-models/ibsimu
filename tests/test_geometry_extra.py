"""Triangulated surfaces, STL files, DXF contents and the extra Geometry queries."""

import math

import pytest

import ibsimu

CUBE_VERTICES = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]
# Outward oriented triangles of the unit cube.
CUBE_TRIANGLES = [
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
]


def cube_stl():
    return ibsimu.STLFile(
        [ibsimu.Vec3D(*v) for v in CUBE_VERTICES],
        [ibsimu.VTriangle(*t) for t in CUBE_TRIANGLES],
    )


def test_stlfile_from_data_and_file(tmp_path):
    stl = cube_stl()
    assert stl.inside(ibsimu.Vec3D(0.5, 0.5, 0.5))
    assert not stl.inside(ibsimu.Vec3D(1.5, 0.5, 0.5))
    lo, hi = stl.get_bbox()
    assert tuple(lo) == (0, 0, 0) and tuple(hi) == (1, 1, 1)
    fn = str(tmp_path / "cube.stl")
    stl.save(fn)
    again = ibsimu.STLFile(fn)
    assert again.inside(ibsimu.Vec3D(0.2, 0.7, 0.4))


def test_stlfile_bad_index():
    with pytest.raises(IndexError):
        ibsimu.STLFile([ibsimu.Vec3D(0, 0, 0)], [ibsimu.VTriangle(0, 1, 2)])


def test_stlsolid_add_stl_file():
    solid = ibsimu.STLSolid()
    solid.add_stl_file(cube_stl())
    assert solid.inside(ibsimu.Vec3D(0.5, 0.5, 0.5))
    assert solid.get_stl_file(0).inside(ibsimu.Vec3D(0.5, 0.5, 0.5))
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(5, 5, 5), ibsimu.Vec3D(-0.5, -0.5, -0.5), 0.5
    )
    geom.set_solid(7, solid)
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 1.0))
    geom.build_mesh()
    assert geom.inside(ibsimu.Vec3D(0.5, 0.5, 0.5)) == 7


def test_vtriangle_surface_solid():
    s = ibsimu.VTriangleSurfaceSolid()
    for t in CUBE_TRIANGLES:
        s.add_triangle(*(ibsimu.Vec3D(*CUBE_VERTICES[i]) for i in t))
    assert s.vertexc() == 8 and s.trianglec() == 12
    s.check_data()
    s.prepare_for_inside()
    assert s.inside(ibsimu.Vec3D(0.5, 0.5, 0.5))
    assert list(s.triangle(0)) == [0, 1, 2]


def test_geometry_near_solid_queries():
    geom = ibsimu.Geometry(
        ibsimu.MODE_2D, ibsimu.Int3D(21, 11, 1), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    geom.set_solid(7, ibsimu.FuncSolid(lambda x, y, z: x > 10.4e-3))
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 1.0))
    geom.build_mesh()
    # Node 10 is a vacuum node 0.4 h from the solid in the +x direction.
    rec = geom.nearsolid_data(10, 5, 0)
    assert rec["dist"][1] == pytest.approx(0.4, abs=1 / 255)
    assert rec["dist"][0] is None
    assert abs(geom.solid_dist(10, 5, 0, 1) - 0.4 * 255) <= 1
    with pytest.raises(ValueError):
        geom.nearsolid_data(2, 5, 0)
    assert geom.mesh_check(5, 5) == geom.mesh3(5, 5, 0)
    assert geom.mesh_check(-1, 5) == geom.mesh_check2(-1, 5)


def test_geometry_surface_inside_3d():
    geom = ibsimu.Geometry(
        ibsimu.MODE_3D, ibsimu.Int3D(11, 11, 11), ibsimu.Vec3D(0, 0, 0), 1e-3
    )
    geom.set_solid(
        7,
        ibsimu.FuncSolid(
            lambda x, y, z: (x - 5e-3) ** 2 + (y - 5e-3) ** 2 + (z - 5e-3) ** 2 < 9e-6
        ),
    )
    geom.set_boundary(7, ibsimu.Bound(ibsimu.BOUND_DIRICHLET, 1.0))
    geom.build_mesh()
    geom.build_surface()
    assert geom.surface_inside(ibsimu.Vec3D(5e-3, 5e-3, 5e-3)) == 7
    assert geom.surface_inside(ibsimu.Vec3D(1e-3, 1e-3, 1e-3)) == 0


@pytest.fixture()
def dxf_file(tmp_path):
    """Square on layer 'electrode' and a circle on layer 'other'."""
    groups = []

    def g(code, value):
        groups.extend([str(code), str(value)])

    g(0, "SECTION")
    g(2, "ENTITIES")
    for x1, y1, x2, y2 in [(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)]:
        for code, value in [
            (0, "LINE"),
            (8, "electrode"),
            (10, x1),
            (20, y1),
            (30, 0),
            (11, x2),
            (21, y2),
            (31, 0),
        ]:
            g(code, value)
    for code, value in [
        (0, "CIRCLE"),
        (8, "other"),
        (10, 5),
        (20, 5),
        (30, 0),
        (40, 2),
    ]:
        g(code, value)
    g(0, "ENDSEC")
    g(0, "EOF")
    fn = tmp_path / "t.dxf"
    fn.write_text("\n".join(groups) + "\n")
    return str(fn)


def test_dxf_entities(dxf_file):
    dxf = ibsimu.MyDXFFile(dxf_file)
    ent = dxf.get_entities()
    assert len(ent) == 5
    assert isinstance(ent[0], ibsimu.MyDXFLine)
    circle = ent[4]
    assert isinstance(circle, ibsimu.MyDXFCircle)
    assert circle.radius() == 2.0 and tuple(circle.center()) == (5, 5, 0)
    assert ent[0].get_layer() == "electrode"
    assert ent[0].get_type() == ibsimu.ENTITY_LINE

    sel = ent.selection_layer("electrode")
    assert sel.tolist() == [0, 1, 2, 3]
    lo, hi = ent.get_bbox(sel, dxf)
    assert tuple(lo)[:2] == (0, 0) and tuple(hi)[:2] == (1, 1)
    loop = ent.selection_path_loop(sel)
    assert len(loop) == 4
    assert ent.inside_loop(loop, 0.5, 0.5)
    assert not ent.inside_loop(loop, 3.0, 0.5)
    assert len(ent.selection_type(ibsimu.ENTITY_CIRCLE)) == 1

    ent.rotate_z(sel, math.pi / 2)
    assert ent[0].end()[1] == pytest.approx(1.0)
    ent.translate(None, dxf, ibsimu.Vec3D(1, 0, 0))
    assert circle.center()[0] == pytest.approx(6.0)

    # Rotated by 90 degrees and moved by +1 in x, the square is back at 0..1.
    solid = ibsimu.DXFSolid(dxf, "electrode")
    assert solid.inside(ibsimu.Vec3D(0.5, 0.5, 0))
    assert not solid.inside(ibsimu.Vec3D(-0.5, 0.5, 0))
