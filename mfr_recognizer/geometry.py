from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, isfinite, pi, sqrt
from typing import Iterable

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepTools import breptools
from OCC.Core.gp import gp_Pnt
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherCurve,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopoDS import topods


LABELS = {
    0: "other",
    1: "hole",
    2: "boss",
    3: "chamfer",
}

SURFACE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset",
    GeomAbs_OtherSurface: "other",
}

CURVE_NAMES = {
    GeomAbs_Line: "line",
    GeomAbs_Circle: "circle",
    GeomAbs_Ellipse: "ellipse",
    GeomAbs_OtherCurve: "other",
}


Vec3 = tuple[float, float, float]


def read_step(path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != 1:
        raise ValueError(f"Could not read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def norm(a: Vec3) -> float:
    return sqrt(max(dot(a, a), 0.0))


def unit(a: Vec3 | None) -> Vec3 | None:
    if a is None:
        return None
    n = norm(a)
    if n <= 1.0e-12:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def abs_dot(a: Vec3 | None, b: Vec3 | None) -> float:
    if a is None or b is None:
        return 0.0
    return abs(dot(a, b))


def angle_degrees(a: Vec3 | None, b: Vec3 | None) -> float | None:
    if a is None or b is None:
        return None
    v = max(-1.0, min(1.0, dot(a, b)))
    return acos(v) * 180.0 / pi


def point_tuple(p) -> Vec3:
    return (float(p.X()), float(p.Y()), float(p.Z()))


def edge_polyline(edge, samples: int = 24) -> list[Vec3]:
    """Sample a single edge into 3D points (linear curves use just their ends)."""
    curve = BRepAdaptor_Curve(edge)
    ctype = curve.GetType()
    u0 = float(curve.FirstParameter())
    u1 = float(curve.LastParameter())
    pts: list[Vec3] = []
    if ctype == GeomAbs_Line:
        p0 = gp_Pnt()
        p1 = gp_Pnt()
        curve.D0(u0, p0)
        curve.D0(u1, p1)
        pts = [point_tuple(p0), point_tuple(p1)]
    else:
        n = max(samples, 2)
        for k in range(n + 1):
            u = u0 + (u1 - u0) * k / n
            p = gp_Pnt()
            curve.D0(u, p)
            pts.append(point_tuple(p))
    return pts


def face_outer_loop_polyline(face) -> list[Vec3]:
    """Polygonize the outer wire of a face into a list of 3D points in order.

    Points are deduplicated so consecutive equal samples do not produce zero-length
    segments. Returns an empty list if the wire cannot be walked.
    """
    from OCC.Core.BRepTools import breptools as _bt  # local import to reuse existing symbol
    wires = enumerate_wires(face)
    if not wires:
        return []
    wire = wires[0]  # outer wire is first per STEP convention
    pts: list[Vec3] = []
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    edges = []
    while explorer.More():
        edges.append(topods.Edge(explorer.Current()))
        explorer.Next()
    if not edges:
        return []
    for edge in edges:
        for p in edge_polyline(edge):
            if not pts or norm(sub(p, pts[-1])) > 1.0e-9:
                pts.append(p)
    if pts and norm(sub(pts[0], pts[-1])) <= 1.0e-9:
        pts.pop()
    return pts


def planar_point_in_polygon(
    point: Vec3, polygon: list[Vec3], origin: Vec3, normal: Vec3
) -> bool:
    """Test whether ``point`` lies inside ``polygon`` (a planar 3D loop), with the
    polygon lying in the plane through ``origin`` with ``normal``.

    Both point and polygon vertices are projected onto a 2D basis of the plane and a
    ray-casting test is applied. A point on the boundary counts as inside.
    """
    if len(polygon) < 3:
        return False
    # Build an in-plane 2D basis from the normal.
    ref = (0.0, 1.0, 0.0) if abs(normal[0]) > 0.9 else (1.0, 0.0, 0.0)
    x_axis = unit(sub(ref, scale(normal, dot(ref, normal))))
    if x_axis is None:
        return False
    y_axis = unit((normal[1] * x_axis[2] - normal[2] * x_axis[1],
                   normal[2] * x_axis[0] - normal[0] * x_axis[2],
                   normal[0] * x_axis[1] - normal[1] * x_axis[0]))
    if y_axis is None:
        return False

    def to2d(p: Vec3) -> tuple[float, float]:
        d = sub(p, origin)
        return (dot(d, x_axis), dot(d, y_axis))

    px, py = to2d(point)
    poly2d = [to2d(v) for v in polygon]
    inside = False
    n = len(poly2d)
    for i in range(n):
        xi, yi = poly2d[i]
        xj, yj = poly2d[(i + 1) % n]
        # Robust half-open ray cast: count an edge if the ray crosses it, treating
        # the lower vertex as inclusive and the upper as exclusive to avoid
        # double-counting at shared vertices.
        if (yi > py) != (yj > py):
            denom = yj - yi
            x_at = xi if abs(denom) < 1.0e-12 else xi + (xj - xi) * (py - yi) / denom
            if x_at > px:
                inside = not inside
    return inside



def dir_tuple(d) -> Vec3:
    return unit((float(d.X()), float(d.Y()), float(d.Z()))) or (0.0, 0.0, 0.0)


def face_area(face) -> float:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return float(props.Mass())


def edge_length(edge) -> float:
    props = GProp_GProps()
    brepgprop.LinearProperties(edge, props)
    return float(props.Mass())


def bbox_diagonal(shape) -> float:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)


def enumerate_faces(shape) -> list:
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(topods.Face(explorer.Current()))
        explorer.Next()
    return faces


def enumerate_edges(shape) -> list:
    edges = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edges.append(topods.Edge(explorer.Current()))
        explorer.Next()
    return edges


def enumerate_wires(face) -> list:
    wires = []
    explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while explorer.More():
        wires.append(topods.Wire(explorer.Current()))
        explorer.Next()
    return wires


def shape_key(shape) -> int:
    return hash(shape)


def same_shape(a, b) -> bool:
    return a.IsSame(b)


def face_center(face) -> Vec3:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return point_tuple(props.CentreOfMass())


def uv_midpoint(face) -> tuple[float, float]:
    u1, u2, v1, v2 = breptools.UVBounds(face)
    vals = (u1, u2, v1, v2)
    if not all(isfinite(float(v)) for v in vals):
        return (0.0, 0.0)
    return ((float(u1) + float(u2)) * 0.5, (float(v1) + float(v2)) * 0.5)


def uv_spans(face) -> tuple[float, float]:
    u1, u2, v1, v2 = breptools.UVBounds(face)
    vals = (u1, u2, v1, v2)
    if not all(isfinite(float(v)) for v in vals):
        return (0.0, 0.0)
    return (abs(float(u2) - float(u1)), abs(float(v2) - float(v1)))


def surface_normal(face, surf: BRepAdaptor_Surface | None = None) -> Vec3 | None:
    surf = surf or BRepAdaptor_Surface(face, True)
    u, v = uv_midpoint(face)
    props = BRepLProp_SLProps(surf, u, v, 1, 1.0e-6)
    if not props.IsNormalDefined():
        return None
    normal = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()
    return dir_tuple(normal)


def surface_point(face, surf: BRepAdaptor_Surface | None = None) -> Vec3:
    surf = surf or BRepAdaptor_Surface(face, True)
    u, v = uv_midpoint(face)
    props = BRepLProp_SLProps(surf, u, v, 1, 1.0e-6)
    return point_tuple(props.Value())


def radial_alignment(face, surf: BRepAdaptor_Surface | None = None) -> float | None:
    """Return outward-normal versus surface radial direction for cylinders/cones.

    Positive values are external/convex side walls, while negative values are
    internal/concave walls such as holes.
    """

    surf = surf or BRepAdaptor_Surface(face, True)
    stype = surf.GetType()
    if stype not in (GeomAbs_Cylinder, GeomAbs_Cone):
        return None

    if stype == GeomAbs_Cylinder:
        axis = surf.Cylinder().Axis()
    else:
        axis = surf.Cone().Axis()
    axis_point = point_tuple(axis.Location())
    axis_dir = dir_tuple(axis.Direction())
    sample = surface_point(face, surf)
    normal = surface_normal(face, surf)
    if normal is None:
        return None

    ap = sub(sample, axis_point)
    projected = add(axis_point, scale(axis_dir, dot(ap, axis_dir)))
    radial = unit(sub(sample, projected))
    if radial is None:
        return None
    return dot(normal, radial)


def surface_axis_direction(face, surf: BRepAdaptor_Surface | None = None) -> Vec3 | None:
    surf = surf or BRepAdaptor_Surface(face, True)
    stype = surf.GetType()
    if stype == GeomAbs_Cylinder:
        axis = surf.Cylinder().Axis()
    elif stype == GeomAbs_Cone:
        axis = surf.Cone().Axis()
    else:
        return None
    return dir_tuple(axis.Direction())


def surface_axis_point(face, surf: BRepAdaptor_Surface | None = None) -> Vec3 | None:
    surf = surf or BRepAdaptor_Surface(face, True)
    stype = surf.GetType()
    if stype == GeomAbs_Cylinder:
        axis = surf.Cylinder().Axis()
    elif stype == GeomAbs_Cone:
        axis = surf.Cone().Axis()
    else:
        return None
    return point_tuple(axis.Location())


def edge_curve_name(edge) -> str:
    curve = BRepAdaptor_Curve(edge)
    return CURVE_NAMES.get(curve.GetType(), "other")


def circular_edge_count(face) -> int:
    return sum(1 for edge in enumerate_edges(face) if edge_curve_name(edge) in {"circle", "ellipse"})


def full_circular_edge_count(face, tolerance: float = 1.0e-3) -> int:
    count = 0
    for edge in enumerate_edges(face):
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() not in (GeomAbs_Circle, GeomAbs_Ellipse):
            continue
        span = abs(float(curve.LastParameter()) - float(curve.FirstParameter()))
        if span >= 2.0 * pi - tolerance:
            count += 1
    return count


def linear_edge_count(face) -> int:
    return sum(1 for edge in enumerate_edges(face) if edge_curve_name(edge) == "line")


def face_map_index(faces: list, face) -> int | None:
    key = shape_key(face)
    for idx, candidate in enumerate(faces):
        if shape_key(candidate) == key or same_shape(candidate, face):
            return idx
    return None


@dataclass
class FaceInfo:
    index: int
    shape: object
    surface_type: int
    surface_name: str
    area: float
    center: Vec3
    normal: Vec3 | None
    axis_dir: Vec3 | None
    axis_point: Vec3 | None
    u_span: float
    v_span: float
    wire_count: int
    inner_wire_count: int
    radial: float | None
    edge_count: int
    line_edges: int
    circle_edges: int
    full_circle_edges: int
    neighbors: set[int] = field(default_factory=set)
    shared_edges: dict[int, list[int]] = field(default_factory=dict)
    inner_loop_neighbors: set[int] = field(default_factory=set)

    @property
    def is_plane(self) -> bool:
        return self.surface_type == GeomAbs_Plane

    @property
    def is_cylinder(self) -> bool:
        return self.surface_type == GeomAbs_Cylinder

    @property
    def is_cone(self) -> bool:
        return self.surface_type == GeomAbs_Cone

    @property
    def is_round_side(self) -> bool:
        return self.surface_type in (GeomAbs_Cylinder, GeomAbs_Cone)

    @property
    def has_inner_loop(self) -> bool:
        return self.inner_wire_count > 0


@dataclass
class BrepGraph:
    shape: object
    faces: list
    infos: list[FaceInfo]
    edge_lengths: list[float]
    model_diagonal: float
    edges: list = field(default_factory=list)

    @classmethod
    def from_step(cls, path: str) -> "BrepGraph":
        return cls.from_shape(read_step(path))

    @classmethod
    def from_shape(cls, shape) -> "BrepGraph":
        faces = enumerate_faces(shape)
        infos = [build_face_info(i, face) for i, face in enumerate(faces)]
        edge_lengths: list[float] = []
        edges: list = []

        edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)
        for edge_idx in range(1, edge_to_faces.Size() + 1):
            edge = topods.Edge(edge_to_faces.FindKey(edge_idx))
            edge_lengths.append(edge_length(edge))
            edges.append(edge)
            ancestor_indices: list[int] = []
            for ancestor in edge_to_faces.FindFromIndex(edge_idx):
                face_index = face_map_index(faces, ancestor)
                if face_index is not None and face_index not in ancestor_indices:
                    ancestor_indices.append(face_index)
            for a in ancestor_indices:
                for b in ancestor_indices:
                    if a == b:
                        continue
                    infos[a].neighbors.add(b)
                    infos[a].shared_edges.setdefault(b, []).append(edge_idx - 1)

        mark_inner_loop_neighbors(faces, infos)
        return cls(shape=shape, faces=faces, infos=infos, edge_lengths=edge_lengths,
                   model_diagonal=bbox_diagonal(shape), edges=edges)

    def shared_full_circle_count(self, a: int, b: int, tolerance: float = 1.0e-3) -> int:
        """Number of full-circle (near 2π) edges shared between faces a and b."""
        shared = self.infos[a].shared_edges.get(b, [])
        count = 0
        for edge_idx in shared:
            if edge_idx < 0 or edge_idx >= len(self.edges):
                continue
            curve = BRepAdaptor_Curve(self.edges[edge_idx])
            if curve.GetType() not in (GeomAbs_Circle, GeomAbs_Ellipse):
                continue
            span = abs(float(curve.LastParameter()) - float(curve.FirstParameter()))
            if span >= 2.0 * pi - tolerance:
                count += 1
        return count

    def shared_full_circle_radii(self, a: int, b: int, tolerance: float = 1.0e-3) -> list[float]:
        """Radii of the full-circle edges shared between faces a and b."""
        shared = self.infos[a].shared_edges.get(b, [])
        radii: list[float] = []
        for edge_idx in shared:
            if edge_idx < 0 or edge_idx >= len(self.edges):
                continue
            curve = BRepAdaptor_Curve(self.edges[edge_idx])
            if curve.GetType() not in (GeomAbs_Circle, GeomAbs_Ellipse):
                continue
            span = abs(float(curve.LastParameter()) - float(curve.FirstParameter()))
            if span < 2.0 * pi - tolerance:
                continue
            try:
                circ = curve.Circle()
            except Exception:
                continue
            radii.append(float(circ.Radius()))
        return radii

    def connected_component(self, seeds: Iterable[int], blocked: set[int] | None = None, limit: int = 256) -> set[int]:
        blocked = blocked or set()
        queue = [s for s in seeds if s not in blocked]
        seen: set[int] = set(queue)
        while queue and len(seen) <= limit:
            current = queue.pop(0)
            for neighbor in self.infos[current].neighbors:
                if neighbor in blocked or neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        return seen


def build_face_info(index: int, face) -> FaceInfo:
    surface = BRepAdaptor_Surface(face, True)
    stype = surface.GetType()
    wires = enumerate_wires(face)
    u_span, v_span = uv_spans(face)
    return FaceInfo(
        index=index,
        shape=face,
        surface_type=stype,
        surface_name=SURFACE_NAMES.get(stype, "other"),
        area=face_area(face),
        center=face_center(face),
        normal=surface_normal(face, surface),
        axis_dir=surface_axis_direction(face, surface),
        axis_point=surface_axis_point(face, surface),
        u_span=u_span,
        v_span=v_span,
        wire_count=len(wires),
        inner_wire_count=max(0, len(wires) - 1),
        radial=radial_alignment(face, surface),
        edge_count=len(enumerate_edges(face)),
        line_edges=linear_edge_count(face),
        circle_edges=circular_edge_count(face),
        full_circle_edges=full_circular_edge_count(face),
    )


def mark_inner_loop_neighbors(faces: list, infos: list[FaceInfo]) -> None:
    face_by_key = {shape_key(face): idx for idx, face in enumerate(faces)}
    for idx, face in enumerate(faces):
        wires = enumerate_wires(face)
        if len(wires) <= 1:
            continue
        outer = breptools.OuterWire(face)
        outer_key = shape_key(outer)
        for wire in wires:
            if shape_key(wire) == outer_key or same_shape(wire, outer):
                continue
            for edge in enumerate_edges(wire):
                for neighbor in infos[idx].neighbors:
                    shared = infos[idx].shared_edges.get(neighbor, [])
                    if any(edge_is_in_face_edge_list(edge, faces, neighbor, edge_idx, infos) for edge_idx in shared):
                        infos[idx].inner_loop_neighbors.add(neighbor)
                        infos[neighbor].inner_loop_neighbors.add(idx)


def edge_is_in_face_edge_list(edge, faces: list, face_index: int, edge_idx: int, infos: list[FaceInfo]) -> bool:
    # The caller already knows the two faces share this edge index in the global
    # edge map. A direct shape comparison against the neighbor face edges keeps
    # this helper independent from the map object lifetime.
    for candidate in enumerate_edges(faces[face_index]):
        if shape_key(candidate) == shape_key(edge) or same_shape(candidate, edge):
            return True
    return False
