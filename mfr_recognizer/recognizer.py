from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from statistics import median

from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder

from geometry import (
    BrepGraph,
    FaceInfo,
    abs_dot,
    angle_degrees,
    dot,
    face_outer_loop_polyline,
    norm,
    planar_point_in_polygon,
    scale,
    sub,
    unit,
)


HOLE = 1
BOSS = 2
CHAMFER = 3


@dataclass
class FeatureInstance:
    label: int
    kind: str
    faces: set[int]
    instance_id: int = 0
    hint_faces: set[int] = field(default_factory=set)
    reason: str = ""


@dataclass
class RecognitionResult:
    labels: list[int]
    instance_ids: list[int]
    features: list[FeatureInstance]
    graph: BrepGraph

    def one_based_faces(self, feature: FeatureInstance) -> list[int]:
        return [idx + 1 for idx in sorted(feature.faces)]

    def segment_map(self, *, face_index_base: int = 0) -> dict[int, int]:
        return {idx + face_index_base: label for idx, label in enumerate(self.labels)}

    def instance_adjacency_matrix(self) -> list[list[int]]:
        matrix = [[0] * len(self.labels) for _ in self.labels]
        for feature in self.features:
            if feature.label == 0:
                continue
            for row in feature.faces:
                for column in feature.faces:
                    matrix[row][column] = 1
        return matrix

    def full_payload(self, sample_id: str, *, face_index_base: int = 0) -> list:
        return [[sample_id, {"seg": self.segment_map(face_index_base=face_index_base), "inst": self.instance_adjacency_matrix()}]]


class HintBasedRecognizer:
    """Practical subset of Li et al.'s hint-based B-Rep recognition algorithm.

    The paper describes generic hints in FEG/VEG terms. This implementation
    keeps that structure and specializes it for three useful manufacturing
    classes:
    - internal-loop components with inward round side walls -> holes;
    - internal-loop/face-partition components with outward side walls -> bosses;
    - edge-elimination transitional faces -> chamfers.
    """

    def __init__(
        self,
        *,
        radial_threshold: float = 0.2,
        axis_alignment_threshold: float = 0.7,
        axis_distance_tolerance_ratio: float = 1.0e-5,
        hole_angular_coverage_tolerance: float = 0.05,
        chamfer_min_angle: float = 18.0,
        chamfer_max_angle: float = 72.0,
        chamfer_max_support_area_ratio: float = 0.35,
    ) -> None:
        self.radial_threshold = radial_threshold
        self.axis_alignment_threshold = axis_alignment_threshold
        self.axis_distance_tolerance_ratio = axis_distance_tolerance_ratio
        self.hole_angular_coverage_tolerance = hole_angular_coverage_tolerance
        self.chamfer_min_angle = chamfer_min_angle
        self.chamfer_max_angle = chamfer_max_angle
        self.chamfer_max_support_area_ratio = chamfer_max_support_area_ratio

    def recognize_step(self, path: str) -> RecognitionResult:
        return self.recognize_graph(BrepGraph.from_step(path))

    def recognize_graph(self, graph: BrepGraph) -> RecognitionResult:
        labels = [0] * len(graph.infos)
        features: list[FeatureInstance] = []

        for feature in self._recognize_holes(graph):
            self._apply(labels, features, feature)

        for feature in self._recognize_chamfers(graph, labels):
            self._apply(labels, features, feature)

        for feature in self._recognize_structural_bosses(graph, labels):
            self._apply(labels, features, feature)

        features = self._group_chamfer_instances(graph, labels, features)
        instance_ids = self._build_instance_ids(len(graph.infos), features)
        return RecognitionResult(labels=labels, instance_ids=instance_ids, features=features, graph=graph)

    def _apply(self, labels: list[int], features: list[FeatureInstance], feature: FeatureInstance) -> None:
        owned_faces = {idx for idx in feature.faces if labels[idx] == 0}
        if not owned_faces:
            return
        feature.faces = owned_faces
        for idx in owned_faces:
            if labels[idx] == 0:
                labels[idx] = feature.label
        features.append(feature)

    def _build_instance_ids(self, face_count: int, features: list[FeatureInstance]) -> list[int]:
        instance_ids = [0] * face_count
        for instance_id, feature in enumerate(features, start=1):
            feature.instance_id = instance_id
            for face_idx in feature.faces:
                instance_ids[face_idx] = instance_id
        return instance_ids

    # ------------------------------------------------------------------
    # Hole recognition (definition-driven, single pass)
    # ------------------------------------------------------------------
    #
    # Per definition.md a Hole is "the removal of a cylindrical volume". The
    # signature of that removal is a *complete* 2π cylindrical side wall — one
    # unbroken tube around the axis. The tube may be a single face with u_span
    # = 2π, or several coaxial-same-radius faces that meet along axial seams
    # (a half-cylinder pair, or an arc-and-arc quartet cut by planes). What is
    # NOT a hole is a wall broken so that no 2π loop of side wall remains — a
    # slot, a channel, a fillet.
    #
    # This single pass works wall-first:
    #   1. Enumerate every inward cylindrical face and group them by (axis,
    #      radius) via topological BFS along shared edges. Coaxial faces that
    #      are not linked by shared edges belong to separate groups (this is
    #      the user's "topologically disconnected → not a hole" rule).
    #   2. A group is a hole candidate iff the sum of its faces' u_spans ≈ 2π
    #      (the group closes a full circumference).
    #   3. For each qualifying group, classify its two axial ends:
    #        - If a plane face shares an edge with the group AND that shared
    #          edge lies on the plane's *outer* wire → the plane sits inside
    #          the mouth circle → blind bottom (may itself carry inner holes).
    #        - Otherwise the mouth is open (through-hole, or the plane is a
    #          carrier that the wall passes through as an inner wire).
    #   4. Emit a FeatureInstance containing the walls plus any blind bottoms.
    #      Transition faces (cones/tori/chamfers) keep their own label.
    #   5. Merge stepped holes: two coaxial hole features that share a shelf
    #      plane (a plane that is one feature's blind bottom AND the other
    #      feature's carrier) collapse into one feature with the shelf inside.

    def _recognize_holes(self, graph: BrepGraph) -> list[FeatureInstance]:
        groups = self._enumerate_wall_groups(graph)
        features: list[FeatureInstance] = []
        for group in groups:
            feature = self._build_hole_from_wall_group(graph, group)
            if feature is not None:
                features.append(feature)
        return self._merge_stepped_holes(graph, features)

    def _enumerate_wall_groups(self, graph: BrepGraph) -> list[set[int]]:
        """Group all inward cylindrical faces into connected coaxial+radius sets.

        Two inward cylinders belong to the same group iff they share an edge
        and have matching axis + radius. Purely geometric coaxiality without a
        shared edge does NOT merge groups.
        """
        candidates = [
            idx
            for idx, info in enumerate(graph.infos)
            if info.is_cylinder
            and info.radial is not None
            and info.radial < -self.radial_threshold
            and info.axis_dir is not None
            and info.axis_point is not None
            and info.radius is not None
        ]
        assigned: dict[int, int] = {}
        groups: list[set[int]] = []
        for seed in candidates:
            if seed in assigned:
                continue
            group_idx = len(groups)
            group: set[int] = {seed}
            assigned[seed] = group_idx
            queue = [seed]
            seed_info = graph.infos[seed]
            while queue:
                current = queue.pop(0)
                for neighbor_idx in graph.infos[current].neighbors:
                    if neighbor_idx in assigned:
                        continue
                    if neighbor_idx not in candidates:
                        continue
                    if not self._coaxial_same_radius(graph, seed_info, graph.infos[neighbor_idx]):
                        continue
                    assigned[neighbor_idx] = group_idx
                    group.add(neighbor_idx)
                    queue.append(neighbor_idx)
            groups.append(group)
        return groups

    def _coaxial_same_radius(
        self, graph: BrepGraph, a: FaceInfo, b: FaceInfo
    ) -> bool:
        if a.radius is None or b.radius is None:
            return False
        tol_radius = max(graph.model_diagonal * 1.0e-5, 1.0e-6)
        if abs(a.radius - b.radius) > tol_radius:
            return False
        if a.axis_dir is None or b.axis_dir is None:
            return False
        if abs_dot(a.axis_dir, b.axis_dir) < 1.0 - self.axis_distance_tolerance_ratio:
            return False
        if a.axis_point is None or b.axis_point is None:
            return False
        delta = sub(b.axis_point, a.axis_point)
        projection = dot(delta, a.axis_dir)
        perpendicular = (
            delta[0] - projection * a.axis_dir[0],
            delta[1] - projection * a.axis_dir[1],
            delta[2] - projection * a.axis_dir[2],
        )
        tol_dist = max(graph.model_diagonal * self.axis_distance_tolerance_ratio, 1.0e-7)
        return norm(perpendicular) <= tol_dist

    def _build_hole_from_wall_group(
        self, graph: BrepGraph, walls: set[int]
    ) -> FeatureInstance | None:
        if not walls:
            return None
        # Closure: the group must cover a complete 2π circumference.
        u_total = sum(graph.infos[idx].u_span or 0.0 for idx in walls)
        if u_total < 2.0 * pi - self.hole_angular_coverage_tolerance:
            return None

        # Hint anchor (per Li et al.): a hole is anchored to a mouth carrier —
        # a face that has one of the walls on its inner wire. A single
        # cone/torus chamfer between wall and carrier is allowed. Without a
        # carrier hint, an inward cylindrical fragment (e.g. an oblique cut
        # through a solid that happens to close 2π at one end) is not a hole.
        # A blind bottom is admitted only when a carrier hint exists: any
        # plane adjoining an unanchored wall is an oblique cut, not a proper
        # bottom.
        has_carrier = self._wall_group_has_carrier(graph, walls)
        if not has_carrier:
            bottoms: set[int] = set()
        else:
            bottoms = self._find_blind_bottoms(graph, walls)

        faces: set[int] = set(walls) | bottoms
        return FeatureInstance(
            label=HOLE,
            kind="hole",
            faces=faces,
            hint_faces=set(walls),
            reason="inward cylindrical side wall closes a full 2π circumference",
        )

    def _wall_group_has_carrier(self, graph: BrepGraph, walls: set[int]) -> bool:
        """True when at least one wall face has an inner-loop carrier reachable
        either directly or through a single cone/torus transition face.

        A carrier is a face that has one of the walls in its inner_loop_neighbors
        (i.e. the wall sits inside the carrier's inner wire, the classic hole
        hint from Li et al.). When a chamfer/fillet hides the mouth from the
        carrier, the transition face itself becomes the wall's neighbor and the
        carrier is one hop further out.
        """
        for wall_idx in walls:
            for neighbor_idx in graph.infos[wall_idx].neighbors:
                if neighbor_idx in walls:
                    continue
                neighbor = graph.infos[neighbor_idx]
                # Direct carrier: the wall is on the neighbor's inner wire.
                if wall_idx in neighbor.inner_loop_neighbors:
                    return True
                # Transition-hopped carrier: the wall's neighbor is a cone or
                # torus (a chamfer/fillet), and that transition face is itself
                # on some carrier's inner wire.
                if neighbor.surface_name in ("cone", "torus"):
                    for hop_idx in neighbor.neighbors:
                        if hop_idx in walls or hop_idx == wall_idx:
                            continue
                        if neighbor_idx in graph.infos[hop_idx].inner_loop_neighbors:
                            return True
        return False

    def _find_blind_bottoms(self, graph: BrepGraph, walls: set[int]) -> set[int]:
        """A plane P is a blind bottom of the wall group when:
        - the shared edges between P and the wall group are circle arcs on the
          wall's axis whose parameter spans sum to ≈ 2π (a complete mouth), AND
        - those shared edges lie on P's outer wire (P sits inside the mouth
          circle, not on the mouth's outside).

        A plane whose only shared edges lie on its inner wire is the mouth's
        carrier (a face with a hole in it), not the bottom.
        """
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Circle as _GeomAbs_Circle

        # Reference axis of the wall group.
        ref_info = None
        for idx in walls:
            info = graph.infos[idx]
            if info.axis_dir and info.axis_point:
                ref_info = info
                break
        if ref_info is None:
            return set()
        ref_axis = ref_info.axis_dir
        ref_radius = ref_info.radius

        candidates: set[int] = set()
        for wall_idx in walls:
            for neighbor_idx in graph.infos[wall_idx].neighbors:
                if neighbor_idx in walls:
                    continue
                if not graph.infos[neighbor_idx].is_plane:
                    continue
                candidates.add(neighbor_idx)

        tol_radius = max(graph.model_diagonal * 1.0e-5, 1.0e-6)
        bottoms: set[int] = set()
        for plane_idx in candidates:
            plane_info = graph.infos[plane_idx]
            # Sum the circle-arc spans on wall's axis, per mouth position (grouped by z along axis).
            mouth_spans: dict[float, tuple[float, bool]] = {}
            # key: rounded axial-projection z, value: (accumulated span, on_outer_wire)
            for wall_idx in walls:
                shared = plane_info.shared_edges.get(wall_idx, [])
                if not shared:
                    continue
                is_inner = wall_idx in plane_info.inner_loop_neighbors
                for edge_idx in shared:
                    if edge_idx < 0 or edge_idx >= len(graph.edges):
                        continue
                    curve = BRepAdaptor_Curve(graph.edges[edge_idx])
                    if curve.GetType() != _GeomAbs_Circle:
                        continue
                    try:
                        circ = curve.Circle()
                    except Exception:
                        continue
                    if abs(float(circ.Radius()) - (ref_radius or 0.0)) > tol_radius:
                        continue
                    span = abs(float(curve.LastParameter()) - float(curve.FirstParameter()))
                    from geometry import point_tuple
                    center = point_tuple(circ.Location())
                    z = dot(center, ref_axis)
                    z_key = round(z, 6)
                    prev = mouth_spans.get(z_key, (0.0, False))
                    mouth_spans[z_key] = (prev[0] + span, prev[1] or (not is_inner))
            # Any mouth (z-key) with total span ≥ 2π-tol AND on outer wire → blind bottom.
            for z_key, (span, on_outer) in mouth_spans.items():
                if span >= 2.0 * pi - self.hole_angular_coverage_tolerance and on_outer:
                    bottoms.add(plane_idx)
                    break
        return bottoms

    def _merge_stepped_holes(
        self, graph: BrepGraph, features: list[FeatureInstance]
    ) -> list[FeatureInstance]:
        """Merge coaxial hole features that share a stepped shelf.

        A shelf is a plane that is one feature's blind bottom and the other
        feature's carrier (i.e. plane's outer wire touches the wider wall and
        inner wire touches the narrower wall).
        """
        if len(features) < 2:
            return features
        holes = [f for f in features if f.label == HOLE]
        others = [f for f in features if f.label != HOLE]
        if len(holes) < 2:
            return features

        # Compute each hole's wall axis representative.
        def hole_axis(feature: FeatureInstance):
            for idx in feature.hint_faces:
                info = graph.infos[idx]
                if info.axis_dir and info.axis_point:
                    return info.axis_dir, info.axis_point
            return None

        axes = {id(f): hole_axis(f) for f in holes}

        # Union-find over hole features by shared shelf plane.
        parent = list(range(len(holes)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        # Find shelf candidates: planes that are the blind bottom of one hole
        # and share an inner-loop edge with another hole's walls.
        bottom_owner: dict[int, int] = {}
        for i, hole in enumerate(holes):
            walls = hole.hint_faces
            faces_in = hole.faces
            for face_idx in faces_in - walls:
                info = graph.infos[face_idx]
                if info.is_plane:
                    bottom_owner[face_idx] = i

        for shelf_idx, owner_i in bottom_owner.items():
            shelf_info = graph.infos[shelf_idx]
            # Does this shelf's inner wire touch another hole's walls?
            axis_i = axes.get(id(holes[owner_i]))
            for other_j, other in enumerate(holes):
                if other_j == owner_i:
                    continue
                axis_j = axes.get(id(other))
                if axis_i is None or axis_j is None:
                    continue
                # Same axis line?
                if abs_dot(axis_i[0], axis_j[0]) < 1.0 - self.axis_distance_tolerance_ratio:
                    continue
                delta = sub(axis_j[1], axis_i[1])
                proj = dot(delta, axis_i[0])
                perp = (
                    delta[0] - proj * axis_i[0][0],
                    delta[1] - proj * axis_i[0][1],
                    delta[2] - proj * axis_i[0][2],
                )
                tol_dist = max(graph.model_diagonal * self.axis_distance_tolerance_ratio, 1.0e-7)
                if norm(perp) > tol_dist:
                    continue
                # Does the shelf's inner wire share an edge with any of the
                # other hole's walls?
                inner_neighbors = shelf_info.inner_loop_neighbors
                if inner_neighbors & other.hint_faces:
                    union(owner_i, other_j)

        # Rebuild feature list with unions applied.
        merged: dict[int, FeatureInstance] = {}
        for i, hole in enumerate(holes):
            root = find(i)
            if root in merged:
                merged[root].faces |= hole.faces
                merged[root].hint_faces |= hole.hint_faces
            else:
                merged[root] = FeatureInstance(
                    label=HOLE,
                    kind="hole",
                    faces=set(hole.faces),
                    hint_faces=set(hole.hint_faces),
                    reason="stepped hole" if any(find(j) == root and j != i for j in range(len(holes))) else hole.reason,
                )
        return list(merged.values()) + others

    def _group_chamfer_instances(
        self, graph: BrepGraph, labels: list[int], features: list[FeatureInstance]
    ) -> list[FeatureInstance]:
        chamfer_indices = [idx for idx, feature in enumerate(features) if feature.label == CHAMFER]
        if len(chamfer_indices) < 2:
            return features

        face_to_solid_feature = {
            face_idx: feature_idx
            for feature_idx, feature in enumerate(features)
            if feature.label in {HOLE, BOSS}
            for face_idx in feature.faces
        }
        face_to_chamfer_feature = {
            face_idx: feature_idx
            for feature_idx in chamfer_indices
            for face_idx in features[feature_idx].faces
        }
        anchor_indices = {
            feature_idx: self._chamfer_anchor_feature_indices(graph, features[feature_idx], labels, face_to_solid_feature)
            for feature_idx in chamfer_indices
        }

        chamfer_neighbors: dict[int, set[int]] = {feature_idx: set() for feature_idx in chamfer_indices}
        for feature_idx in chamfer_indices:
            for face_idx in features[feature_idx].faces:
                for neighbor_idx in graph.infos[face_idx].neighbors:
                    neighbor_feature_idx = face_to_chamfer_feature.get(neighbor_idx)
                    if neighbor_feature_idx is not None and neighbor_feature_idx != feature_idx:
                        chamfer_neighbors[feature_idx].add(neighbor_feature_idx)

        group_first: dict[int, int] = {}
        merged_features: dict[int, FeatureInstance] = {}
        visited: set[int] = set()

        for feature_idx in chamfer_indices:
            if feature_idx in visited:
                continue
            group = {feature_idx}
            queue = [feature_idx]
            visited.add(feature_idx)
            while queue:
                current_idx = queue.pop(0)
                for neighbor_idx in sorted(chamfer_neighbors[current_idx]):
                    if neighbor_idx in visited:
                        continue
                    if not anchor_indices[current_idx] or not anchor_indices[neighbor_idx]:
                        continue
                    if anchor_indices[current_idx].isdisjoint(anchor_indices[neighbor_idx]):
                        continue
                    visited.add(neighbor_idx)
                    group.add(neighbor_idx)
                    queue.append(neighbor_idx)

            first_idx = min(group)
            for member_idx in group:
                group_first[member_idx] = first_idx
            if len(group) > 1:
                faces = set().union(*(features[member_idx].faces for member_idx in group))
                hint_faces = set().union(*(features[member_idx].hint_faces for member_idx in group))
                anchor_kinds = sorted(
                    {features[anchor_idx].kind for member_idx in group for anchor_idx in anchor_indices[member_idx]}
                )
                anchor_text = "/".join(anchor_kinds) if anchor_kinds else "feature"
                merged_features[first_idx] = FeatureInstance(
                    label=CHAMFER,
                    kind="chamfer",
                    faces=faces,
                    hint_faces=hint_faces,
                    reason=f"chamfer ring around {anchor_text}",
                )

        grouped: list[FeatureInstance] = []
        for feature_idx, feature in enumerate(features):
            if feature.label != CHAMFER:
                grouped.append(feature)
                continue
            first_idx = group_first.get(feature_idx, feature_idx)
            if first_idx == feature_idx:
                grouped.append(merged_features.get(feature_idx, feature))
        return grouped

    def _chamfer_anchor_feature_indices(
        self,
        graph: BrepGraph,
        feature: FeatureInstance,
        labels: list[int],
        face_to_solid_feature: dict[int, int],
    ) -> set[int]:
        anchors: set[int] = set()
        for face_idx in feature.faces:
            for neighbor_idx in graph.infos[face_idx].neighbors:
                if labels[neighbor_idx] not in {HOLE, BOSS}:
                    continue
                feature_idx = face_to_solid_feature.get(neighbor_idx)
                if feature_idx is not None:
                    anchors.add(feature_idx)
        return anchors


    def _side_protrudes_from_carrier(
        self, graph: BrepGraph, side: FaceInfo, carrier: FaceInfo, axis: Vec3
    ) -> bool:
        offset = dot(sub(side.center, carrier.center), axis)
        tolerance = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        return offset > tolerance


    def _faces_are_coaxial(self, graph: BrepGraph, a: FaceInfo, b: FaceInfo) -> bool:
        if a.axis_dir is None or b.axis_dir is None or a.axis_point is None or b.axis_point is None:
            return False
        if abs_dot(a.axis_dir, b.axis_dir) < 1.0 - self.axis_distance_tolerance_ratio:
            return False
        distance = self._axis_distance(a, b)
        tolerance = max(graph.model_diagonal * self.axis_distance_tolerance_ratio, 1.0e-7)
        return distance <= tolerance

    def _same_axis_dir(self, graph: BrepGraph, a: FaceInfo, b: FaceInfo) -> bool:
        """True when two cylinder/cone walls share their generatrix direction."""
        if a.axis_dir is None or b.axis_dir is None:
            return False
        return abs_dot(a.axis_dir, b.axis_dir) >= 1.0 - self.axis_distance_tolerance_ratio

    def _axis_distance(self, a: FaceInfo, b: FaceInfo) -> float:
        # For parallel lines, distance is the length of the component of point
        # delta perpendicular to the shared axis.
        delta = sub(b.axis_point, a.axis_point)
        projection = dot(delta, a.axis_dir)
        perpendicular = (
            delta[0] - projection * a.axis_dir[0],
            delta[1] - projection * a.axis_dir[1],
            delta[2] - projection * a.axis_dir[2],
        )
        return norm(perpendicular)


    def _recognize_structural_bosses(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        """Discover bosses from structure: closed side-wall ring + covering top + bottom.

        A boss is a closed shape protruding from a base surface. Structurally that is:
        (1) a ring of side-wall faces that closes around an opening (the ring must not
        leak onto unrelated exterior faces), (2) a top face that caps the ring — either
        directly adjacent to the side walls, or reachable through transition faces
        (chamfer/fillet blends) so that transition+top together cover the ring, and
        (3) a bottom, i.e. the base surface the ring protrudes from. The ring + top +
        transition faces form one boss instance; the bottom/carrier is excluded.

        This pass only claims still-unlabelled faces, so bosses already recognised by
        the typed rules are left intact and cannot regress.
        """
        features: list[FeatureInstance] = []
        consumed: set[int] = set()
        for seed_idx in range(len(graph.infos)):
            if labels[seed_idx] != 0 or seed_idx in consumed:
                continue
            seed = graph.infos[seed_idx]
            if not self._is_boss_side_wall_seed(graph, seed):
                continue
            # Case E — coaxial flange/collar boss (e.g. a screw head on its
            # shaft): a ring of outward cylinders capped by an axis-aligned
            # plane whose inner hole lets a smaller coaxial shaft through. The
            # head protrudes radially from the shaft, so the cap sits at the
            # wall's axial end rather than beyond it and the axial-protrusion
            # model below cannot fit it. Detect it structurally first.
            flange = self._try_coaxial_flange_boss(graph, labels, seed, consumed)
            if flange is not None:
                consumed |= flange.faces
                features.append(flange)
                continue
            carrier_axis = self._structural_boss_carrier(graph, seed)
            if carrier_axis is None:
                continue
            carrier, axis = carrier_axis
            protrusion_sign = self._boss_protrusion_sign(graph, seed, carrier, axis)
            # A boss protrudes outward from the carrier (positive offset along the
            # protrusion axis, since carrier normals point outward from the material).
            # A recess — e.g. a cross-slot milled into a screw head — is structurally
            # identical (a ring of outward cylinders held by the carrier's inner loop)
            # but sits on the negative side. Rejecting non-positive protrusion keeps
            # recesses out without affecting real bosses (all positive-side).
            if protrusion_sign <= 0:
                continue
            ring = self._grow_boss_ring(graph, labels, seed, carrier, axis, consumed, protrusion_sign)
            if len(ring) < 1:
                continue
            if not self._boss_ring_is_closed(graph, ring, carrier, axis, protrusion_sign):
                continue
            top, transitions = self._boss_ring_covering_top(graph, ring, carrier, axis, labels, consumed, protrusion_sign)
            if top is None:
                continue
            # Some bosses enclose an inward recess wall (e.g. the neck of a spool
            # boss): an inward cylindrical wall fully surrounded by the boss faces
            # and carrier. Collect them so the cap check does not mistake them for
            # a spill, and so they are labelled as part of the boss instance.
            enclosed = self._collect_enclosed_recess_walls(
                graph, set(ring) | {top} | set(transitions), carrier, labels
            )
            cap_faces = {top} | set(transitions) | enclosed
            # The top (+ any transition faces) must cover the side-wall ring without
            # extending past it: the cap assembly's boundary may at most coincide with
            # the ring's top edge. If the top touches any face outside the boss
            # structure (ring/carrier/hole-wall/assembly), it spills past the ring and
            # this is not a boss.
            if not self._boss_cap_within_ring(graph, top, cap_faces, ring, carrier):
                continue
            # Proportion gate: perimeter of the carrier (base face) / π must exceed
            # the boss height (carrier → top). The check compares the whole base's
            # size against the boss's height — a boss on a large base plate passes
            # even when the boss itself is tall, while pegs/pins on a small base
            # (whose base perimeter is close to the peg's own perimeter) still fail.
            if not self._boss_perimeter_exceeds_height(graph, ring, top, transitions, carrier, axis):
                continue
            # Per the Boss definition, the radius-shaped blends (cone/torus
            # transitions at the boss root or cap rim) are part of the boss. The
            # top-finding BFS collects them in `transitions`; they are unlabelled
            # cone/torus faces (planar chamfers, a separate Transition_feature, were
            # already labelled by the chamfer pass and never enter transitions), so
            # they merge into the instance alongside the ring, top, and any enclosed
            # recess walls.
            faces = set(ring) | {top} | enclosed | transitions
            consumed |= faces
            features.append(
                FeatureInstance(
                    label=BOSS,
                    kind="boss",
                    faces=faces,
                    hint_faces={carrier.index},
                    reason="structural boss: closed side-wall ring with covering top",
                )
            )
        return features

    def _try_coaxial_flange_boss(
        self, graph: BrepGraph, labels: list[int], seed: FaceInfo, consumed: set[int]
    ) -> FeatureInstance | None:
        """Coaxial flange/collar boss: a ring of outward cylinders capped at one
        axial end by a plane whose inner hole passes a smaller coaxial shaft
        (e.g. a screw head on its shaft). The head protrudes radially from the
        shaft, so the cap sits at the wall's axial end rather than beyond it —
        the axial-protrusion boss pass cannot recognise it. Detect it directly:
        a full-circumference ring of coaxial outward cylinders + the cap, with
        the smaller coaxial outward shaft as the carrier (excluded like any base).
        The shaft gate (outward, smaller radius, coaxial) distinguishes this from
        a boss with a through-hole, whose cap hole leads to an inward hole wall.
        """
        if not seed.is_cylinder or seed.axis_dir is None or seed.radius is None:
            return None
        if seed.radial is None or seed.radial <= self.radial_threshold:
            return None
        axis = seed.axis_dir
        radius_tol = max(graph.model_diagonal * 1.0e-5, 1.0e-6)
        # 1. Cap: an axis-aligned planar neighbour of the seed with an inner loop
        #    (the shaft passes through this hole).
        cap: FaceInfo | None = None
        for idx in seed.neighbors:
            if idx in consumed or labels[idx] != 0:
                continue
            nb = graph.infos[idx]
            if not nb.is_plane or nb.normal is None or not nb.has_inner_loop:
                continue
            if abs_dot(nb.normal, axis) >= self.axis_alignment_threshold:
                cap = nb
                break
        if cap is None:
            return None
        # 2. Through the cap's inner-loop blend neighbours, find a smaller coaxial
        #    outward cylinder (the shaft) — the carrier the head protrudes from.
        shaft: FaceInfo | None = None
        for iln_idx in cap.inner_loop_neighbors:
            iln = graph.infos[iln_idx]
            if not (iln.is_cone or iln.surface_name == "torus"):
                continue
            for bn_idx in iln.neighbors:
                bn = graph.infos[bn_idx]
                if not bn.is_cylinder or bn.radius is None or bn.radial is None:
                    continue
                if bn.radial <= self.radial_threshold:
                    continue
                if bn.radius < seed.radius - radius_tol and self._faces_are_coaxial(graph, bn, seed):
                    shaft = bn
                    break
            if shaft is not None:
                break
        if shaft is None:
            return None
        # 3. Ring: BFS among coaxial same-radius outward cylinders adjacent to it.
        ring: set[int] = {seed.index}
        frontier = [seed.index]
        while frontier:
            cur = frontier.pop()
            for idx in graph.infos[cur].neighbors:
                if idx in ring or idx in consumed or labels[idx] != 0:
                    continue
                nb = graph.infos[idx]
                if not nb.is_cylinder or nb.radius is None or nb.radial is None:
                    continue
                if nb.radial <= self.radial_threshold:
                    continue
                if abs(nb.radius - seed.radius) > radius_tol:
                    continue
                if not self._faces_are_coaxial(graph, nb, seed):
                    continue
                ring.add(idx)
                frontier.append(idx)
        # 4. The ring must close a full circumference.
        if sum(graph.infos[r].u_span for r in ring) < 2.0 * pi - self.hole_angular_coverage_tolerance:
            return None
        # 5. Closure: every ring face's non-ring neighbour must be the cap, the
        #    shaft, a cone/torus blend, or an axis-aligned plane (the head's open
        #    other end, e.g. a slotted face). Anything else means the ring leaks
        #    onto an unrelated exterior surface.
        for r in ring:
            for idx in graph.infos[r].neighbors:
                if idx in ring or idx == cap.index or idx == shaft.index:
                    continue
                nb = graph.infos[idx]
                if nb.is_cone or nb.surface_name == "torus":
                    continue
                if nb.is_plane and nb.normal is not None and abs_dot(nb.normal, axis) >= self.axis_alignment_threshold:
                    continue
                return None
        # 6. Proportion: head diameter > head axial thickness (disk-like, not a
        #    tall pin).
        thickness = max(graph.infos[r].v_span for r in ring)
        if 2.0 * seed.radius <= thickness:
            return None
        # 7. The carrier shaft must be longer axially than the radial protrusion
        #    (R_head - R_shaft). A short hub whose radial step exceeds its length
        #    is a pulley/wheel (disk + through-hub), not a head protruding from a
        #    shaft — the shaft would not be a real base surface.
        if shaft.v_span <= seed.radius - shaft.radius:
            return None
        faces = ring | {cap.index}
        return FeatureInstance(
            label=BOSS,
            kind="boss",
            faces=faces,
            hint_faces={shaft.index},
            reason="coaxial flange boss: ring + cap on a smaller coaxial shaft",
        )

    def _is_boss_side_wall_seed(self, graph: BrepGraph, face: FaceInfo) -> bool:
        # An outward cylindrical side wall is an unambiguous boss seed: its radial
        # direction (outward) is the one signal that distinguishes a boss from a
        # structurally identical recess, whose side walls have no such radial.
        if face.has_inner_loop:
            return False
        if face.is_cylinder and face.radial is not None and face.radial > self.radial_threshold:
            return True
        # A planar side wall may seed a pure-planar boss when it is held by an
        # internal-loop carrier. The boss/pocket discriminator is the offset sign
        # along the carrier normal (which points to the exterior): a boss wall sits on
        # the +offset side, a pocket wall on the -offset side. Requiring the seed to
        # protrude from the carrier therefore seeds bosses and never pockets.
        if face.is_plane and face.normal is not None:
            for idx in face.inner_loop_neighbors:
                carrier = graph.infos[idx]
                if not carrier.has_inner_loop or carrier.normal is None:
                    continue
                if abs_dot(face.normal, carrier.normal) <= 0.35 and self._side_protrudes_from_carrier(
                    graph, face, carrier, carrier.normal
                ):
                    return True
        return False

    def _structural_boss_carrier(
        self, graph: BrepGraph, face: FaceInfo
    ) -> tuple[FaceInfo, Vec3] | None:
        # Returns the carrier face plus the protrusion axis (the direction the boss
        # rises from the carrier). For a planar carrier the axis is carrier.normal;
        # for a curved carrier (Case D) it is the cap plane's normal, since the
        # carrier's own normal is a radial vector that is perpendicular to the
        # protrusion direction.
        # Case A — internal-loop carrier (the classic hint): a face holding this
        # side wall in its inner loop. The boss protrudes through a hole in the base.
        for idx in face.inner_loop_neighbors:
            carrier = graph.infos[idx]
            if carrier.has_inner_loop and carrier.normal is not None:
                return carrier, carrier.normal
        # Case A' — carrier reachable through one base transition face. When a
        # fillet or cone step sits between the side wall and the base, the carrier's
        # inner loop bounds the transition face rather than the side wall itself, so
        # Case A misses it. Cross one transition (cone/torus) neighbour and look for
        # an internal-loop carrier there.
        for idx in face.neighbors:
            neighbor = graph.infos[idx]
            if not (neighbor.is_cone or neighbor.surface_name == "torus"):
                continue
            for carrier_idx in neighbor.inner_loop_neighbors:
                carrier = graph.infos[carrier_idx]
                if carrier.has_inner_loop and carrier.normal is not None:
                    return carrier, carrier.normal
        if not face.is_cylinder or face.axis_dir is None:
            return None
        # Classify axis-aligned planar neighbours into caps (their only neighbours
        # are this wall and its coaxial siblings) and bases (they extend beyond the
        # coaxial group — a real base surface the wall rises from).
        caps: list[FaceInfo] = []
        bases: list[FaceInfo] = []
        for idx in face.neighbors:
            neighbor = graph.infos[idx]
            if not neighbor.is_plane or neighbor.normal is None:
                continue
            if abs_dot(face.axis_dir, neighbor.normal) < self.axis_alignment_threshold:
                continue
            if all(
                n == face.index or self._faces_are_coaxial(graph, graph.infos[n], face)
                for n in neighbor.neighbors
            ):
                caps.append(neighbor)
            else:
                bases.append(neighbor)
        # Case B — the side wall is the OUTER wall of a base surface (connected via
        # the base's outer wire, not through an inner loop). That is the body's own
        # cylindrical wall, not a protrusion, so it is not a boss.
        if bases:
            return None
        # Case C — free-standing cylinder: both ends are caps, no base surface the
        # wall attaches to through topology. A boss must sit on a base whose boundary
        # exceeds the side-wall connection circle (底面边界 > 连接线). The base may live
        # in a separate solid of an assembly, so search geometrically for a coplanar
        # face larger than the cap. If a larger base is found, it is the carrier; if
        # neither cap end has one, try a curved carrier (Case D) before giving up.
        if caps:
            for cap in caps:
                base = self._boss_geometric_base(graph, cap)
                if base is not None and base.normal is not None:
                    return base, base.normal
            # Case D — curved carrier: the wall protrudes from a non-planar face
            # (e.g. a cylindrical body) via its outer wire, so no larger planar base
            # exists. The planar cap gives the protrusion axis; the curved cross-axis
            # neighbour (an internal-loop face whose axis is not parallel to the
            # wall's) is the carrier. Requiring cross-axis excludes the body's own
            # coaxial surface, and requiring an inner loop ensures a real base face.
            cap_axis = caps[0].normal
            cap_indices = {c.index for c in caps}
            for idx in face.neighbors:
                nb = graph.infos[idx]
                if nb.is_plane or idx in cap_indices:
                    continue
                if not nb.has_inner_loop or nb.axis_dir is None:
                    continue
                if abs_dot(face.axis_dir, nb.axis_dir) >= 0.9:
                    continue
                return nb, cap_axis
            return None
        # Fallback: reach a base through one transition face (multi-stage boss whose
        # bottom sits behind a cone step / blend).
        for idx in face.neighbors:
            neighbor = graph.infos[idx]
            if neighbor.is_plane or neighbor.has_inner_loop:
                continue
            through = self._axis_aligned_planar_neighbor(graph, neighbor, axis=face)
            if through is not None and through.normal is not None:
                return through, through.normal
        return None

    def _axis_aligned_planar_neighbor(
        self, graph: BrepGraph, face: FaceInfo, axis: FaceInfo | None = None
    ) -> FaceInfo | None:
        ref = axis or face
        if ref.axis_dir is None:
            return None
        best: FaceInfo | None = None
        best_align = 0.0
        for idx in face.neighbors:
            neighbor = graph.infos[idx]
            if not neighbor.is_plane or neighbor.normal is None or neighbor.has_inner_loop:
                continue
            align = abs_dot(ref.axis_dir, neighbor.normal)
            if align > best_align:
                best_align = align
                best = neighbor
        if best is not None and best_align >= self.axis_alignment_threshold:
            return best
        return None

    def _boss_geometric_base(self, graph: BrepGraph, cap: FaceInfo) -> FaceInfo | None:
        """For a free-standing cylinder's end cap, find the base surface it sits on.

        The base is a planar face coplanar with the cap whose boundary strictly
        contains the cap's connection circle (底面边界 > 侧壁连接线圆周). The base frequently
        lives in a separate solid of an assembly, so this is a geometric search over
        all faces, not a topological one.

        Shape-agnostic containment: the cap's outer boundary is sampled to points, and
        each must lie inside the candidate base's outer-wire polygon (the base boundary
        must wrap the connection circle). This works for any base outline — round,
        rectangular, slot-shaped — and any cap connection curve.
        """
        if cap.normal is None or cap.area <= 0:
            return None
        tol = max(graph.model_diagonal * 1.0e-6, 1.0e-6)
        cap_points = self._face_boundary_points(graph, cap)
        if len(cap_points) < 3:
            return None
        for other in graph.infos:
            if other.index == cap.index or not other.is_plane or other.normal is None:
                continue
            if abs_dot(other.normal, cap.normal) < self.axis_alignment_threshold:
                continue
            if abs(dot(sub(other.center, cap.center), cap.normal)) > tol:
                continue
            base_poly = self._face_boundary_points(graph, other)
            if len(base_poly) < 3:
                continue
            if not all(planar_point_in_polygon(p, base_poly, other.center, other.normal) for p in cap_points):
                continue
            # The cap boundary must extend past the base — at least one cap sample
            # must be strictly interior (not coincident with the base edge), ensuring
            # the base is genuinely larger rather than a coincident same-size face.
            if self._all_points_on_boundary(cap_points, base_poly, other.center, other.normal, tol):
                continue
            return other
        return None

    def _face_boundary_points(self, graph: BrepGraph, info: FaceInfo) -> list[Vec3]:
        """Polygonize the outer boundary of a face into ordered 3D points."""
        try:
            return face_outer_loop_polyline(info.shape)
        except Exception:
            return []

    def _all_points_on_boundary(
        self,
        points: list[Vec3],
        polygon: list[Vec3],
        origin: Vec3,
        normal: Vec3,
        tol: float,
    ) -> bool:
        """True when every point lies on (or negligibly near) a polygon edge — i.e.
        nothing is strictly interior, so the polygon is no larger than the loop."""
        for p in points:
            if not self._point_on_polygon_edge(p, polygon, origin, normal, tol):
                return False
        return True

    def _point_on_polygon_edge(
        self,
        point: Vec3,
        polygon: list[Vec3],
        origin: Vec3,
        normal: Vec3,
        tol: float,
    ) -> bool:
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
        n = len(polygon)
        for i in range(n):
            ax, ay = to2d(polygon[i])
            bx, by = to2d(polygon[(i + 1) % n])
            # distance from point to segment (a,b) in 2D
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= tol * tol:
                if abs(px - ax) <= tol and abs(py - ay) <= tol:
                    return True
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
            cx, cy = ax + t * dx, ay + t * dy
            if abs(px - cx) <= tol and abs(py - cy) <= tol:
                return True
        return False

    def _face_offset_from_carrier(
        self, graph: BrepGraph, face: FaceInfo, carrier: FaceInfo, axis: Vec3
    ) -> float:
        """Signed offset of a face along the protrusion axis (carrier.normal for a
        planar carrier, or the cap normal for a curved carrier)."""
        return dot(sub(face.center, carrier.center), axis)

    def _boss_protrusion_sign(
        self, graph: BrepGraph, seed: FaceInfo, carrier: FaceInfo, axis: Vec3
    ) -> float:
        """Direction the boss protrudes from the carrier, as the sign of the seed's
        offset along the protrusion axis. The seed is an outward cylindrical side wall
        or a planar side wall already known to rise from the carrier, so its offset
        sign fixes the protrusion direction the rest of the ring must share.
        """
        tol = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        offset = self._face_offset_from_carrier(graph, seed, carrier, axis)
        if offset > tol:
            return 1.0
        if offset < -tol:
            return -1.0
        return 0.0

    def _is_boss_ring_face(
        self,
        graph: BrepGraph,
        face: FaceInfo,
        carrier: FaceInfo,
        axis: Vec3,
        side_refs: list[FaceInfo],
        protrusion_sign: float,
    ) -> bool:
        if face.index == carrier.index:
            return False
        # Cones and tori are transition faces (chamfer / fillet / blend) and are
        # never side walls — they bridge boss segments but keep their own label.
        if face.is_cone or face.surface_name == "torus":
            return False
        # Boss side walls protrude from the carrier. The protrusion direction is fixed
        # by the seed's offset sign along the protrusion axis (consistent across a
        # part), which is what separates a boss from a recess, otherwise structurally
        # identical.
        if protrusion_sign > 0:
            if not self._side_protrudes_from_carrier(graph, face, carrier, axis):
                return False
        elif protrusion_sign < 0:
            tol = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
            if self._face_offset_from_carrier(graph, face, carrier, axis) >= -tol:
                return False
        else:
            return False
        # A side wall may carry an inner loop (e.g. a ring-shaped wall whose inner
        # boundary is a hole through the boss). Such a wall is still a valid side wall;
        # its inner-loop neighbours are hole walls (handled by the closure check), not
        # boss fragments. To stop two adjacent bosses merging through one shared wall,
        # require every inner-loop neighbour to be inward (a hole), never outward.
        if face.has_inner_loop:
            for inner_idx in face.inner_loop_neighbors:
                inner = graph.infos[inner_idx]
                if inner.is_cylinder and inner.radial is not None and inner.radial > self.radial_threshold:
                    return False
        if face.is_cylinder and face.radial is not None:
            if face.radial <= self.radial_threshold:
                return False
            if not side_refs:
                return True
            # A multi-stage boss admits coaxial cylinder segments. A free-form
            # outline boss (ring of cylinders around a free-shape opening) has
            # sibling walls that share the generatrix direction (axis_dir) but
            # sit at different axis positions; admit those when the wall is held
            # by the carrier's inner loop, so the ring can close around a non-
            # circular opening without merging into adjacent bosses (which are
            # not inner-loop siblings of this carrier).
            if any(self._faces_are_coaxial(graph, ref, face) for ref in side_refs):
                return True
            if carrier.index in face.inner_loop_neighbors and any(
                self._same_axis_dir(graph, ref, face) for ref in side_refs
            ):
                return True
            return False
        # Lateral test, type-agnostic: a side wall's normal is roughly perpendicular to
        # the protrusion axis (it points sideways, not along it). This admits planes
        # and free-form side walls of a free-shape boss alike. Cylinders with an
        # axis-aligned wall also pass, but they are handled by the branch above.
        if face.normal is None:
            return False
        return abs_dot(face.normal, axis) <= 0.35

    def _grow_boss_ring(
        self,
        graph: BrepGraph,
        labels: list[int],
        seed: FaceInfo,
        carrier: FaceInfo,
        axis: Vec3,
        consumed: set[int],
        protrusion_sign: float,
    ) -> set[int]:
        ring: set[int] = {seed.index}
        side_refs: list[FaceInfo] = [seed] if seed.is_cylinder else []
        queue = [seed.index]
        while queue:
            current_idx = queue.pop(0)
            for neighbor_idx in graph.infos[current_idx].neighbors:
                if neighbor_idx in ring or neighbor_idx in consumed:
                    continue
                if labels[neighbor_idx] != 0:
                    continue
                neighbor = graph.infos[neighbor_idx]
                if self._is_boss_ring_face(graph, neighbor, carrier, axis, side_refs, protrusion_sign):
                    ring.add(neighbor_idx)
                    queue.append(neighbor_idx)
                    if neighbor.is_cylinder:
                        side_refs.append(neighbor)
                    continue
                # A multi-stage boss reaches its next coaxial cylinder stage through
                # a transition bridge (cone step/fillet). Cross the bridge to claim the
                # next outward coaxial cylinder segment; the bridge face itself is never
                # added to the ring, so it keeps its own label (chamfer/other).
                if self._is_boss_bridge_face(graph, labels, neighbor_idx):
                    for far_idx in neighbor.neighbors:
                        if far_idx in ring or far_idx in consumed or labels[far_idx] != 0:
                            continue
                        far = graph.infos[far_idx]
                        if self._is_boss_ring_face(graph, far, carrier, axis, side_refs, protrusion_sign):
                            ring.add(far_idx)
                            queue.append(far_idx)
                            if far.is_cylinder:
                                side_refs.append(far)
        return ring

    def _boss_ring_is_closed(
        self, graph: BrepGraph, ring: set[int], carrier: FaceInfo, axis: Vec3, protrusion_sign: float
    ) -> bool:
        """The ring is closed if every non-side neighbour of a ring face is either the
        carrier, a transition face leading to the top, or another ring face — i.e. the
        ring does not spill out onto unrelated exterior faces.
        """
        for idx in ring:
            for neighbor_idx in graph.infos[idx].neighbors:
                if neighbor_idx in ring or neighbor_idx == carrier.index:
                    continue
                neighbor = graph.infos[neighbor_idx]
                # A side wall may border a hole through the boss: an inward cylindrical
                # wall (or a face already recognised as HOLE) is a valid ring boundary,
                # not a spill onto an exterior face.
                if self._is_hole_wall(neighbor) or neighbor_idx in graph.infos[idx].inner_loop_neighbors:
                    continue
                # A split cylindrical carrier may come as several coaxial pieces
                # (e.g. two half-cylinders of one body); siblings of the carrier
                # are valid ring boundaries, not spills.
                if (
                    carrier.axis_dir is not None
                    and neighbor.is_cylinder
                    and self._faces_are_coaxial(graph, carrier, neighbor)
                ):
                    continue
                if neighbor.has_inner_loop:
                    # A boss top may carry an inner loop (a hole passes through the
                    # protrusion). Such a top is a valid ring boundary, not a spill.
                    if self._is_boss_top_cap(graph, neighbor, ring, carrier, axis, protrusion_sign):
                        continue
                    return False
                if neighbor.is_plane and neighbor.normal is not None:
                    if abs_dot(neighbor.normal, axis) >= self.axis_alignment_threshold:
                        continue
                if self._is_boss_bridge_face(graph, [0] * len(graph.infos), neighbor_idx):
                    continue
                return False
        return True

    def _is_hole_wall(self, face: FaceInfo) -> bool:
        if face.is_cylinder and face.radial is not None:
            return face.radial < -self.radial_threshold
        return False

    def _boss_ring_covering_top(
        self,
        graph: BrepGraph,
        ring: set[int],
        carrier: FaceInfo,
        axis: Vec3,
        labels: list[int],
        consumed: set[int],
        protrusion_sign: float,
    ) -> tuple[int | None, set[int]]:
        """Find the top face covering the ring, possibly through transition faces."""
        direct = self._boss_direct_top(graph, ring, carrier, axis, labels, consumed, protrusion_sign)
        if direct is not None:
            return direct, set()
        return self._boss_bridged_top(graph, ring, carrier, axis, labels, consumed, protrusion_sign)

    def _boss_direct_top(
        self, graph: BrepGraph, ring: set[int], carrier: FaceInfo, axis: Vec3, labels: list[int], consumed: set[int],
        protrusion_sign: float,
    ) -> int | None:
        for idx in ring:
            for neighbor_idx in graph.infos[idx].neighbors:
                if neighbor_idx in ring or neighbor_idx in consumed:
                    continue
                if labels[neighbor_idx] != 0:
                    continue
                candidate = graph.infos[neighbor_idx]
                if self._is_boss_top_cap(graph, candidate, ring, carrier, axis, protrusion_sign):
                    return neighbor_idx
        return None

    def _boss_bridged_top(
        self,
        graph: BrepGraph,
        ring: set[int],
        carrier: FaceInfo,
        axis: Vec3,
        labels: list[int],
        consumed: set[int],
        protrusion_sign: float,
    ) -> tuple[int | None, set[int]]:
        transitions: set[int] = set()
        seen: set[int] = set(ring)
        queue = [idx for idx in ring]
        while queue:
            current_idx = queue.pop(0)
            for neighbor_idx in graph.infos[current_idx].neighbors:
                if neighbor_idx in seen or neighbor_idx in consumed or neighbor_idx == carrier.index:
                    continue
                if labels[neighbor_idx] != 0:
                    continue
                seen.add(neighbor_idx)
                candidate = graph.infos[neighbor_idx]
                # A top reached through a bridge is not adjacent to the ring, so use
                # the geometry-only check (alignment + position); connectivity back to
                # the ring is guaranteed by the bridge BFS that reached it.
                if self._boss_top_cap_geometry(graph, candidate, ring, carrier, axis, protrusion_sign):
                    return neighbor_idx, transitions
                if self._is_boss_bridge_face(graph, labels, neighbor_idx):
                    transitions.add(neighbor_idx)
                    queue.append(neighbor_idx)
                else:
                    transitions.discard(neighbor_idx)
        return None, set()

    def _boss_top_cap_geometry(
        self, graph: BrepGraph, candidate: FaceInfo, ring: set[int], carrier: FaceInfo, axis: Vec3, protrusion_sign: float
    ) -> bool:
        """Position + alignment checks for a boss top cap, independent of how the
        cap is reached. The top must be a plane aligned with the protrusion axis and
        sit strictly beyond every ring wall along the protrusion direction."""
        if candidate.index == carrier.index:
            return False
        if not candidate.is_plane or candidate.normal is None:
            return False
        if abs_dot(candidate.normal, axis) < self.axis_alignment_threshold:
            return False
        tol = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        cap_offset = self._face_offset_from_carrier(graph, candidate, carrier, axis)
        ring_offsets = [self._face_offset_from_carrier(graph, graph.infos[idx], carrier, axis) for idx in ring]
        if not ring_offsets:
            return False
        if protrusion_sign * cap_offset <= max(protrusion_sign * o for o in ring_offsets) + tol:
            return False
        return True

    def _is_boss_top_cap(
        self, graph: BrepGraph, candidate: FaceInfo, ring: set[int], carrier: FaceInfo, axis: Vec3, protrusion_sign: float
    ) -> bool:
        return self._boss_top_cap_geometry(graph, candidate, ring, carrier, axis, protrusion_sign) and any(
            idx in candidate.neighbors for idx in ring
        )

    def _boss_perimeter_exceeds_height(
        self,
        graph: BrepGraph,
        ring: set[int],
        top: int,
        transitions: set[int],
        carrier: FaceInfo,
        axis: Vec3,
    ) -> bool:
        """Reject bosses whose base is small relative to their height.

        Height is the top's distance from the carrier along the protrusion axis —
        i.e. carrier-to-top, which includes any base fillet/cone transition sitting
        between the carrier and the side-wall ring. Perimeter is measured on the
        carrier's own outer boundary — the base face itself, not the boss's top.
        This lets a genuinely tall boss on a large base plate pass, while pegs/pins
        whose base plate is barely larger than the peg still fail. For a cylindrical
        boss on a base of diameter D this reduces to D > height.
        """
        top_info = graph.infos[top]
        top_offset = self._face_offset_from_carrier(graph, top_info, carrier, axis)
        height = abs(top_offset)
        if height <= 1.0e-9:
            return False
        boundary = self._face_boundary_points(graph, carrier)
        if len(boundary) < 3:
            return False
        perimeter = 0.0
        n = len(boundary)
        for i in range(n):
            perimeter += norm(sub(boundary[(i + 1) % n], boundary[i]))
        return perimeter / pi > height

    def _boss_cap_within_ring(
        self,
        graph: BrepGraph,
        top: int,
        cap_faces: set[int],
        ring: set[int],
        carrier: FaceInfo,
    ) -> bool:
        """The cap assembly (top + transition faces + enclosed recess walls) must
        not extend past the side-wall ring. Its outer boundary may at most coincide
        with the ring's top edge — i.e. every neighbour of the assembly is either a
        ring side wall, the carrier, another assembly face, or a hole wall through
        the top. A neighbour that is none of these means the top spills onto an
        unrelated exterior face.
        """
        assembly = set(cap_faces)
        top_info = graph.infos[top]
        hole_walls = set(top_info.inner_loop_neighbors) if top_info.has_inner_loop else set()
        for face_idx in assembly:
            for neighbor_idx in graph.infos[face_idx].neighbors:
                if neighbor_idx in ring or neighbor_idx == carrier.index:
                    continue
                if neighbor_idx in assembly:
                    continue
                if neighbor_idx in hole_walls:
                    continue
                return False
        return True

    def _collect_enclosed_recess_walls(
        self,
        graph: BrepGraph,
        boss_faces: set[int],
        carrier: FaceInfo,
        labels: list[int],
    ) -> set[int]:
        """Faces fully enclosed by the boss structure, collected into the instance:

        - inward cylindrical walls (e.g. the neck of a spool boss);
        - cone/torus blends (the radius-shaped blends of AP224, at the boss root
          or cap rim) — these are part of the boss per definition, not standalone
          features.

        A candidate is an unlabelled inward cylinder or cone/torus face reachable
        from the boss through other candidates. A candidate is kept only when every
        one of its neighbours is the carrier, a boss face, or another kept
        candidate; candidates that touch an unrelated exterior face are removed.
        Mutual dependencies (two blends flanking the same cap) are resolved by
        fixpoint removal. Already-labelled faces (holes/chamfers) are skipped, so
        planar chamfers (a separate Transition_feature) stay out.
        """
        enclosed: set[int] = set()
        seen: set[int] = set(boss_faces) | {carrier.index}
        frontier: list[int] = [idx for idx in boss_faces]
        while frontier:
            current = frontier.pop(0)
            for neighbor_idx in graph.infos[current].neighbors:
                if neighbor_idx in seen or labels[neighbor_idx] != 0:
                    continue
                info = graph.infos[neighbor_idx]
                is_recess = (
                    info.is_cylinder
                    and info.radial is not None
                    and info.radial < -self.radial_threshold
                )
                is_blend = info.is_cone or info.surface_name == "torus"
                if not (is_recess or is_blend):
                    continue
                enclosed.add(neighbor_idx)
                seen.add(neighbor_idx)
                frontier.append(neighbor_idx)
        internal = set(boss_faces) | {carrier.index}
        changed = True
        while changed:
            changed = False
            for idx in list(enclosed):
                info = graph.infos[idx]
                if any(m not in internal and m not in enclosed for m in info.neighbors):
                    enclosed.discard(idx)
                    changed = True
        return enclosed


    def _is_boss_bridge_face(self, graph: BrepGraph, labels: list[int], face_idx: int) -> bool:
        """A transition face that may transparently bridge two boss fragments.

        Chamfer faces are always bridges. Unlabelled cone/torus faces are treated as
        fillet/blend bridges too, since the dataset has no fillet label and they sit
        as ``other`` between boss segments. Plain planar faces and the boss carrier
        are never bridges — that is what prevents two independent bosses sharing one
        carrier from being merged through the carrier face.
        """
        if labels[face_idx] == CHAMFER:
            return True
        if labels[face_idx] != 0:
            return False
        info = graph.infos[face_idx]
        return info.is_cone or info.surface_name == "torus"


    def _recognize_chamfers(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        median_area = median([info.area for info in graph.infos]) if graph.infos else 0.0

        for info in graph.infos:
            if labels[info.index] != 0:
                continue
            if not info.is_plane:
                continue
            if self._plane_is_chamfer(graph, info, median_area):
                features.append(FeatureInstance(label=CHAMFER, kind="chamfer", faces={info.index}, reason="oblique narrow transition face"))
        return features

    def _plane_is_chamfer(self, graph: BrepGraph, info: FaceInfo, median_area: float) -> bool:
        if info.has_inner_loop or info.inner_loop_neighbors or info.normal is None:
            return False
        if info.edge_count < 3 or len(info.neighbors) < 2:
            return False

        oblique_supports: list[int] = []
        for neighbor_idx in info.neighbors:
            neighbor = graph.infos[neighbor_idx]
            if neighbor.normal is None:
                continue
            angle = angle_degrees(info.normal, neighbor.normal)
            if angle is None:
                continue
            acute = min(angle, 180.0 - angle)
            if self.chamfer_min_angle <= acute <= self.chamfer_max_angle:
                oblique_supports.append(neighbor_idx)
        if len(oblique_supports) < 2:
            return False
        if self._connects_only_curved_surfaces(graph, oblique_supports):
            return False
        if not self._has_small_chamfer_area_ratio(graph, info, oblique_supports):
            return False

        # Chamfers are often small, but long edge chamfers can be large. The
        # extra normal-neighborhood test below prevents broad ordinary planes
        # from being marked solely because they meet two faces at an angle.
        area_ok = info.area <= max(median_area * 4.0, (graph.model_diagonal ** 2) * 0.025)
        if area_ok:
            return True

        distinct_supports = 0
        for a_idx in info.neighbors:
            a = graph.infos[a_idx]
            if a.normal is None:
                continue
            for b_idx in info.neighbors:
                if a_idx >= b_idx:
                    continue
                b = graph.infos[b_idx]
                if b.normal is None:
                    continue
                if abs_dot(a.normal, b.normal) < 0.35:
                    distinct_supports += 1
        return distinct_supports >= 1 and info.area <= max(median_area * 8.0, (graph.model_diagonal ** 2) * 0.08)

    def _connects_only_curved_surfaces(self, graph: BrepGraph, support_indices: set[int] | list[int]) -> bool:
        supports = [graph.infos[idx] for idx in support_indices]
        curved_count = sum(1 for support in supports if not support.is_plane)
        planar_count = sum(1 for support in supports if support.is_plane)
        return curved_count >= 2 and planar_count == 0

    def _has_small_chamfer_area_ratio(self, graph: BrepGraph, info: FaceInfo, support_indices: list[int]) -> bool:
        support_count = 0
        for idx in support_indices:
            support = graph.infos[idx]
            if support.area <= 1.0e-12:
                continue
            if info.area <= support.area * self.chamfer_max_support_area_ratio:
                support_count += 1
        return support_count >= 2
