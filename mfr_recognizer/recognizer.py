from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from statistics import median

from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder

from geometry import BrepGraph, FaceInfo, abs_dot, angle_degrees, dot, norm, sub


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
        full_cylinder_u_tolerance: float = 1.0e-3,
        hole_angular_coverage_tolerance: float = 0.05,
        chamfer_min_angle: float = 18.0,
        chamfer_max_angle: float = 72.0,
        chamfer_max_support_area_ratio: float = 0.35,
    ) -> None:
        self.radial_threshold = radial_threshold
        self.axis_alignment_threshold = axis_alignment_threshold
        self.axis_distance_tolerance_ratio = axis_distance_tolerance_ratio
        self.full_cylinder_u_tolerance = full_cylinder_u_tolerance
        self.hole_angular_coverage_tolerance = hole_angular_coverage_tolerance
        self.chamfer_min_angle = chamfer_min_angle
        self.chamfer_max_angle = chamfer_max_angle
        self.chamfer_max_support_area_ratio = chamfer_max_support_area_ratio

    def recognize_step(self, path: str) -> RecognitionResult:
        return self.recognize_graph(BrepGraph.from_step(path))

    def recognize_graph(self, graph: BrepGraph) -> RecognitionResult:
        labels = [0] * len(graph.infos)
        features: list[FeatureInstance] = []

        for feature in self._recognize_internal_loop_features(graph):
            self._apply(labels, features, feature)

        for feature in self._recognize_chamfers(graph, labels):
            self._apply(labels, features, feature)

        for feature in self._recognize_chamfer_bridged_holes(graph, labels):
            self._apply(labels, features, feature)

        for feature in self._recognize_split_cylindrical_holes(graph, labels):
            self._apply(labels, features, feature)

        for feature in self._recognize_round_side_features(graph, labels):
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

    def _recognize_internal_loop_features(self, graph: BrepGraph) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        carrier_faces = {info.index for info in graph.infos if info.has_inner_loop}
        visited_seeds: set[int] = set()

        for carrier_idx in sorted(carrier_faces):
            carrier = graph.infos[carrier_idx]
            for seed in sorted(carrier.inner_loop_neighbors):
                if seed in carrier_faces or seed in visited_seeds:
                    continue
                blocked = set(carrier_faces)
                component = graph.connected_component([seed], blocked=blocked, limit=128)
                if not component:
                    continue
                if len(component) > 32:
                    # An internal-loop hint should isolate a local subgraph. If
                    # removing carrier faces still leaves a large open component,
                    # the hint is not specific enough; later round-wall rules can
                    # still recover individual cylindrical holes/bosses.
                    continue
                visited_seeds.update(component)
                label, feature_faces, reason = self._classify_loop_component(graph, component, carrier)
                if label is None:
                    continue
                features.append(
                    FeatureInstance(
                        label=label,
                        kind="hole" if label == HOLE else "boss",
                        faces=feature_faces,
                        hint_faces={carrier_idx},
                        reason=reason,
                    )
                )
        return features

    def _classify_loop_component(
        self, graph: BrepGraph, component: set[int], carrier: FaceInfo
    ) -> tuple[int | None, set[int], str]:
        round_sides = [idx for idx in component if graph.infos[idx].is_cylinder and graph.infos[idx].radial is not None]
        radials = [graph.infos[idx].radial for idx in round_sides]
        if radials:
            if not self._component_axis_matches_carrier(graph, round_sides, carrier):
                return None, set(), "round side axis is not normal to internal-loop carrier"
            if not self._round_sides_are_coaxial(graph, round_sides):
                return None, set(), "round side walls do not share one cylinder axis"
            has_inward = any(value < -self.radial_threshold for value in radials)
            has_outward = any(value > self.radial_threshold for value in radials)
            if has_inward and has_outward:
                return None, set(), "mixed inward and outward round walls"
            if min(radials) < -self.radial_threshold:
                if not self._round_sides_cover_full_circle(round_sides, graph):
                    return None, set(), "round side walls do not close a circular hole"
                if not all(graph.infos[idx].radial < -self.radial_threshold for idx in round_sides):
                    return None, set(), "round loop contains weakly inward cylinder walls"
                if not self._round_loop_is_circular_hole_component(graph, component, round_sides):
                    return None, set(), "round loop contains non-circular side walls"
                feature_faces = self._round_loop_feature_faces(graph, component, round_sides, HOLE, carrier)
                if len(feature_faces) == 1 and not any(
                    self._has_multiple_aligned_inner_loop_carriers(graph, graph.infos[idx]) for idx in round_sides
                ):
                    if not self._round_sides_cover_full_circle(round_sides, graph):
                        return None, set(), "single cylindrical wall without a closed hole boundary"
                return HOLE, feature_faces, "internal loop with inward cylindrical side wall"
            if max(radials) > self.radial_threshold:
                # Boss recognition is handled by the structural boss pass.
                return None, set(), "outward cylindrical wall deferred to structural boss pass"

        planar_label = self._classify_planar_loop_component(graph, component, carrier)
        if planar_label == HOLE:
            return None, set(), "planar side-wall ring is not a circular hole"
        if planar_label == BOSS:
            # Boss recognition is handled by the structural boss pass.
            return None, set(), "planar side-wall ring deferred to structural boss pass"
        return None, set(), "ambiguous internal-loop component"

    def _component_axis_matches_carrier(self, graph: BrepGraph, side_indices: list[int], carrier: FaceInfo) -> bool:
        return any(abs_dot(graph.infos[idx].axis_dir, carrier.normal) >= self.axis_alignment_threshold for idx in side_indices)

    def _round_loop_protrudes_from_carrier(self, graph: BrepGraph, side_indices: list[int], carrier: FaceInfo) -> bool:
        return all(self._side_protrudes_from_carrier(graph, graph.infos[idx], carrier) for idx in side_indices)

    def _side_protrudes_from_carrier(self, graph: BrepGraph, side: FaceInfo, carrier: FaceInfo) -> bool:
        if carrier.normal is None:
            return False
        offset = dot(sub(side.center, carrier.center), carrier.normal)
        tolerance = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        return offset > tolerance

    def _round_sides_are_coaxial(self, graph: BrepGraph, side_indices: list[int]) -> bool:
        if len(side_indices) <= 1:
            return True
        base = graph.infos[side_indices[0]]
        return all(self._faces_are_coaxial(graph, base, graph.infos[idx]) for idx in side_indices[1:])

    def _round_sides_cover_full_circle(self, side_indices: list[int], graph: BrepGraph) -> bool:
        if any(self._is_full_cylinder_side(graph.infos[idx]) for idx in side_indices):
            return True
        coverage = sum(min(graph.infos[idx].u_span, 2.0 * pi) for idx in side_indices if graph.infos[idx].is_cylinder)
        return coverage >= (2.0 * pi - self.hole_angular_coverage_tolerance)

    def _faces_are_coaxial(self, graph: BrepGraph, a: FaceInfo, b: FaceInfo) -> bool:
        if a.axis_dir is None or b.axis_dir is None or a.axis_point is None or b.axis_point is None:
            return False
        if abs_dot(a.axis_dir, b.axis_dir) < 1.0 - self.axis_distance_tolerance_ratio:
            return False
        distance = self._axis_distance(a, b)
        tolerance = max(graph.model_diagonal * self.axis_distance_tolerance_ratio, 1.0e-7)
        return distance <= tolerance

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

    def _round_loop_is_circular_hole_component(
        self, graph: BrepGraph, component: set[int], side_indices: list[int]
    ) -> bool:
        sides = [graph.infos[idx] for idx in side_indices]
        for idx in component:
            if idx in side_indices:
                continue
            info = graph.infos[idx]
            if info.is_cone and self._is_hole_loop_chamfer(graph, info, sides):
                continue
            if info.is_plane and any(self._is_hole_feature_cap(graph, side, info) for side in sides):
                continue
            if info.is_plane and any(self._is_oblique_hole_cut_boundary(graph, side, info) for side in sides):
                continue
            return False
        return True

    def _is_hole_loop_chamfer(self, graph: BrepGraph, chamfer: FaceInfo, sides: list[FaceInfo]) -> bool:
        if chamfer.radial is None:
            return False
        if not (self.radial_threshold < abs(chamfer.radial) < 0.95):
            return False
        if not any(self._faces_are_coaxial(graph, side, chamfer) for side in sides):
            return False
        return any(side.index in chamfer.neighbors for side in sides)

    def _round_loop_feature_faces(
        self, graph: BrepGraph, component: set[int], side_indices: list[int], label: int, carrier: FaceInfo
    ) -> set[int]:
        faces = set(side_indices)
        sides = [graph.infos[idx] for idx in side_indices]
        candidate_indices = set(component)
        for side in sides:
            candidate_indices.update(side.neighbors)
        if label == BOSS:
            candidate_indices.discard(carrier.index)
        for idx in candidate_indices:
            info = graph.infos[idx]
            if label == HOLE and info.is_plane:
                if not any(self._is_hole_feature_cap(graph, side, info, carrier) for side in sides):
                    continue
                faces.add(idx)
            elif label == BOSS and info.is_plane:
                if not any(self._is_boss_feature_cap(graph, side, info, carrier) for side in sides):
                    continue
                faces.add(idx)
        if label == BOSS:
            faces.update(self._round_loop_boss_top_faces(graph, side_indices, carrier))
        return faces

    def _is_loop_feature_cap(self, side: FaceInfo, candidate: FaceInfo) -> bool:
        if side.axis_dir is None or candidate.normal is None:
            return False
        return (
            candidate.circle_edges > 0
            and candidate.line_edges == 0
            and abs_dot(side.axis_dir, candidate.normal) >= self.axis_alignment_threshold
        )

    def _classify_planar_loop_component(self, graph: BrepGraph, component: set[int], carrier: FaceInfo) -> int | None:
        if carrier.normal is None:
            return None
        planes = [idx for idx in component if graph.infos[idx].is_plane and graph.infos[idx].normal is not None]
        side_walls = [
            idx
            for idx in planes
            if abs_dot(graph.infos[idx].normal, carrier.normal) <= 0.35
            and (carrier.index in graph.infos[idx].neighbors or carrier.index in graph.infos[idx].inner_loop_neighbors)
        ]
        caps = [idx for idx in planes if abs_dot(graph.infos[idx].normal, carrier.normal) >= 0.88]
        if len(side_walls) < 3 or not caps:
            return None
        for cap in caps:
            if len(graph.infos[cap].neighbors & set(side_walls)) < 3:
                continue
            offset = dot(sub(graph.infos[cap].center, carrier.center), carrier.normal)
            if offset < -1.0e-6:
                return HOLE
            if offset > 1.0e-6:
                return BOSS
        return None

    def _recognize_chamfer_bridged_holes(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        for info in graph.infos:
            if labels[info.index] != 0:
                continue
            if not info.is_cylinder or info.radial is None or info.radial >= -self.radial_threshold:
                continue
            if not self._is_full_cylinder_side(info):
                continue
            chamfers = [
                neighbor_idx
                for neighbor_idx in info.neighbors
                if labels[neighbor_idx] == CHAMFER
                and graph.infos[neighbor_idx].is_cone
                and self._cone_bridges_to_aligned_carrier(graph, info, graph.infos[neighbor_idx])
            ]
            if not chamfers:
                continue
            features.append(
                FeatureInstance(
                    label=HOLE,
                    kind="hole",
                    faces=self._round_feature_faces(graph, info, HOLE),
                    hint_faces=self._bridged_hole_hint_faces(graph, info, chamfers),
                    reason="inward cylindrical wall behind conical chamfer",
                )
            )
        return features

    def _cone_bridges_to_aligned_carrier(self, graph: BrepGraph, side: FaceInfo, cone: FaceInfo) -> bool:
        if cone.radial is None:
            return False
        if not (self.radial_threshold < abs(cone.radial) < 0.95):
            return False
        for neighbor_idx in cone.neighbors:
            if neighbor_idx == side.index:
                continue
            neighbor = graph.infos[neighbor_idx]
            if neighbor.has_inner_loop and abs_dot(side.axis_dir, neighbor.normal) >= self.axis_alignment_threshold:
                return True
        return False

    def _bridged_hole_hint_faces(self, graph: BrepGraph, side: FaceInfo, chamfer_indices: list[int]) -> set[int]:
        hints: set[int] = set()
        for chamfer_idx in chamfer_indices:
            chamfer = graph.infos[chamfer_idx]
            for neighbor_idx in chamfer.neighbors:
                neighbor = graph.infos[neighbor_idx]
                if neighbor.has_inner_loop and abs_dot(side.axis_dir, neighbor.normal) >= self.axis_alignment_threshold:
                    hints.add(neighbor_idx)
        return hints

    def _recognize_split_cylindrical_holes(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        seen: set[int] = set()

        for info in graph.infos:
            if info.index in seen or not self._is_inward_cylindrical_hole_wall(info):
                continue
            if labels[info.index] != 0:
                continue
            group = self._coaxial_inward_cylinder_group(graph, info, labels)
            seen.update(group)
            if len(group) < 2:
                continue
            side_indices = sorted(group)
            if not self._round_sides_cover_full_circle(side_indices, graph):
                continue
            if not self._split_hole_has_boundary_evidence(graph, group):
                continue
            if not self._split_hole_boundaries_are_allowed(graph, group):
                continue
            features.append(
                FeatureInstance(
                    label=HOLE,
                    kind="hole",
                    faces=set(side_indices),
                    hint_faces=self._split_hole_hint_faces(graph, group),
                    reason="coaxial split cylindrical hole wall",
                )
            )
        return features

    def _is_inward_cylindrical_hole_wall(self, info: FaceInfo) -> bool:
        return info.is_cylinder and info.radial is not None and info.radial < -self.radial_threshold

    def _coaxial_inward_cylinder_group(self, graph: BrepGraph, seed: FaceInfo, labels: list[int]) -> set[int]:
        group = {seed.index}
        queue = [seed.index]
        while queue:
            current_idx = queue.pop(0)
            current = graph.infos[current_idx]
            for neighbor_idx in sorted(current.neighbors):
                if neighbor_idx in group or labels[neighbor_idx] != 0:
                    continue
                neighbor = graph.infos[neighbor_idx]
                if not self._is_inward_cylindrical_hole_wall(neighbor):
                    continue
                if not self._faces_are_coaxial(graph, seed, neighbor):
                    continue
                group.add(neighbor_idx)
                queue.append(neighbor_idx)
        return group

    def _split_hole_has_boundary_evidence(self, graph: BrepGraph, group: set[int]) -> bool:
        for side_idx in group:
            side = graph.infos[side_idx]
            if side.inner_loop_neighbors:
                return True
            if self._has_oblique_hole_cut_boundary(graph, side):
                return True
            for neighbor_idx in side.neighbors - group:
                neighbor = graph.infos[neighbor_idx]
                if neighbor.has_inner_loop:
                    return True
                if neighbor.is_plane and self._is_hole_feature_cap(graph, side, neighbor, self._hole_opening_carrier(graph, side)):
                    return True
        return False

    def _split_hole_boundaries_are_allowed(self, graph: BrepGraph, group: set[int]) -> bool:
        for side_idx in group:
            side = graph.infos[side_idx]
            for neighbor_idx in side.neighbors - group:
                neighbor = graph.infos[neighbor_idx]
                if neighbor.is_cylinder:
                    if self._is_inward_cylindrical_hole_wall(neighbor) and self._faces_are_coaxial(graph, side, neighbor):
                        continue
                    return False
                if neighbor.is_cone and self._connects_only_curved_surfaces(graph, neighbor.neighbors):
                    return False
        return True

    def _split_hole_hint_faces(self, graph: BrepGraph, group: set[int]) -> set[int]:
        hints: set[int] = set()
        for side_idx in group:
            side = graph.infos[side_idx]
            hints.update(idx for idx in side.inner_loop_neighbors if graph.infos[idx].has_inner_loop)
            hints.update(idx for idx in side.neighbors if graph.infos[idx].has_inner_loop)
        return hints

    def _recognize_round_side_features(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        median_area = median([info.area for info in graph.infos]) if graph.infos else 0.0

        for info in graph.infos:
            if labels[info.index] != 0 or info.radial is None:
                continue
            if info.radial < -self.radial_threshold and self._is_local_round_feature(graph, info, HOLE, median_area):
                faces = self._round_feature_faces(graph, info, HOLE)
                features.append(
                    FeatureInstance(
                        label=HOLE,
                        kind="hole",
                        faces=faces,
                        reason="inward cylindrical wall",
                    )
                )
            # Outward cylindrical walls are recognized as bosses by the structural
            # boss pass, not here.
        return features

    def _is_local_round_feature(self, graph: BrepGraph, info: FaceInfo, label: int, median_area: float) -> bool:
        if info.surface_type != GeomAbs_Cylinder:
            return False
        if info.inner_loop_neighbors:
            if label == BOSS:
                carriers = self._aligned_inner_loop_carriers(graph, info)
                if len(carriers) != 1:
                    return False
                carrier = graph.infos[carriers[0]]
                return self._side_protrudes_from_carrier(graph, info, carrier) and bool(
                    self._round_loop_boss_top_faces(graph, [info.index], carrier)
                )
            if label == HOLE and not self._is_full_cylinder_side(info):
                return False
            return self._has_aligned_inner_loop_carrier(graph, info) or self._has_oblique_hole_cut_boundary(graph, info)
        if label == HOLE:
            return self._is_complete_cylindrical_hole_side(graph, info)
        if label == BOSS and any(graph.infos[idx].has_inner_loop for idx in info.neighbors):
            return False
        if self._has_simple_round_cap(graph, info):
            return True
        return False

    def _has_aligned_inner_loop_carrier(self, graph: BrepGraph, side: FaceInfo) -> bool:
        return self._aligned_inner_loop_carrier_count(graph, side) > 0

    def _aligned_inner_loop_carriers(self, graph: BrepGraph, side: FaceInfo) -> list[int]:
        return [
            idx
            for idx in side.inner_loop_neighbors
            if graph.infos[idx].has_inner_loop
            and abs_dot(side.axis_dir, graph.infos[idx].normal) >= self.axis_alignment_threshold
        ]

    def _aligned_inner_loop_carrier_count(self, graph: BrepGraph, side: FaceInfo) -> int:
        return len(self._aligned_inner_loop_carriers(graph, side))

    def _single_aligned_boss_carrier(self, graph: BrepGraph, side: FaceInfo) -> FaceInfo | None:
        carriers = self._aligned_inner_loop_carriers(graph, side)
        if len(carriers) != 1:
            return None
        return graph.infos[carriers[0]]

    def _extend_boss_side_walls(self, graph: BrepGraph, labels: list[int], features: list[FeatureInstance]) -> None:
        """Extend each boss with protruding side-wall faces that share its carrier.

        Bosses are often fragmented when their side wall mixes cylinder, plane, and
        cone blend faces: the typed rules grab the cylindrical segment and leave the
        planar/blend segment as ``other``. This pass walks the boundary of every
        recognized boss and pulls in any still-unclaimed neighbor that protrudes from
        the same carrier and behaves as a side wall (perpendicular to the carrier
        normal, or a coaxial outward cylinder), crossing chamfer bridges. It is
        surface-type agnostic, which is what lets a mixed-wall boss close back into a
        single instance.
        """
        for feature in features:
            if feature.label != BOSS:
                continue
            carrier = self._boss_feature_carrier(graph, feature)
            if carrier is None:
                continue
            side_refs = [
                graph.infos[idx]
                for idx in feature.faces
                if graph.infos[idx].is_cylinder
                and graph.infos[idx].radial is not None
                and graph.infos[idx].radial > self.radial_threshold
            ]
            while True:
                new_faces: set[int] = set()
                for idx in list(feature.faces):
                    for neighbor_idx in graph.infos[idx].neighbors:
                        if neighbor_idx in feature.faces:
                            continue
                        if labels[neighbor_idx] != 0:
                            continue
                        candidate = graph.infos[neighbor_idx]
                        if not self._is_boss_side_wall_extension(graph, side_refs, candidate, carrier):
                            continue
                        new_faces.add(neighbor_idx)
                if not new_faces:
                    break
                for neighbor_idx in new_faces:
                    labels[neighbor_idx] = BOSS
                    feature.faces.add(neighbor_idx)
                    candidate = graph.infos[neighbor_idx]
                    if candidate.is_cylinder and candidate.radial is not None and candidate.radial > self.radial_threshold:
                        side_refs.append(candidate)

    def _boss_feature_carrier(self, graph: BrepGraph, feature: FeatureInstance) -> FaceInfo | None:
        for idx in feature.faces:
            carrier = self._single_aligned_boss_carrier(graph, graph.infos[idx])
            if carrier is not None:
                return carrier
        # Pure-planar bosses have no axis_dir, so the aligned-carrier lookup above
        # returns nothing. Fall back to any internal-loop face that holds one of the
        # boss's side walls in its inner loop — that face is the boss's carrier.
        for idx in feature.faces:
            info = graph.infos[idx]
            if not info.is_plane:
                continue
            for neighbor_idx in info.inner_loop_neighbors:
                if graph.infos[neighbor_idx].has_inner_loop:
                    return graph.infos[neighbor_idx]
        return None

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
            carrier = self._structural_boss_carrier(graph, seed)
            if carrier is None:
                continue
            protrusion_sign = self._boss_protrusion_sign(graph, seed, carrier)
            if protrusion_sign == 0:
                continue
            ring = self._grow_boss_ring(graph, labels, seed, carrier, consumed, protrusion_sign)
            if len(ring) < 1:
                continue
            if not self._boss_ring_is_closed(graph, ring, carrier, protrusion_sign):
                continue
            top, transitions = self._boss_ring_covering_top(graph, ring, carrier, labels, consumed, protrusion_sign)
            if top is None:
                continue
            faces = set(ring) | {top} | transitions
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
                    graph, face, carrier
                ):
                    return True
        return False

    def _structural_boss_carrier(self, graph: BrepGraph, face: FaceInfo) -> FaceInfo | None:
        # Case A — internal-loop carrier (the classic hint): a face holding this
        # side wall in its inner loop. The boss protrudes through a hole in the base.
        for idx in face.inner_loop_neighbors:
            if graph.infos[idx].has_inner_loop:
                return graph.infos[idx]
        if not face.is_cylinder or face.axis_dir is None:
            return None
        # Classify axis-aligned planar neighbours into caps (their only neighbour is
        # this side wall) and bases (they extend beyond the side wall — a real base
        # surface the wall rises from).
        caps: list[FaceInfo] = []
        bases: list[FaceInfo] = []
        for idx in face.neighbors:
            neighbor = graph.infos[idx]
            if not neighbor.is_plane or neighbor.normal is None:
                continue
            if abs_dot(face.axis_dir, neighbor.normal) < self.axis_alignment_threshold:
                continue
            if any(n != face.index for n in neighbor.neighbors):
                bases.append(neighbor)
            else:
                caps.append(neighbor)
        # Case B — the side wall is the OUTER wall of a base surface (connected via
        # the base's outer wire, not through an inner loop). That is the body's own
        # cylindrical wall, not a protrusion, so it is not a boss.
        if bases:
            return None
        # Case C — free-standing cylinder: both ends are caps, no base. The boss is
        # the side wall + the top cap; the bottom cap is the carrier (the "bottom"
        # may be any face). Pick the cap whose normal is aligned with the axis as the
        # bottom; the covering-top offset check then selects the anti-aligned top.
        if caps:
            best: FaceInfo | None = None
            best_dot = -1.0
            for cap in caps:
                d = dot(face.axis_dir, cap.normal)
                if d > best_dot:
                    best_dot = d
                    best = cap
            if best is not None and best_dot >= self.axis_alignment_threshold:
                return best
        # Fallback: reach a base through one transition face (multi-stage boss whose
        # bottom sits behind a cone step / blend).
        for idx in face.neighbors:
            neighbor = graph.infos[idx]
            if neighbor.is_plane or neighbor.has_inner_loop:
                continue
            through = self._axis_aligned_planar_neighbor(graph, neighbor, axis=face)
            if through is not None:
                return through
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

    def _face_offset_from_carrier(self, graph: BrepGraph, face: FaceInfo, carrier: FaceInfo) -> float:
        """Signed offset of a face along the carrier normal."""
        if carrier.normal is None:
            return 0.0
        return dot(sub(face.center, carrier.center), carrier.normal)

    def _boss_protrusion_sign(self, graph: BrepGraph, seed: FaceInfo, carrier: FaceInfo) -> float:
        """Direction the boss protrudes from the carrier, as the sign of the seed's
        offset along the carrier normal. The seed is an outward cylindrical side wall
        or a planar side wall already known to rise from the carrier, so its offset
        sign fixes the protrusion direction the rest of the ring must share.
        """
        tol = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        offset = self._face_offset_from_carrier(graph, seed, carrier)
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
        side_refs: list[FaceInfo],
        protrusion_sign: float,
    ) -> bool:
        if face.has_inner_loop or face.index == carrier.index:
            return False
        # Boss side walls protrude from the carrier. The protrusion direction is fixed
        # by the seed's offset sign (the carrier normal's sign is consistent across a
        # part, so this is the same signed test the typed rules rely on — it is what
        # separates a boss from a recess, which is otherwise structurally identical).
        if protrusion_sign > 0:
            if not self._side_protrudes_from_carrier(graph, face, carrier):
                return False
        elif protrusion_sign < 0:
            tol = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
            if self._face_offset_from_carrier(graph, face, carrier) >= -tol:
                return False
        else:
            return False
        if face.is_cylinder and face.radial is not None:
            return face.radial > self.radial_threshold and (
                not side_refs or any(self._faces_are_coaxial(graph, ref, face) for ref in side_refs)
            )
        if face.is_plane and face.normal is not None and carrier.normal is not None:
            return abs_dot(face.normal, carrier.normal) <= 0.35
        return False

    def _grow_boss_ring(
        self,
        graph: BrepGraph,
        labels: list[int],
        seed: FaceInfo,
        carrier: FaceInfo,
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
                if self._is_boss_ring_face(graph, neighbor, carrier, side_refs, protrusion_sign):
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
                        if self._is_boss_ring_face(graph, far, carrier, side_refs, protrusion_sign):
                            ring.add(far_idx)
                            queue.append(far_idx)
                            if far.is_cylinder:
                                side_refs.append(far)
        return ring

    def _boss_ring_is_closed(
        self, graph: BrepGraph, ring: set[int], carrier: FaceInfo, protrusion_sign: float
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
                if neighbor.has_inner_loop:
                    # A boss top may carry an inner loop (a hole passes through the
                    # protrusion). Such a top is a valid ring boundary, not a spill.
                    if self._is_boss_top_cap(graph, neighbor, ring, carrier, protrusion_sign):
                        continue
                    return False
                if neighbor.is_plane and carrier.normal is not None and neighbor.normal is not None:
                    if abs_dot(neighbor.normal, carrier.normal) >= self.axis_alignment_threshold:
                        continue
                if self._is_boss_bridge_face(graph, [0] * len(graph.infos), neighbor_idx):
                    continue
                return False
        return True

    def _boss_ring_covering_top(
        self,
        graph: BrepGraph,
        ring: set[int],
        carrier: FaceInfo,
        labels: list[int],
        consumed: set[int],
        protrusion_sign: float,
    ) -> tuple[int | None, set[int]]:
        """Find the top face covering the ring, possibly through transition faces."""
        direct = self._boss_direct_top(graph, ring, carrier, labels, consumed, protrusion_sign)
        if direct is not None:
            return direct, set()
        return self._boss_bridged_top(graph, ring, carrier, labels, consumed, protrusion_sign)

    def _boss_direct_top(
        self, graph: BrepGraph, ring: set[int], carrier: FaceInfo, labels: list[int], consumed: set[int],
        protrusion_sign: float,
    ) -> int | None:
        for idx in ring:
            for neighbor_idx in graph.infos[idx].neighbors:
                if neighbor_idx in ring or neighbor_idx in consumed:
                    continue
                if labels[neighbor_idx] != 0:
                    continue
                candidate = graph.infos[neighbor_idx]
                if self._is_boss_top_cap(graph, candidate, ring, carrier, protrusion_sign):
                    return neighbor_idx
        return None

    def _boss_bridged_top(
        self,
        graph: BrepGraph,
        ring: set[int],
        carrier: FaceInfo,
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
                if self._is_boss_top_cap(graph, candidate, ring, carrier, protrusion_sign):
                    return neighbor_idx, transitions
                if self._is_boss_bridge_face(graph, labels, neighbor_idx):
                    transitions.add(neighbor_idx)
                    queue.append(neighbor_idx)
                else:
                    transitions.discard(neighbor_idx)
        return None, set()

    def _is_boss_top_cap(
        self, graph: BrepGraph, candidate: FaceInfo, ring: set[int], carrier: FaceInfo, protrusion_sign: float
    ) -> bool:
        if candidate.index == carrier.index:
            return False
        if not candidate.is_plane or candidate.normal is None or carrier.normal is None:
            return False
        if abs_dot(candidate.normal, carrier.normal) < self.axis_alignment_threshold:
            return False
        tol = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        cap_offset = self._face_offset_from_carrier(graph, candidate, carrier)
        ring_offsets = [self._face_offset_from_carrier(graph, graph.infos[idx], carrier) for idx in ring]
        if not ring_offsets:
            return False
        if protrusion_sign * cap_offset <= max(protrusion_sign * o for o in ring_offsets) + tol:
            return False
        return any(idx in candidate.neighbors for idx in ring)

    def _is_boss_side_wall_extension(
        self, graph: BrepGraph, side_refs: list[FaceInfo], candidate: FaceInfo, carrier: FaceInfo
    ) -> bool:
        if not self._side_protrudes_from_carrier(graph, candidate, carrier):
            return False
        if not self._side_protrudes_from_carrier(graph, candidate, carrier):
            return False
        if candidate.has_inner_loop:
            return False
        if candidate.is_cylinder and candidate.radial is not None:
            return candidate.radial > self.radial_threshold and (
                not side_refs or any(self._faces_are_coaxial(graph, ref, candidate) for ref in side_refs)
            )
        if candidate.is_plane and candidate.normal is not None and carrier.normal is not None:
            return abs_dot(candidate.normal, carrier.normal) <= 0.35
        if candidate.is_cone and candidate.radial is not None:
            return self.radial_threshold < abs(candidate.radial) < 0.995 and bool(side_refs) and any(
                self._faces_are_coaxial(graph, ref, candidate) for ref in side_refs
            )
        return False

    def _group_boss_instances(
        self, graph: BrepGraph, labels: list[int], features: list[FeatureInstance]
    ) -> list[FeatureInstance]:
        """Merge boss instances that are fragments of one protrusion.

        A boss may be split into several instances when its side wall is cut by
        chamfer or fillet/blend faces. This pass treats such transition faces as
        transparent bridges: two boss instances belong together when they share the
        same carrier and are connected through the boss-instance adjacency graph once
        bridge faces are made passable. The bridge faces keep their own label
        (chamfer stays chamfer, unlabelled fillet stays other) — only the boss faces
        are regrouped into one instance.

        To avoid merging two independent bosses that happen to sit on one carrier,
        connectivity is constrained to the inner-loop opening: a bridge may only
        connect boss segments that border the same carrier inner-loop neighbour set.
        """
        boss_indices = [idx for idx, feature in enumerate(features) if feature.label == BOSS]
        if len(boss_indices) < 2:
            return features

        carriers = {
            feature_idx: self._boss_feature_carrier(graph, features[feature_idx])
            for feature_idx in boss_indices
        }
        boss_face_to_feature = {
            face_idx: feature_idx
            for feature_idx in boss_indices
            for face_idx in features[feature_idx].faces
        }

        def carrier_id(feature_idx: int) -> int:
            carrier = carriers[feature_idx]
            return carrier.index if carrier is not None else -1 - feature_idx

        def bridged_reach(start: int) -> set[int]:
            """Boss feature indices reachable from start through transition-face bridges.

            A neighbour of a boss face is a bridge only when it is a transition face
            (chamfer, or an unlabelled cone/torus blend) that also touches another boss
            segment of the same carrier. The boss's own carrier face is never a bridge,
            which is what stops two independent bosses sharing one carrier from being
            merged: they touch the carrier, not a shared transition face.
            """
            start_carrier = carriers[start]
            reachable: set[int] = {start}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for face_idx in features[current].faces:
                    for neighbour_idx in graph.infos[face_idx].neighbors:
                        if neighbour_idx in boss_face_to_feature:
                            continue
                        if not self._is_boss_bridge_face(graph, labels, neighbour_idx):
                            continue
                        target_feature = None
                        for other_idx in graph.infos[neighbour_idx].neighbors:
                            candidate_feature = boss_face_to_feature.get(other_idx)
                            if candidate_feature is None or candidate_feature == current:
                                continue
                            if carrier_id(candidate_feature) != carrier_id(current):
                                continue
                            target_feature = candidate_feature
                            break
                        if target_feature is not None and target_feature not in reachable:
                            reachable.add(target_feature)
                            queue.append(target_feature)
            return reachable

        groups: list[set[int]] = []
        visited: set[int] = set()
        for feature_idx in boss_indices:
            if feature_idx in visited:
                continue
            group = bridged_reach(feature_idx)
            visited |= group
            groups.append(group)

        if all(len(group) == 1 for group in groups):
            return features

        merged_features: dict[int, FeatureInstance] = {}
        first_of: dict[int, int] = {}
        for group in groups:
            if len(group) == 1:
                first_of[next(iter(group))] = next(iter(group))
                continue
            first_idx = min(group)
            for member_idx in group:
                first_of[member_idx] = first_idx
            faces = set().union(*(features[idx].faces for idx in group))
            hint_faces = set().union(*(features[idx].hint_faces for idx in group))
            merged_features[first_idx] = FeatureInstance(
                label=BOSS,
                kind="boss",
                faces=faces,
                hint_faces=hint_faces,
                reason="boss fragments rejoined across transition-face bridges",
            )

        grouped: list[FeatureInstance] = []
        for feature_idx, feature in enumerate(features):
            if feature.label != BOSS or feature_idx not in first_of:
                grouped.append(feature)
                continue
            first_idx = first_of[feature_idx]
            if feature_idx == first_idx:
                grouped.append(merged_features.get(feature_idx, feature))
        return grouped

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

    def _has_multiple_aligned_inner_loop_carriers(self, graph: BrepGraph, side: FaceInfo) -> bool:
        return (
            sum(
                1
                for idx in side.inner_loop_neighbors
                if graph.infos[idx].has_inner_loop
                and abs_dot(side.axis_dir, graph.infos[idx].normal) >= self.axis_alignment_threshold
            )
            >= 2
        )

    def _has_simple_round_cap(self, graph: BrepGraph, side: FaceInfo) -> bool:
        if len(side.neighbors) > 3:
            return False
        return any(
            graph.infos[idx].is_plane
            and graph.infos[idx].circle_edges > 0
            and graph.infos[idx].area <= side.area * 0.65
            and abs_dot(side.axis_dir, graph.infos[idx].normal) >= self.axis_alignment_threshold
            for idx in side.neighbors
        )

    def _is_complete_cylindrical_hole_side(self, graph: BrepGraph, side: FaceInfo) -> bool:
        if not self._is_full_cylinder_side(side):
            return False
        carrier = self._hole_opening_carrier(graph, side)
        return any(self._is_hole_feature_cap(graph, side, graph.infos[idx], carrier) for idx in side.neighbors) or (
            carrier is None and self._has_oblique_hole_cut_boundary(graph, side)
        )

    def _is_full_cylinder_side(self, side: FaceInfo) -> bool:
        return side.is_cylinder and (
            abs(side.u_span - (2.0 * pi)) <= self.full_cylinder_u_tolerance or side.full_circle_edges >= 2
        )

    def _is_axis_aligned_round_cap(self, side: FaceInfo, candidate: FaceInfo) -> bool:
        if side.axis_dir is None or candidate.normal is None:
            return False
        return (
            candidate.is_plane
            and candidate.circle_edges > 0
            and candidate.line_edges == 0
            and abs_dot(side.axis_dir, candidate.normal) >= self.axis_alignment_threshold
        )

    def _is_hole_feature_cap(
        self, graph: BrepGraph, side: FaceInfo, candidate: FaceInfo, opening_carrier: FaceInfo | None = None
    ) -> bool:
        if not self._is_axis_aligned_round_cap(side, candidate):
            return False
        if opening_carrier is not None and not self._cap_is_depressed_from_carrier(graph, candidate, opening_carrier):
            return False
        if not candidate.has_inner_loop:
            return True
        if not self._has_coaxial_inward_step_side(graph, side, candidate):
            return False
        for neighbor_idx in candidate.neighbors:
            neighbor = graph.infos[neighbor_idx]
            if neighbor.index == side.index or not neighbor.is_cylinder or neighbor.radial is None:
                continue
            if neighbor.radial > self.radial_threshold and self._faces_are_coaxial(graph, side, neighbor):
                return False
        return True

    def _has_coaxial_inward_step_side(self, graph: BrepGraph, side: FaceInfo, candidate: FaceInfo) -> bool:
        for neighbor_idx in candidate.neighbors:
            neighbor = graph.infos[neighbor_idx]
            if neighbor.index == side.index or not neighbor.is_cylinder or neighbor.radial is None:
                continue
            if neighbor.radial < -self.radial_threshold and self._faces_are_coaxial(graph, side, neighbor):
                return True
        return False

    def _cap_is_depressed_from_carrier(self, graph: BrepGraph, cap: FaceInfo, carrier: FaceInfo) -> bool:
        if cap.index == carrier.index or carrier.normal is None:
            return False
        offset = dot(sub(cap.center, carrier.center), carrier.normal)
        tolerance = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        return abs(offset) > tolerance

    def _hole_opening_carrier(self, graph: BrepGraph, side: FaceInfo) -> FaceInfo | None:
        carriers = self._aligned_inner_loop_carriers(graph, side)
        if len(carriers) == 1:
            return graph.infos[carriers[0]]
        return None

    def _has_oblique_hole_cut_boundary(self, graph: BrepGraph, side: FaceInfo) -> bool:
        return any(self._is_oblique_hole_cut_boundary(graph, side, graph.infos[idx]) for idx in side.neighbors)

    def _is_oblique_hole_cut_boundary(self, graph: BrepGraph, side: FaceInfo, candidate: FaceInfo) -> bool:
        if side.axis_dir is None or candidate.normal is None:
            return False
        if not candidate.is_plane:
            return False
        if side.index not in candidate.neighbors:
            return False
        if candidate.circle_edges <= 0:
            return False
        alignment = abs_dot(side.axis_dir, candidate.normal)
        return 0.15 <= alignment <= 0.98

    def _is_boss_feature_cap(self, graph: BrepGraph, side: FaceInfo, candidate: FaceInfo, carrier: FaceInfo) -> bool:
        if side.axis_dir is None or candidate.normal is None or carrier.normal is None:
            return False
        if not candidate.is_plane:
            return False
        if abs_dot(side.axis_dir, candidate.normal) < self.axis_alignment_threshold:
            return False
        cap_offset = dot(sub(candidate.center, carrier.center), carrier.normal)
        side_offset = dot(sub(side.center, carrier.center), carrier.normal)
        tolerance = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        return cap_offset > side_offset + tolerance

    def _round_loop_boss_top_faces(self, graph: BrepGraph, side_indices: list[int], carrier: FaceInfo) -> set[int]:
        top_faces: set[int] = set()
        for side_idx in side_indices:
            side = graph.infos[side_idx]
            for neighbor_idx in side.neighbors:
                neighbor = graph.infos[neighbor_idx]
                if self._is_boss_feature_cap(graph, side, neighbor, carrier):
                    top_faces.add(neighbor_idx)
                    continue
                if not self._is_boss_top_transition(graph, side, neighbor):
                    continue
                for top_idx in neighbor.neighbors - {side.index, carrier.index}:
                    top = graph.infos[top_idx]
                    if self._is_boss_feature_cap(graph, side, top, carrier):
                        top_faces.add(top_idx)
        return top_faces

    def _is_boss_top_transition(self, graph: BrepGraph, side: FaceInfo, candidate: FaceInfo) -> bool:
        if side.axis_dir is None or candidate.index not in side.neighbors:
            return False
        if not candidate.is_cone or candidate.radial is None:
            return False
        if len(candidate.neighbors) > 6 or candidate.edge_count > 8:
            return False
        if self._connects_only_curved_surfaces(graph, candidate.neighbors):
            return False
        return self.radial_threshold < abs(candidate.radial) < 0.995

    def _round_feature_faces(
        self, graph: BrepGraph, side: FaceInfo, label: int, carrier: FaceInfo | None = None
    ) -> set[int]:
        faces = {side.index}
        opening_carrier = carrier if label == HOLE else None
        if label == HOLE and opening_carrier is None:
            opening_carrier = self._hole_opening_carrier(graph, side)
        for neighbor_idx in side.neighbors:
            neighbor = graph.infos[neighbor_idx]
            if neighbor.has_inner_loop:
                if label == HOLE and self._is_hole_feature_cap(graph, side, neighbor, opening_carrier):
                    faces.add(neighbor_idx)
                elif label == BOSS and neighbor.area <= side.area * 0.55:
                    faces.add(neighbor_idx)
                continue
            if neighbor.is_round_side and neighbor.radial is not None:
                same_sign = (neighbor.radial < -self.radial_threshold) if label == HOLE else (neighbor.radial > self.radial_threshold)
                if same_sign and self._faces_are_coaxial(graph, side, neighbor):
                    faces.add(neighbor_idx)
            elif neighbor.is_plane and self._is_feature_cap(side, neighbor, label):
                faces.add(neighbor_idx)
        return faces

    def _is_feature_cap(self, side: FaceInfo, candidate: FaceInfo, label: int) -> bool:
        if side.axis_dir is None or candidate.normal is None:
            return candidate.area <= side.area
        if len(side.neighbors) > 3:
            return False
        alignment = dot(side.axis_dir, candidate.normal)
        if label == BOSS and alignment > -self.axis_alignment_threshold:
            return False
        if label == HOLE and candidate.full_circle_edges < 1:
            return False
        return (
            candidate.area <= side.area * 1.25
            and candidate.circle_edges > 0
            and abs(alignment) >= self.axis_alignment_threshold
        )

    def _recognize_planar_bosses(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features = self._recognize_general_planar_bosses(graph, labels)
        reserved_faces = {face_idx for feature in features for face_idx in feature.faces}
        median_area = median([info.area for info in graph.infos]) if graph.infos else 0.0
        small_limit = max(median_area * 2.2, (graph.model_diagonal ** 2) * 0.015)
        candidates = {
            info.index
            for info in graph.infos
            if labels[info.index] == 0
            and info.index not in reserved_faces
            and info.is_plane
            and not info.has_inner_loop
            and info.area <= small_limit
            and self._has_many_planar_neighbors(graph, info)
        }

        seen: set[int] = set()
        for seed in sorted(candidates):
            if seed in seen:
                continue
            component = graph.connected_component([seed], blocked=set(range(len(graph.infos))) - candidates, limit=64)
            seen.update(component)
            if len(component) < 4:
                continue
            if not self._component_has_inner_loop_support(graph, component):
                continue
            if self._component_is_recessed_planar_loop(graph, component):
                continue
            if self._component_is_planar_pad(graph, component) and self._component_has_planar_boss_top(graph, component):
                features.append(
                    FeatureInstance(
                        label=BOSS,
                        kind="boss",
                        faces=component,
                        reason="face-partition style planar protrusion",
                    )
                )
        return features

    def _recognize_general_planar_bosses(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        consumed_faces: set[int] = set()

        for carrier in graph.infos:
            if not carrier.has_inner_loop or carrier.normal is None:
                continue
            for seed_idx in sorted(carrier.inner_loop_neighbors):
                if seed_idx in consumed_faces:
                    continue
                component, chamfer_bridges = self._general_boss_component_from_seed(graph, carrier, seed_idx, labels)
                if not component or component & consumed_faces:
                    continue
                if not self._general_boss_component_is_valid(graph, component, carrier):
                    continue
                consumed_faces.update(component)
                features.append(
                    FeatureInstance(
                        label=BOSS,
                        kind="boss",
                        faces=component,
                        hint_faces={carrier.index} | chamfer_bridges,
                        reason="general planar protrusion from internal-loop carrier",
                    )
                )
        return features

    def _general_boss_component_from_seed(
        self, graph: BrepGraph, carrier: FaceInfo, seed_idx: int, labels: list[int]
    ) -> tuple[set[int], set[int]]:
        if labels[seed_idx] in {HOLE, BOSS} or seed_idx == carrier.index:
            return set(), set()

        component: set[int] = set()
        chamfer_bridges: set[int] = set()
        seen = {carrier.index}
        queue = [seed_idx]
        limit = 128

        while queue and len(seen) <= limit:
            current_idx = queue.pop(0)
            if current_idx in seen:
                continue
            seen.add(current_idx)
            current_label = labels[current_idx]
            current = graph.infos[current_idx]

            if current_label in {HOLE, BOSS}:
                continue
            if current_label == CHAMFER:
                chamfer_bridges.add(current_idx)
            elif self._is_general_boss_candidate_face(graph, current, carrier):
                component.add(current_idx)
            else:
                continue

            for neighbor_idx in sorted(current.neighbors):
                if neighbor_idx in seen or neighbor_idx == carrier.index:
                    continue
                neighbor_label = labels[neighbor_idx]
                if neighbor_label in {HOLE, BOSS}:
                    continue
                if neighbor_label == CHAMFER or self._is_general_boss_candidate_face(graph, graph.infos[neighbor_idx], carrier):
                    queue.append(neighbor_idx)

        return component, chamfer_bridges

    def _is_general_boss_candidate_face(self, graph: BrepGraph, info: FaceInfo, carrier: FaceInfo) -> bool:
        return self._is_general_boss_side_wall(graph, info, carrier) or self._is_general_boss_top_face(graph, info, carrier)

    def _is_general_boss_side_wall(self, graph: BrepGraph, info: FaceInfo, carrier: FaceInfo) -> bool:
        if not info.is_plane or info.normal is None or carrier.normal is None:
            return False
        if abs_dot(info.normal, carrier.normal) > 0.35:
            return False
        offset = dot(sub(info.center, carrier.center), carrier.normal)
        tolerance = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        return offset > -tolerance

    def _is_general_boss_top_face(self, graph: BrepGraph, info: FaceInfo, carrier: FaceInfo) -> bool:
        if not info.is_plane or info.normal is None or carrier.normal is None:
            return False
        if abs_dot(info.normal, carrier.normal) < 0.88:
            return False
        offset = dot(sub(info.center, carrier.center), carrier.normal)
        tolerance = max(graph.model_diagonal * 1.0e-7, 1.0e-7)
        return offset > tolerance

    def _general_boss_component_is_valid(self, graph: BrepGraph, component: set[int], carrier: FaceInfo) -> bool:
        side_walls = {
            idx
            for idx in component
            if self._is_general_boss_side_wall(graph, graph.infos[idx], carrier)
        }
        top_faces = {
            idx
            for idx in component
            if self._is_general_boss_top_face(graph, graph.infos[idx], carrier)
        }
        if len(side_walls) < 3 or not top_faces:
            return False
        for top_idx in top_faces:
            if len(graph.infos[top_idx].neighbors & side_walls) >= 2:
                return True
        return False

    def _component_has_planar_boss_top(self, graph: BrepGraph, component: set[int]) -> bool:
        for carrier_idx in self._component_inner_loop_carriers(graph, component):
            carrier = graph.infos[carrier_idx]
            if self._classify_planar_loop_component(graph, component, carrier) == BOSS:
                return True
        return False

    def _has_many_planar_neighbors(self, graph: BrepGraph, info: FaceInfo) -> bool:
        planar_neighbors = [n for n in info.neighbors if graph.infos[n].is_plane]
        return len(planar_neighbors) >= 2

    def _component_has_inner_loop_support(self, graph: BrepGraph, component: set[int]) -> bool:
        return bool(self._component_inner_loop_carriers(graph, component))

    def _component_inner_loop_carriers(self, graph: BrepGraph, component: set[int]) -> list[int]:
        return [
            info.index
            for info in graph.infos
            if info.has_inner_loop and len(info.inner_loop_neighbors & component) >= 2
        ]

    def _component_is_recessed_planar_loop(self, graph: BrepGraph, component: set[int]) -> bool:
        carriers = {
            neighbor_idx
            for idx in component
            for neighbor_idx in graph.infos[idx].inner_loop_neighbors | graph.infos[idx].neighbors
            if graph.infos[neighbor_idx].has_inner_loop
        }
        return any(self._classify_planar_loop_component(graph, component, graph.infos[carrier_idx]) == HOLE for carrier_idx in carriers)

    def _component_is_planar_pad(self, graph: BrepGraph, component: set[int]) -> bool:
        normals = [graph.infos[idx].normal for idx in component if graph.infos[idx].normal is not None]
        if len(normals) < 3:
            return False
        has_parallel_pair = any(abs_dot(a, b) > 0.9 for i, a in enumerate(normals) for b in normals[i + 1 :])
        has_orthogonal_pair = any(abs_dot(a, b) < 0.25 for i, a in enumerate(normals) for b in normals[i + 1 :])
        return has_parallel_pair and has_orthogonal_pair

    def _recognize_chamfers(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        median_area = median([info.area for info in graph.infos]) if graph.infos else 0.0

        for info in graph.infos:
            if labels[info.index] != 0:
                continue
            if info.is_cone and self._cone_is_chamfer(graph, info):
                features.append(FeatureInstance(label=CHAMFER, kind="chamfer", faces={info.index}, reason="conical transition face"))
                continue
            if not info.is_plane:
                continue
            if self._plane_is_chamfer(graph, info, median_area):
                features.append(FeatureInstance(label=CHAMFER, kind="chamfer", faces={info.index}, reason="oblique narrow transition face"))
        return features

    def _cone_is_chamfer(self, graph: BrepGraph, info: FaceInfo) -> bool:
        if info.radial is None:
            return False
        if len(info.neighbors) < 2:
            return False
        if len(info.neighbors) > 6 or info.edge_count > 8:
            return False
        if self._connects_only_curved_surfaces(graph, info.neighbors):
            return False
        if not any(graph.infos[idx].is_plane for idx in info.neighbors):
            return False
        return self.radial_threshold < abs(info.radial) < 0.995

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
