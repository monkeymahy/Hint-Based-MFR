from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from statistics import median

from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder

from .geometry import BrepGraph, FaceInfo, abs_dot, angle_degrees, dot, norm, sub


HOLE = 1
BOSS = 2
CHAMFER = 3


@dataclass
class FeatureInstance:
    label: int
    kind: str
    faces: set[int]
    hint_faces: set[int] = field(default_factory=set)
    reason: str = ""


@dataclass
class RecognitionResult:
    labels: list[int]
    features: list[FeatureInstance]
    graph: BrepGraph

    def one_based_faces(self, feature: FeatureInstance) -> list[int]:
        return [idx + 1 for idx in sorted(feature.faces)]


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

        for feature in self._recognize_round_side_features(graph, labels):
            self._apply(labels, features, feature)

        for feature in self._recognize_planar_bosses(graph, labels):
            self._apply(labels, features, feature)

        return RecognitionResult(labels=labels, features=features, graph=graph)

    def _apply(self, labels: list[int], features: list[FeatureInstance], feature: FeatureInstance) -> None:
        for idx in feature.faces:
            if labels[idx] == 0:
                labels[idx] = feature.label
        features.append(feature)

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
                if any(self._aligned_inner_loop_carrier_count(graph, graph.infos[idx]) != 1 for idx in round_sides):
                    return None, set(), "outward cylindrical wall does not have one boss base carrier"
                feature_faces = self._round_loop_feature_faces(graph, component, round_sides, BOSS, carrier)
                return BOSS, feature_faces, "internal loop with outward cylindrical side wall"

        planar_label = self._classify_planar_loop_component(graph, component, carrier)
        if planar_label == HOLE:
            return None, set(), "planar side-wall ring is not a circular hole"
        if planar_label == BOSS:
            return BOSS, component, "internal loop with closed planar side-wall ring above carrier"
        return None, set(), "ambiguous internal-loop component"

    def _component_axis_matches_carrier(self, graph: BrepGraph, side_indices: list[int], carrier: FaceInfo) -> bool:
        return any(abs_dot(graph.infos[idx].axis_dir, carrier.normal) >= self.axis_alignment_threshold for idx in side_indices)

    def _round_sides_are_coaxial(self, graph: BrepGraph, side_indices: list[int]) -> bool:
        if len(side_indices) <= 1:
            return True
        base = graph.infos[side_indices[0]]
        return all(self._faces_are_coaxial(graph, base, graph.infos[idx]) for idx in side_indices[1:])

    def _round_sides_cover_full_circle(self, side_indices: list[int], graph: BrepGraph) -> bool:
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
            if info.is_plane and any(self._is_loop_feature_cap(side, info) for side in sides):
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
            if info.is_plane and any(self._is_loop_feature_cap(side, info) for side in sides):
                if label == HOLE and info.has_inner_loop and not any(
                    self._is_hole_inner_loop_cap(graph, side, info) for side in sides
                ):
                    continue
                faces.add(idx)
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
            if len(chamfers) < 2:
                continue
            features.append(
                FeatureInstance(
                    label=HOLE,
                    kind="hole",
                    faces={info.index},
                    hint_faces=self._bridged_hole_hint_faces(graph, info, chamfers),
                    reason="inward cylindrical wall between conical chamfers",
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
            elif info.radial > self.radial_threshold and self._is_local_round_feature(graph, info, BOSS, median_area):
                faces = self._round_feature_faces(graph, info, BOSS)
                features.append(
                    FeatureInstance(
                        label=BOSS,
                        kind="boss",
                        faces=faces,
                        reason="outward cylindrical wall",
                    )
                )
        return features

    def _is_local_round_feature(self, graph: BrepGraph, info: FaceInfo, label: int, median_area: float) -> bool:
        if info.surface_type != GeomAbs_Cylinder:
            return False
        if info.inner_loop_neighbors:
            if label == BOSS and self._aligned_inner_loop_carrier_count(graph, info) != 1:
                return False
            if label == HOLE and not self._is_full_cylinder_side(info):
                return False
            return self._has_aligned_inner_loop_carrier(graph, info)
        if label == HOLE:
            return self._is_complete_cylindrical_hole_side(graph, info)
        if label == BOSS and any(graph.infos[idx].has_inner_loop for idx in info.neighbors):
            return False
        if self._has_simple_round_cap(graph, info):
            return True
        return False

    def _has_aligned_inner_loop_carrier(self, graph: BrepGraph, side: FaceInfo) -> bool:
        return self._aligned_inner_loop_carrier_count(graph, side) > 0

    def _aligned_inner_loop_carrier_count(self, graph: BrepGraph, side: FaceInfo) -> int:
        return sum(
            1
            for idx in side.inner_loop_neighbors
            if graph.infos[idx].has_inner_loop
            and abs_dot(side.axis_dir, graph.infos[idx].normal) >= self.axis_alignment_threshold
        )

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
        return any(self._is_axis_aligned_round_cap(side, graph.infos[idx]) for idx in side.neighbors)

    def _is_full_cylinder_side(self, side: FaceInfo) -> bool:
        return side.is_cylinder and abs(side.u_span - (2.0 * pi)) <= self.full_cylinder_u_tolerance

    def _is_axis_aligned_round_cap(self, side: FaceInfo, candidate: FaceInfo) -> bool:
        if side.axis_dir is None or candidate.normal is None:
            return False
        return (
            candidate.is_plane
            and candidate.circle_edges > 0
            and candidate.line_edges == 0
            and abs_dot(side.axis_dir, candidate.normal) >= self.axis_alignment_threshold
        )

    def _is_hole_inner_loop_cap(self, graph: BrepGraph, side: FaceInfo, candidate: FaceInfo) -> bool:
        if not self._is_axis_aligned_round_cap(side, candidate):
            return False
        for neighbor_idx in candidate.neighbors:
            neighbor = graph.infos[neighbor_idx]
            if neighbor.index == side.index or not neighbor.is_cylinder or neighbor.radial is None:
                continue
            if neighbor.radial > self.radial_threshold and self._faces_are_coaxial(graph, side, neighbor):
                return False
        return True

    def _round_feature_faces(self, graph: BrepGraph, side: FaceInfo, label: int) -> set[int]:
        faces = {side.index}
        for neighbor_idx in side.neighbors:
            neighbor = graph.infos[neighbor_idx]
            if neighbor.has_inner_loop:
                if label == HOLE and self._is_hole_inner_loop_cap(graph, side, neighbor):
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
        return (
            candidate.area <= side.area * 1.25
            and candidate.circle_edges > 0
            and abs(alignment) >= self.axis_alignment_threshold
        )

    def _recognize_planar_bosses(self, graph: BrepGraph, labels: list[int]) -> list[FeatureInstance]:
        features: list[FeatureInstance] = []
        median_area = median([info.area for info in graph.infos]) if graph.infos else 0.0
        small_limit = max(median_area * 2.2, (graph.model_diagonal ** 2) * 0.015)
        candidates = {
            info.index
            for info in graph.infos
            if labels[info.index] == 0
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
            if self._component_is_recessed_planar_loop(graph, component):
                continue
            if self._component_is_planar_pad(graph, component):
                features.append(
                    FeatureInstance(
                        label=BOSS,
                        kind="boss",
                        faces=component,
                        reason="face-partition style planar protrusion",
                    )
                )
        return features

    def _has_many_planar_neighbors(self, graph: BrepGraph, info: FaceInfo) -> bool:
        planar_neighbors = [n for n in info.neighbors if graph.infos[n].is_plane]
        return len(planar_neighbors) >= 2

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
        return self.radial_threshold < abs(info.radial) < 0.95

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

    def _has_small_chamfer_area_ratio(self, graph: BrepGraph, info: FaceInfo, support_indices: list[int]) -> bool:
        support_count = 0
        for idx in support_indices:
            support = graph.infos[idx]
            if support.area <= 1.0e-12:
                continue
            if info.area <= support.area * self.chamfer_max_support_area_ratio:
                support_count += 1
        return support_count >= 2
