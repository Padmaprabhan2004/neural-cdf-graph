import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

examples_root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(examples_root))
sys.path.append(str(repo_root))

from robot_cdf import RobotCDF
from sdf_marching.overlap import get_overlaps_graph_numpy
from utils_env import create_obstacles
from utils_visualization import visualize_workspace,visualize_cdf_field

@dataclass
class BubbleRecord:
    center: np.ndarray
    radius: float
    clearance: float
    sobol_index: int


@dataclass
class CoverageDiagnostics:
    raw_bubbles: int
    raw_edges: int
    raw_connected_components: int
    raw_largest_component: int
    pruned_bubbles: int
    pruned_edges: int
    pruned_connected_components: int
    pruned_largest_component: int
    sobol_samples: int
    free_samples: int
    bubbles: int
    edges: int
    connected_components: int
    largest_component: int
    min_radius: float
    max_radius: float
    mean_radius: float
    median_radius: float
    validation_samples: int
    validation_free_samples: int
    validation_covered_free_samples: int
    estimated_free_coverage: float
    estimated_uncovered_free_fraction: float
    batches_processed: int
    accepted_new_bubbles: int
    radius_prune_threshold: float
    major_components: List[int]
    major_component_supports: List[float]
    major_component_sizes: List[int]


@dataclass
class BubbleGraphResult:
    q_min: np.ndarray
    q_max: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    clearances: np.ndarray
    sobol_indices: np.ndarray
    edges: np.ndarray
    component_ids: np.ndarray
    component_supports: np.ndarray
    major_component_mask: np.ndarray
    diagnostics: CoverageDiagnostics
    obstacles: np.ndarray
    validation_points: np.ndarray
    validation_free_mask: np.ndarray
    validation_covered_mask: np.ndarray

#bubble radius -> r(q)=h(q)-eps

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct a full free-space bubble graph for 2D q-space.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sobol_samples", type=int, default=4096)
    parser.add_argument("--validation_samples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--selection_per_batch", type=int, default=16)
    parser.add_argument("--safety_margin", type=float, default=0.08)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--min_radius", type=float, default=0.04)
    parser.add_argument("--free_threshold", type=float, default=0.0)
    parser.add_argument("--coverage_tol", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_dropout", action="store_true", default=True)
    parser.add_argument("--no_dropout", dest="use_dropout", action="store_false")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--save_name", type=str, default="ncdf_bubble_graph")
    parser.add_argument("--obstacle_points", type=int, default=50)
    parser.add_argument("--validate_batch_size", type=int, default=256)
    parser.add_argument("--prune_radius_percentile", type=float, default=5.0)
    parser.add_argument("--prune_radius_floor", type=float, default=0.05)
    parser.add_argument("--major_component_min_bubbles", type=int, default=6)
    parser.add_argument("--major_component_min_support", type=float, default=0.03)
    return parser.parse_args()


def setup_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def concatenate_obstacle_list(obstacle_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(obstacle_list, axis=0)


def sobol_draw(num_samples: int, q_min: np.ndarray, q_max: np.ndarray, seed: int) -> np.ndarray:
    engine = torch.quasirandom.SobolEngine(dimension=len(q_min), scramble=True, seed=seed)
    samples = engine.draw(num_samples).cpu().numpy()
    return q_min + samples * (q_max - q_min)


def batch_query_clearance(
    robot_cdf: RobotCDF,
    configs: np.ndarray,
    obstacle_points: torch.Tensor,
    batch_size: int = 256,
    safety_margin: float = 0.1,
) -> np.ndarray:
    if len(configs) == 0:
        return np.zeros((0,), dtype=np.float32)

    clearances: List[np.ndarray] = []
    obstacle_points = obstacle_points.to(device=robot_cdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.unsqueeze(0)

    for start in range(0, len(configs), batch_size):
        end = min(start + batch_size, len(configs))
        q_batch = torch.tensor(configs[start:end], device=robot_cdf.device, dtype=torch.float32)#batch of sobol points
        points_batch = obstacle_points.expand(len(q_batch), -1, -1)
        cdf_values = robot_cdf.query_cdf(points_batch, q_batch)
        clearance = cdf_values.min(dim=1).values - safety_margin
        clearances.append(clearance.detach().cpu().numpy())

    return np.concatenate(clearances, axis=0)


def bubble_radius(clearance: float, epsilon: float, min_radius: float) -> float:
    return float(max(clearance - epsilon, min_radius))

#checks if a point is inside any existing bubble, if outisde all value>0 (||q-c_i||-ri>>>0 are chosen)
def signed_distance_to_union(point: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> float:
    if len(centers) == 0:
        return float("inf")
    dists = np.linalg.norm(centers - point[None, :], axis=1) - radii
    return float(np.min(dists))

#chooses bubbles that have large radius and contribute to more coverage, ones which are well apart from other bubbles
def bubble_contribution_score(point: np.ndarray, radius: float, centers: np.ndarray, radii: np.ndarray) -> float:
    return signed_distance_to_union(point, centers, radii) + radius


def prune_bubbles_by_radius(
    centers: np.ndarray,
    radii: np.ndarray,
    clearances: np.ndarray,
    sobol_indices: np.ndarray,
    percentile: float,
    floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    if len(radii) == 0:
        mask = np.zeros((0,), dtype=bool)
        return centers, radii, clearances, sobol_indices, float(floor), mask

    threshold = float(max(floor, np.percentile(radii, percentile)))
    mask = radii >= threshold
    if not np.any(mask):
        mask = np.zeros_like(radii, dtype=bool)
        mask[int(np.argmax(radii))] = True
    return centers[mask], radii[mask], clearances[mask], sobol_indices[mask], threshold, mask


def compute_component_supports(
    centers: np.ndarray,
    radii: np.ndarray,
    validation_points: np.ndarray,
    validation_free_mask: np.ndarray,
    major_component_min_bubbles: int,
    major_component_min_support: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[float], List[int]]:
    if len(centers) == 0:
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_supports = np.zeros((0,), dtype=np.float32)
        empty_mask = np.zeros((0,), dtype=bool)
        return empty_ids, empty_supports, empty_mask, [], [], []

    graph = get_overlaps_graph_numpy(centers, radii)
    if graph.ecount() > 0:
        component_ids = np.array(graph.connected_components().membership, dtype=np.int64)
    else:
        component_ids = np.arange(len(centers), dtype=np.int64)

    validation_points = np.asarray(validation_points, dtype=np.float32)
    validation_free_mask = np.asarray(validation_free_mask, dtype=bool)
    free_total = int(validation_free_mask.sum())

    component_count = int(component_ids.max() + 1) if len(component_ids) else 0
    component_supports = np.zeros(component_count, dtype=np.float32)
    component_sizes = np.zeros(component_count, dtype=np.int64)

    if free_total > 0 and len(validation_points) > 0 and component_count > 0:
        for comp_id in range(component_count):
            bubble_mask = component_ids == comp_id
            component_sizes[comp_id] = int(bubble_mask.sum())
            if not np.any(bubble_mask):
                continue
            comp_centers = centers[bubble_mask]
            comp_radii = radii[bubble_mask]
            d = np.linalg.norm(validation_points[:, None, :] - comp_centers[None, :, :], axis=-1) - comp_radii[None, :]
            covered = np.min(d, axis=1) <= 0.0
            component_supports[comp_id] = float(np.logical_and(validation_free_mask, covered).sum() / free_total)
    else:
        for comp_id in range(component_count):
            component_sizes[comp_id] = int((component_ids == comp_id).sum())

    major_component_ids = [
        int(comp_id)
        for comp_id in range(component_count)
        if component_sizes[comp_id] >= major_component_min_bubbles
        and component_supports[comp_id] >= major_component_min_support
    ]
    major_mask = np.isin(component_ids, major_component_ids)
    return (
        component_ids,
        component_supports,
        major_mask,
        major_component_ids,
        component_supports.tolist(),
        component_sizes.tolist(),
    )


def build_bubble_union(
    robot_cdf: RobotCDF,
    sobol_points: np.ndarray,
    obstacle_points: torch.Tensor,
    q_min: np.ndarray,
    q_max: np.ndarray,
    batch_size: int,
    selection_per_batch: int,
    epsilon: float,
    min_radius: float,
    safety_margin: float,
    free_threshold: float,
    coverage_tol: float,
    patience: int,
    prune_radius_percentile: float,
    prune_radius_floor: float,
    major_component_min_bubbles: int,
    major_component_min_support: float,
    validation_points: Optional[np.ndarray] = None,
    validation_free_mask: Optional[np.ndarray] = None,
) -> Tuple[BubbleGraphResult, List[BubbleRecord]]:
    #point clearance for sampled sobol points
    clearances = batch_query_clearance(
        robot_cdf=robot_cdf,
        configs=sobol_points,#todo: compute clearance for the sobol pts
        obstacle_points=obstacle_points,
        batch_size=batch_size,
        safety_margin=safety_margin,
    )

    free_mask = clearances > free_threshold
    free_points = sobol_points[free_mask]
    free_clearances = clearances[free_mask]
    free_indices = np.nonzero(free_mask)[0]
    #only free sampled sobol points are queried for bubbles

    selected_bubbles: List[BubbleRecord] = []
    batches_processed = 0
    accepted_new_bubbles = 0
    no_progress_batches = 0
    #for points above threshold do the bubble graph
    for start in range(0, len(free_points), batch_size):
        end = min(start + batch_size, len(free_points))

        #take a batch of points 
        batch_points = free_points[start:end]
        batch_clearances = free_clearances[start:end]
        batch_indices = free_indices[start:end]

        if len(batch_points) == 0:
            continue

        if selected_bubbles:
            centers = np.stack([b.center for b in selected_bubbles], axis=0)
            radii = np.array([b.radius for b in selected_bubbles], dtype=np.float32)
        else:
            centers = np.zeros((0, 2), dtype=np.float32)
            radii = np.zeros((0,), dtype=np.float32)

        candidate_records = []
        
        # each batch point is a candidate
        for q, clearance, sobol_idx in zip(batch_points, batch_clearances, batch_indices):
            radius = bubble_radius(clearance, epsilon=epsilon, min_radius=min_radius)
            score = clearance if not selected_bubbles else bubble_contribution_score(q, radius, centers, radii)
            candidate_records.append((score, q, radius, clearance, sobol_idx))

        candidate_records.sort(key=lambda item: item[0], reverse=True)
        accepted_in_batch = 0

        for score, q, radius, clearance, sobol_idx in candidate_records:
            if score <= 0.0 and selected_bubbles:
                continue

            if selected_bubbles:
                current_centers = np.stack([b.center for b in selected_bubbles], axis=0)
                current_radii = np.array([b.radius for b in selected_bubbles], dtype=np.float32)
                current_score = bubble_contribution_score(q, radius, current_centers, current_radii)
            else:
                current_score = float("inf")

            if current_score > 0.0 or not selected_bubbles:
                selected_bubbles.append(
                    BubbleRecord(
                        center=np.asarray(q, dtype=np.float32),
                        radius=float(radius),
                        clearance=float(clearance),
                        sobol_index=int(sobol_idx),
                    )
                )
                accepted_in_batch += 1
                accepted_new_bubbles += 1

                if accepted_in_batch >= selection_per_batch:
                    break

        batches_processed += 1
        if accepted_in_batch == 0:
            no_progress_batches += 1
        else:
            no_progress_batches = 0

        diagnostics = estimate_coverage(
            robot_cdf=robot_cdf,
            bubbles=selected_bubbles,
            q_min=q_min,
            q_max=q_max,
            obstacle_points=obstacle_points,
            num_samples=len(validation_points) if validation_points is not None else 0,
            validation_points=validation_points,
            validation_free_mask=validation_free_mask,
            batch_size=batch_size,
            safety_margin=safety_margin,
            epsilon=epsilon,
            min_radius=min_radius,
        )

        if diagnostics.estimated_uncovered_free_fraction <= coverage_tol:
            break
        if no_progress_batches >= patience:
            break

    if selected_bubbles:
        raw_centers = np.stack([b.center for b in selected_bubbles], axis=0)
        raw_radii = np.array([b.radius for b in selected_bubbles], dtype=np.float32)
        raw_clearances = np.array([b.clearance for b in selected_bubbles], dtype=np.float32)
        raw_sobol_indices = np.array([b.sobol_index for b in selected_bubbles], dtype=np.int64)
    else:
        raw_centers = np.zeros((0, 2), dtype=np.float32)
        raw_radii = np.zeros((0,), dtype=np.float32)
        raw_clearances = np.zeros((0,), dtype=np.float32)
        raw_sobol_indices = np.zeros((0,), dtype=np.int64)

    raw_graph = get_overlaps_graph_numpy(raw_centers, raw_radii) if len(raw_centers) else None
    if raw_graph is not None:
        raw_edges = np.array(raw_graph.get_edgelist(), dtype=np.int64) if raw_graph.ecount() else np.zeros((0, 2), dtype=np.int64)
        raw_components = raw_graph.connected_components()
        raw_component_count = len(raw_components)
        raw_largest_component = max(raw_components.sizes()) if len(raw_components) else 0
    else:
        raw_edges = np.zeros((0, 2), dtype=np.int64)
        raw_component_count = 0
        raw_largest_component = 0

    centers, radii, clearances_kept, sobol_indices, radius_prune_threshold, prune_mask = prune_bubbles_by_radius(
        raw_centers,
        raw_radii,
        raw_clearances,
        raw_sobol_indices,
        percentile=prune_radius_percentile,
        floor=prune_radius_floor,
    )

    graph = get_overlaps_graph_numpy(centers, radii) if len(centers) else None
    if graph is not None:
        edges = np.array(graph.get_edgelist(), dtype=np.int64) if graph.ecount() else np.zeros((0, 2), dtype=np.int64)
        components = graph.connected_components()
        component_count = len(components)
        largest_component = max(components.sizes()) if len(components) else 0
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        component_count = 0
        largest_component = 0

    component_ids, component_supports, major_component_mask, major_component_ids, major_component_supports, major_component_sizes = compute_component_supports(
        centers,
        radii,
        validation_points if validation_points is not None else np.zeros((0, 2), dtype=np.float32),
        validation_free_mask if validation_free_mask is not None else np.zeros((0,), dtype=bool),
        major_component_min_bubbles=major_component_min_bubbles,
        major_component_min_support=major_component_min_support,
    )

    if validation_points is None:
        validation_points = np.zeros((0, 2), dtype=np.float32)
    if validation_free_mask is None:
        validation_free_mask = np.zeros((0,), dtype=bool)

    validation_covered_mask = np.zeros((len(validation_points),), dtype=bool)
    if len(validation_points) and len(centers):
        d = np.linalg.norm(validation_points[:, None, :] - centers[None, :, :], axis=-1) - radii[None, :]
        validation_covered_mask = np.min(d, axis=1) <= 0.0

    validation_free_samples = int(validation_free_mask.sum())
    validation_covered_free_samples = int(np.logical_and(validation_free_mask, validation_covered_mask).sum())
    estimated_free_coverage = (
        validation_covered_free_samples / validation_free_samples if validation_free_samples > 0 else 0.0
    )
    estimated_uncovered_free_fraction = 1.0 - estimated_free_coverage if validation_free_samples > 0 else 1.0

    if len(radii):
        min_radius_val = float(np.min(radii))
        max_radius_val = float(np.max(radii))
        mean_radius_val = float(np.mean(radii))
        median_radius_val = float(np.median(radii))
    else:
        min_radius_val = max_radius_val = mean_radius_val = median_radius_val = 0.0

    diagnostics = CoverageDiagnostics(
        raw_bubbles=int(len(raw_centers)),
        raw_edges=int(len(raw_edges)),
        raw_connected_components=int(raw_component_count),
        raw_largest_component=int(raw_largest_component),
        pruned_bubbles=int(len(centers)),
        pruned_edges=int(len(edges)),
        pruned_connected_components=int(component_count),
        pruned_largest_component=int(largest_component),
        sobol_samples=int(len(sobol_points)),
        free_samples=int(len(free_points)),
        bubbles=int(len(centers)),
        edges=int(len(edges)),
        connected_components=int(component_count),
        largest_component=int(largest_component),
        min_radius=min_radius_val,
        max_radius=max_radius_val,
        mean_radius=mean_radius_val,
        median_radius=median_radius_val,
        validation_samples=int(len(validation_points)),
        validation_free_samples=validation_free_samples,
        validation_covered_free_samples=validation_covered_free_samples,
        estimated_free_coverage=float(estimated_free_coverage),
        estimated_uncovered_free_fraction=float(estimated_uncovered_free_fraction),
        batches_processed=int(batches_processed),
        accepted_new_bubbles=int(accepted_new_bubbles),
        radius_prune_threshold=float(radius_prune_threshold),
        major_components=major_component_ids,
        major_component_supports=major_component_supports,
        major_component_sizes=major_component_sizes,
    )

    result = BubbleGraphResult(
        q_min=np.asarray(q_min, dtype=np.float32),
        q_max=np.asarray(q_max, dtype=np.float32),
        centers=centers.astype(np.float32),
        radii=radii.astype(np.float32),
        clearances=clearances_kept.astype(np.float32),
        sobol_indices=sobol_indices.astype(np.int64),
        edges=edges.astype(np.int64),
        component_ids=component_ids.astype(np.int64),
        component_supports=component_supports.astype(np.float32),
        major_component_mask=major_component_mask.astype(bool),
        diagnostics=diagnostics,
        obstacles=np.asarray(obstacle_points.detach().cpu().numpy(), dtype=np.float32),
        validation_points=np.asarray(validation_points, dtype=np.float32),
        validation_free_mask=np.asarray(validation_free_mask, dtype=bool),
        validation_covered_mask=np.asarray(validation_covered_mask, dtype=bool),
    )

    return result, selected_bubbles


def estimate_coverage(
    robot_cdf: RobotCDF,
    bubbles: List[BubbleRecord],
    q_min: np.ndarray,
    q_max: np.ndarray,
    obstacle_points: torch.Tensor,
    num_samples: int,
    validation_points: Optional[np.ndarray],
    validation_free_mask: Optional[np.ndarray],
    batch_size: int,
    safety_margin: float,
    epsilon: float,
    min_radius: float,
) -> CoverageDiagnostics:
    if validation_points is None or len(validation_points) == 0:
        validation_points = sobol_draw(num_samples or 2048, q_min, q_max, seed=12345)
        validation_clearances = batch_query_clearance(
            robot_cdf=robot_cdf,
            configs=validation_points,
            obstacle_points=obstacle_points,
            batch_size=batch_size,
            safety_margin=safety_margin,
        )
        validation_free_mask = validation_clearances > 0.0
    else:
        validation_points = np.asarray(validation_points, dtype=np.float32)
        if validation_free_mask is None:
            validation_clearances = batch_query_clearance(
                robot_cdf=robot_cdf,
                configs=validation_points,
                obstacle_points=obstacle_points,
                batch_size=batch_size,
                safety_margin=safety_margin,
            )
            validation_free_mask = validation_clearances > 0.0
        else:
            validation_free_mask = np.asarray(validation_free_mask, dtype=bool)

    if len(bubbles):
        centers = np.stack([b.center for b in bubbles], axis=0)
        radii = np.array([b.radius for b in bubbles], dtype=np.float32)
        d = np.linalg.norm(validation_points[:, None, :] - centers[None, :, :], axis=-1) - radii[None, :]
        validation_covered_mask = np.min(d, axis=1) <= 0.0
    else:
        validation_covered_mask = np.zeros((len(validation_points),), dtype=bool)

    validation_free_samples = int(np.asarray(validation_free_mask).sum())
    validation_covered_free_samples = int(np.logical_and(validation_free_mask, validation_covered_mask).sum())
    estimated_free_coverage = (
        validation_covered_free_samples / validation_free_samples if validation_free_samples > 0 else 0.0
    )
    estimated_uncovered_free_fraction = 1.0 - estimated_free_coverage if validation_free_samples > 0 else 1.0

    if len(bubbles):
        radii = np.array([b.radius for b in bubbles], dtype=np.float32)
        min_radius_val = float(np.min(radii))
        max_radius_val = float(np.max(radii))
        mean_radius_val = float(np.mean(radii))
        median_radius_val = float(np.median(radii))
    else:
        min_radius_val = max_radius_val = mean_radius_val = median_radius_val = 0.0

    return CoverageDiagnostics(
        raw_bubbles=0,
        raw_edges=0,
        raw_connected_components=0,
        raw_largest_component=0,
        pruned_bubbles=int(len(bubbles)),
        pruned_edges=0,
        pruned_connected_components=0,
        pruned_largest_component=0,
        sobol_samples=0,
        free_samples=0,
        bubbles=int(len(bubbles)),
        edges=0,
        connected_components=0,
        largest_component=0,
        min_radius=min_radius_val,
        max_radius=max_radius_val,
        mean_radius=mean_radius_val,
        median_radius=median_radius_val,
        validation_samples=int(len(validation_points)),
        validation_free_samples=validation_free_samples,
        validation_covered_free_samples=validation_covered_free_samples,
        estimated_free_coverage=float(estimated_free_coverage),
        estimated_uncovered_free_fraction=float(estimated_uncovered_free_fraction),
        batches_processed=0,
        accepted_new_bubbles=0,
        radius_prune_threshold=0.0,
        major_components=[],
        major_component_supports=[],
        major_component_sizes=[],
    )


def save_result(result: BubbleGraphResult, output_dir: Path, save_name: str) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{save_name}.npz"
    json_path = output_dir / f"{save_name}.json"

    np.savez_compressed(
        npz_path,
        q_min=result.q_min,
        q_max=result.q_max,
        centers=result.centers,
        radii=result.radii,
        clearances=result.clearances,
        sobol_indices=result.sobol_indices,
        edges=result.edges,
        component_ids=result.component_ids,
        component_supports=result.component_supports,
        major_component_mask=result.major_component_mask,
        obstacles=result.obstacles,
        validation_points=result.validation_points,
        validation_free_mask=result.validation_free_mask,
        validation_covered_mask=result.validation_covered_mask,
    )

    payload = {
        "npz_file": npz_path.name,
        "diagnostics": asdict(result.diagnostics),
        "num_bubbles": int(len(result.centers)),
        "num_edges": int(len(result.edges)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return npz_path, json_path


def save_scene_png(
    obstacles: List[np.ndarray],
    output_dir: Path,
    q_min: np.ndarray,
    q_max: np.ndarray,
    file_name: str = "scene.png"
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / file_name

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.set_xlim(float(q_min[0]), float(q_max[0]))
    ax.set_ylim(float(q_min[1]), float(q_max[1]))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$q_1$")
    ax.set_ylabel(r"$q_2$")
    ax.set_title("Scene in Workspace")

    for obstacle in obstacles:
        if len(obstacle) == 0:
            continue
        ax.fill(obstacle[:, 0], obstacle[:, 1], color="tomato", alpha=0.45)
        ax.plot(obstacle[:, 0], obstacle[:, 1], color="firebrick", linewidth=0.8, alpha=0.8)
        ax.scatter(obstacle[:, 0], obstacle[:, 1], s=2, color="firebrick", alpha=0.25)
    
    
    fig.savefig(scene_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return scene_path


def main() -> None:
    args = parse_args()
    device = setup_device(args.device)

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    #joint limits for 2r
    q_min = np.array([-np.pi, -np.pi], dtype=np.float32)
    q_max = np.array([np.pi, np.pi], dtype=np.float32)

    rng = np.random.default_rng()

    #NCSB call for obstacle in c-space
    obstacles = create_obstacles(num_points=args.obstacle_points, rng=rng)
    obstacle_points = torch.tensor(concatenate_obstacle_list(obstacles), device=device, dtype=torch.float32)

    output_dir = Path(args.output_dir)
    #scene_path = save_scene_png(obstacles, output_dir, q_min, q_max)

    robot_cdf = RobotCDF(device=device, use_dropout=args.use_dropout)

    #sobol sampling across the q-space
    sobol_points = sobol_draw(args.sobol_samples, q_min, q_max, seed=seed)
    #validation set for coverage measurement
    validation_points = sobol_draw(args.validation_samples, q_min, q_max, seed=seed + 999)
    validation_clearances = batch_query_clearance(
        robot_cdf=robot_cdf,
        configs=validation_points,
        obstacle_points=obstacle_points,
        batch_size=args.validate_batch_size,
        safety_margin=args.safety_margin,
    )
    #sampled points in free space
    validation_free_mask = validation_clearances > args.free_threshold

    result, bubbles = build_bubble_union(
        robot_cdf=robot_cdf,
        sobol_points=sobol_points,
        obstacle_points=obstacle_points,
        q_min=q_min,
        q_max=q_max,
        batch_size=args.batch_size,
        selection_per_batch=args.selection_per_batch,
        epsilon=args.epsilon,
        min_radius=args.min_radius,
        safety_margin=args.safety_margin,
        free_threshold=args.free_threshold,
        coverage_tol=args.coverage_tol,
        patience=args.patience,
        prune_radius_percentile=args.prune_radius_percentile,
        prune_radius_floor=args.prune_radius_floor,
        major_component_min_bubbles=args.major_component_min_bubbles,
        major_component_min_support=args.major_component_min_support,
        validation_points=validation_points,
        validation_free_mask=validation_free_mask,
    )
    print(output_dir)
    npz_path, json_path = save_result(result, output_dir, args.save_name)
    visualize_workspace(obstacles=obstacles,initial_config=None,save=True)
    visualize_cdf_field(robot_cdf=robot_cdf,obstacle_points=obstacles)
    print("\nFull bubble graph construction complete.")
    #print(f"Saved scene: {scene_path}")
    print(f"Saved result: {npz_path}")
    print(f"Saved diagnostics: {json_path}")
    print(f"Sobol samples: {result.diagnostics.sobol_samples}")
    print(f"Free samples: {result.diagnostics.free_samples}")
    print(f"Raw bubbles: {result.diagnostics.raw_bubbles}")
    print(f"Raw components: {result.diagnostics.raw_connected_components}")
    print(f"Pruned bubbles: {result.diagnostics.pruned_bubbles}")
    print(f"Pruned components: {result.diagnostics.pruned_connected_components}")
    print(f"Major components: {result.diagnostics.major_components}")
    print(f"Largest pruned component: {result.diagnostics.pruned_largest_component}")
    print(f"Radius stats: min={result.diagnostics.min_radius:.4f}, "
          f"mean={result.diagnostics.mean_radius:.4f}, median={result.diagnostics.median_radius:.4f}, "
          f"max={result.diagnostics.max_radius:.4f}")
    print(f"Radius prune threshold: {result.diagnostics.radius_prune_threshold:.4f}")
    print(f"Estimated free coverage: {result.diagnostics.estimated_free_coverage:.4f}")
    print(f"Estimated uncovered free fraction: {result.diagnostics.estimated_uncovered_free_fraction:.4f}")


if __name__ == "__main__":
    main()


""" python 2Dexamples/bubble_graph/visualize_graph.py --input 2Dexamples/bubble_graph/outputs/full_bubble_graph.json --save_fig 2Dexamples/bubble_graph/generated_graph.png"""