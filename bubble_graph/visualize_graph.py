import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

#what does this do over construct_graph?

examples_root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(examples_root))
sys.path.append(str(repo_root))

from sdf_marching.overlap import get_overlaps_graph_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a full q-space bubble graph.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "full_bubble_graph.json"),
        help="Path to the saved .json or .npz result file",
    )
    parser.add_argument(
        "--save_fig",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "full_bubble_graph.png"),
        help="Optional path to save the figure",
    )
    parser.add_argument("--show_validation", action="store_true", default=True)
    parser.add_argument("--no_show_validation", dest="show_validation", action="store_false")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_result(input_path: str) -> Tuple[Dict, Path]:
    path = Path(input_path).expanduser().resolve()
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        payload = {key: data[key] for key in data.files}
        return payload, path

    if path.suffix != ".json":
        raise ValueError("Input must be a .json or .npz file")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    npz_path = path.parent / payload["npz_file"]
    data = np.load(npz_path, allow_pickle=True)
    for key in data.files:
        payload[key] = data[key]

    return payload, path


def component_colors(num_components: int):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(num_components)]


def plot_bubble_graph(payload: Dict, dpi: int = 200, show_validation: bool = True):
    centers = np.asarray(payload["centers"], dtype=np.float32)
    radii = np.asarray(payload["radii"], dtype=np.float32)
    edges = np.asarray(payload["edges"], dtype=np.int64)
    q_min = np.asarray(payload["q_min"], dtype=np.float32)
    q_max = np.asarray(payload["q_max"], dtype=np.float32)
    component_ids = np.asarray(payload.get("component_ids", []), dtype=np.int64)
    component_supports = np.asarray(payload.get("component_supports", []), dtype=np.float32)
    major_component_mask = np.asarray(payload.get("major_component_mask", []), dtype=bool)

    graph = get_overlaps_graph_numpy(centers, radii) if len(centers) else None
    if graph is not None and len(graph.vs) > 0:
        if len(component_ids) != len(centers):
            components = graph.connected_components()
            membership = np.array(components.membership)
        else:
            membership = component_ids
        num_components = int(membership.max() + 1) if len(membership) else 0
    else:
        membership = np.zeros((len(centers),), dtype=int)
        num_components = 0

    colors = component_colors(max(num_components, 1))

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_xlim(q_min[0], q_max[0])
    ax.set_ylim(q_min[1], q_max[1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$q_1$")
    ax.set_ylabel(r"$q_2$")

    if show_validation and "validation_points" in payload and len(payload["validation_points"]):
        validation_points = np.asarray(payload["validation_points"], dtype=np.float32)
        validation_free_mask = np.asarray(payload["validation_free_mask"], dtype=bool)
        validation_covered_mask = np.asarray(payload["validation_covered_mask"], dtype=bool)

        if validation_free_mask.any():
            free_pts = validation_points[validation_free_mask]
            ax.scatter(
                free_pts[:, 0],
                free_pts[:, 1],
                s=4,
                color="lightgray",
                alpha=0.35,
                label="validation free",
            )
        if validation_free_mask.any():
            uncovered_free = validation_points[np.logical_and(validation_free_mask, ~validation_covered_mask)]
            if len(uncovered_free):
                ax.scatter(
                    uncovered_free[:, 0],
                    uncovered_free[:, 1],
                    s=10,
                    color="crimson",
                    alpha=0.6,
                    label="uncovered free",
                )

    if len(edges):
        for i, j in edges:
            ax.plot(
                [centers[i, 0], centers[j, 0]],
                [centers[i, 1], centers[j, 1]],
                color="0.55",
                linewidth=0.8,
                alpha=0.5,
                zorder=1,
            )

    theta = np.linspace(0, 2 * np.pi, 240)
    for idx, (center, radius) in enumerate(zip(centers, radii)):
        if len(major_component_mask) == len(centers) and major_component_mask[idx]:
            color = colors[membership[idx] % len(colors)] if len(colors) else "C0"
            alpha = 0.9
            lw = 1.8
        else:
            color = "0.75"
            alpha = 0.35
            lw = 1.0
        circle_x = center[0] + radius * np.cos(theta)
        circle_y = center[1] + radius * np.sin(theta)
        ax.plot(circle_x, circle_y, color=color, linewidth=lw, alpha=alpha, zorder=2)
        ax.scatter(center[0], center[1], s=16, color=color, alpha=alpha, zorder=3)

    diag = payload.get("diagnostics", {})
    title = "Full Bubble Graph"
    if diag:
        title += (
            f" | raw={diag.get('raw_bubbles', len(centers))}"
            f", pruned={diag.get('pruned_bubbles', len(centers))}"
            f", major={len(diag.get('major_components', []))}"
            f", components={diag.get('pruned_connected_components', diag.get('connected_components', num_components))}"
            f", free coverage={diag.get('estimated_free_coverage', 0.0):.3f}"
        )
    ax.set_title(title)

    handles = []
    labels = []
    if len(diag.get("major_components", [])):
        handles.append(plt.Line2D([0], [0], color="0.75", lw=2, alpha=0.35))
        labels.append("minor component")
        for comp_id in diag.get("major_components", []):
            handles.append(plt.Line2D([0], [0], color=colors[comp_id % len(colors)], lw=2))
            support = 0.0
            if len(component_supports) > comp_id:
                support = float(component_supports[comp_id])
            labels.append(f"major {comp_id} ({support:.2f})")
    elif num_components > 1:
        for comp_idx in range(num_components):
            handles.append(plt.Line2D([0], [0], color=colors[comp_idx], lw=2))
            labels.append(f"component {comp_idx}")

    if show_validation and "validation_points" in payload and len(payload["validation_points"]):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=6))
        labels.append("uncovered free")

    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=9)

    return fig, ax


def main() -> None:
    args = parse_args()
    payload, path = load_result(args.input)

    fig, ax = plot_bubble_graph(payload, dpi=args.dpi, show_validation=args.show_validation)

    diag = payload.get("diagnostics", {})
    if diag:
        print("Bubble graph diagnostics")
        for key in [
            "raw_bubbles",
            "raw_edges",
            "raw_connected_components",
            "raw_largest_component",
            "pruned_bubbles",
            "pruned_edges",
            "pruned_connected_components",
            "pruned_largest_component",
            "sobol_samples",
            "free_samples",
            "bubbles",
            "edges",
            "connected_components",
            "largest_component",
            "min_radius",
            "mean_radius",
            "median_radius",
            "max_radius",
            "radius_prune_threshold",
            "estimated_free_coverage",
            "estimated_uncovered_free_fraction",
        ]:
            if key in diag:
                print(f"{key}: {diag[key]}")
        if "major_components" in diag:
            print(f"major_components: {diag['major_components']}")
        if "major_component_supports" in diag:
            print(f"major_component_supports: {diag['major_component_supports']}")
        if "major_component_sizes" in diag:
            print(f"major_component_sizes: {diag['major_component_sizes']}")

    if args.save_fig:
        save_path = Path(args.save_fig).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()
    

if __name__ == "__main__":
    main()
