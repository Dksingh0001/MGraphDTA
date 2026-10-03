"""Render a molecular-style layout of the first 50 AKT1 residues."""

from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = PROJECT_ROOT / "results/davis/akt1_2d_graph.pt"
OUTPUT_PATH = PROJECT_ROOT / "results/davis/akt1_molecular_style_2d_first50.png"
NUM_RESIDUES = 50
LAYOUT_SEED = 42

CATEGORY_RESIDUES = {
    "Hydrophobic": set("AVILMFWY"),
    "Polar": set("STNQ"),
    "Positively charged": set("KRH"),
    "Negatively charged": set("DE"),
}
CATEGORY_COLORS = {
    "Hydrophobic": "#168C83",
    "Polar": "#3D79B7",
    "Positively charged": "#D88918",
    "Negatively charged": "#CC554A",
    "Special/other": "#8172A8",
}


def residue_category(residue: str) -> str:
    for category, residues in CATEGORY_RESIDUES.items():
        if residue in residues:
            return category
    return "Special/other"


def main() -> None:
    graph_data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=True)
    residues = graph_data["residues"]
    residue_numbers = graph_data["residue_numbers"]
    edge_index = graph_data["edge_index"]

    node_count = min(NUM_RESIDUES, len(residues))
    if node_count != NUM_RESIDUES:
        raise ValueError(
            f"Expected at least {NUM_RESIDUES} residues, found {len(residues)}"
        )

    residue_graph = nx.Graph()
    for node_index in range(node_count):
        residue_graph.add_node(
            node_index,
            residue=residues[node_index],
            residue_number=residue_numbers[node_index],
            category=residue_category(residues[node_index]),
        )

    for source, target in edge_index.t().tolist():
        if source < node_count and target < node_count and source != target:
            residue_graph.add_edge(source, target)

    if residue_graph.number_of_nodes() != node_count:
        raise RuntimeError("Displayed graph does not contain all first-50 residue nodes")

    circular_positions = nx.circular_layout(residue_graph, scale=4.8)
    rotation = random.Random(LAYOUT_SEED).uniform(0.0, 2.0 * 3.141592653589793)
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    positions = {
        node_index: (
            circular_positions[node_index][0] * cosine
            - circular_positions[node_index][1] * sine,
            circular_positions[node_index][0] * sine
            + circular_positions[node_index][1] * cosine,
        )
        for node_index in residue_graph.nodes
    }
    labels = {
        node_index: f"{residue_numbers[node_index]}\n{residues[node_index]}"
        for node_index in residue_graph.nodes
    }
    node_colors = [
        CATEGORY_COLORS[residue_graph.nodes[node_index]["category"]]
        for node_index in residue_graph.nodes
    ]

    figure, axis = plt.subplots(figsize=(14, 12), facecolor="#F7F9FA")
    axis.set_facecolor("#F7F9FA")
    nx.draw_networkx_edges(
        residue_graph,
        positions,
        ax=axis,
        edge_color="#75838C",
        width=1.5,
        alpha=0.62,
    )
    nx.draw_networkx_nodes(
        residue_graph,
        positions,
        ax=axis,
        node_color=node_colors,
        node_size=1120,
        edgecolors="white",
        linewidths=1.7,
    )
    nx.draw_networkx_labels(
        residue_graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=8.2,
        font_weight="bold",
        font_color="#14232B",
        verticalalignment="center",
    )

    figure.suptitle(
        "AKT1 Molecular-Style 2D Residue Graph — First 50 Residues",
        fontsize=17,
        fontweight="bold",
        color="#182A38",
        y=0.975,
    )
    axis.set_title(
        "Sequence-neighbor graph; layout is for visualization only",
        fontsize=10.5,
        color="#53636D",
        pad=12,
    )
    axis.set_axis_off()
    axis.set_aspect("equal")

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#D9E4E8",
            markeredgecolor="#70828B",
            markersize=9,
            label="Node = amino-acid residue",
        ),
        Line2D(
            [0], [0],
            color="#75838C",
            linewidth=1.8,
            label="Edge = sequence-neighbor connection",
        ),
    ]
    legend_handles.extend(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=9,
            label=category,
        )
        for category, color in CATEGORY_COLORS.items()
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    figure.tight_layout(rect=(0.025, 0.08, 0.975, 0.94))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    print(f"Number of nodes: {residue_graph.number_of_nodes()}")
    print(f"Number of unique displayed edges: {residue_graph.number_of_edges()}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
