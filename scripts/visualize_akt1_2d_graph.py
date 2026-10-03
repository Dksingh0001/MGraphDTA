"""Visualize the first 50 residues of the standalone AKT1 residue graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = PROJECT_ROOT / "results/davis/akt1_2d_graph.pt"
OUTPUT_PATH = PROJECT_ROOT / "results/davis/akt1_2d_graph_first50.png"
NUM_RESIDUES = 50
COLUMNS = 10


def main() -> None:
    graph_data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=True)
    residues = graph_data["residues"]
    residue_numbers = graph_data["residue_numbers"]
    edge_index = graph_data["edge_index"]
    amino_acids = graph_data["amino_acid_feature_order"]

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
        )

    for source, target in edge_index.t().tolist():
        if source < node_count and target < node_count and source != target:
            residue_graph.add_edge(source, target)

    positions = {}
    for node_index in residue_graph.nodes:
        row, column = divmod(node_index, COLUMNS)
        display_column = column if row % 2 == 0 else COLUMNS - 1 - column
        positions[node_index] = (display_column * 1.65, -row * 1.6)

    color_map = ListedColormap(plt.get_cmap("tab20").colors[:len(amino_acids)])
    amino_acid_colors = {
        amino_acid: color_map(index)
        for index, amino_acid in enumerate(amino_acids)
    }
    node_colors = [
        amino_acid_colors.get(residues[node_index], "#B8C0C8")
        for node_index in residue_graph.nodes
    ]
    labels = {
        node_index: f"{residue_numbers[node_index]}\n{residues[node_index]}"
        for node_index in residue_graph.nodes
    }

    figure, axis = plt.subplots(figsize=(16, 9), facecolor="white")
    axis.set_facecolor("white")
    nx.draw_networkx_edges(
        residue_graph,
        positions,
        ax=axis,
        edge_color="#8493A0",
        width=1.7,
    )
    nx.draw_networkx_nodes(
        residue_graph,
        positions,
        ax=axis,
        node_color=node_colors,
        node_size=1120,
        edgecolors="white",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        residue_graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=8,
        font_weight="bold",
        font_color="#14212B",
        verticalalignment="center",
    )

    axis.set_title(
        "AKT1 2D Residue Graph — First 50 Residues",
        fontsize=18,
        fontweight="bold",
        color="#182A38",
        pad=20,
    )
    axis.set_axis_off()
    axis.set_aspect("equal")

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#D9E8EF",
            markeredgecolor="#8096A3",
            markersize=11,
            label="Node = amino acid residue",
        ),
        Line2D(
            [0], [0],
            color="#8493A0",
            linewidth=2,
            label="Edge = sequence-neighbor connection",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Number of nodes visualized: {residue_graph.number_of_nodes()}")
    print(f"Number of edges visualized: {residue_graph.number_of_edges()}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
