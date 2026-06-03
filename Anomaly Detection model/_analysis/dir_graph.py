# %%
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ========== USER CONFIG ==========
tensor_file = "graph_WADI.pt"    # your torch file
labels_file = "WADI_labels.txt"  # one label per line
use_labels = True                # whether to use labels at all
highlight_node_label = "1_MV_001_STATUS"  # set to a label string to highlight, or None to disable

# ========== LOAD TENSOR ==========
tensor = torch.load(tensor_file)    # expect shape (N, M) where row i lists targets including self maybe
# remove first column (self-loop or whatever) if that's how your tensor is structured
edges = tensor[:, 1:]               # shape (N, M-1)
num_nodes = edges.size(0)

# ========== LOAD LABELS ==========
if use_labels:
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f]

    if len(labels) != num_nodes:
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of nodes ({num_nodes})")
else:
    labels = [str(i) for i in range(num_nodes)]  # fallback numeric labels as strings

# Map index -> label (used only for display)
index_to_label = {i: labels[i] for i in range(num_nodes)}

# ========== BUILD DIRECTED GRAPH (nodes are numeric indices) ==========
G = nx.DiGraph()
G.add_nodes_from(range(num_nodes))
for i in range(num_nodes):
    for j in range(edges.size(1)):
        target = int(edges[i, j].item())
        # Optionally skip invalid targets (e.g., -1) if present
        if 0 <= target < num_nodes:
            G.add_edge(i, target)

# ========== HIGHLIGHTING LOGIC (find the index of highlight_node_label) ==========
# Default node colors
node_colors = ["skyblue"] * num_nodes

# Build a set of node indices that should have their labels displayed
indices_with_labels = set()

if highlight_node_label is not None and use_labels:
    # find index corresponding to the requested label
    highlight_index = None
    for idx, lab in index_to_label.items():
        if lab == highlight_node_label:
            highlight_index = idx
            break

    if highlight_index is None:
        print(f"Warning: highlight label '{highlight_node_label}' not found among labels.")
    else:
        # mark the highlighted node and its direct descendants (successors)
        node_colors[highlight_index] = "red"
        indices_with_labels.add(highlight_index)

        # successors returns indices (because nodes are indices)
        for succ in G.successors(highlight_index):
            node_colors[succ] = "orange"
            indices_with_labels.add(succ)

# If highlighting disabled or not found, indices_with_labels remains empty -> no labels drawn

# ========== PREPARE LABELS DICT (only labels for indices_with_labels) ==========
labels_to_draw = {i: index_to_label[i] for i in indices_with_labels}

# ========== DRAW GRAPH ==========
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42, k=1.0, iterations=100)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=450)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=16)

# Draw only the selected labels
if len(labels_to_draw) > 0:
    nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=10)
else:
    # If you still want numeric indices shown when not using labels, uncomment:
    # nx.draw_networkx_labels(G, pos, labels={i: str(i) for i in G.nodes()}, font_size=9)
    pass

plt.title("Directed Graph (highlighted node + descendants labeled)")
plt.axis("off")
plt.tight_layout()
plt.savefig("_test_force_directed_graph_WADI_selective_labels.png", dpi=300, bbox_inches="tight")
plt.show()
