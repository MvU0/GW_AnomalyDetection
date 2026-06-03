import torch
import networkx as nx
import matplotlib.pyplot as plt


# ===== USER CONFIG =====
glitch_id = 6
graph_and_weights = torch.load(f"./attention_weights/glitch_{glitch_id}.pt") # attention weights + edge indices
labels_file = "WADI_labels.txt"                     # one label per line
use_labels = True                                   # wether to use labels at all
highlight_node_label = "2_P_003_STATUS"            # set to a label string to highlight, or None to disable
show_all_labels = False

# ===== GET WEIGHTS AND SOURCE- AND DESTINATION NODES =====
weights = graph_and_weights[0]['layers'][0]['attention_weights'].squeeze(-1).squeeze(-1)
edges = graph_and_weights[0]['layers'][0]['edge_index']             # Sources are edge[0], destinations are edge[1]

num_nodes = 127
edge_strength = 0.1

# ===== LOAD LABELS =====
if use_labels:
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f]

    if len(labels) != num_nodes:
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of nodes ({num_nodes})")
else:
    labels = [str(i) for i in range(num_nodes)]  # fallback numeric labels as strings

# Map index -> label (used only for display)
index_to_label = {i: labels[i] for i in range(num_nodes)}

# ===== SELECT STRONG EDGES =====
edge_mask = (weights >= 0.1)
strong_weights = weights[edge_mask]
strong_edges = edges[:,edge_mask].T

# ===== BUILD DIRECTED GRAPH ===== 
G = nx.DiGraph()
G.add_nodes_from(range(num_nodes))
for i in range(num_nodes):
    for j in range(strong_edges.size(1)):
        target = int(strong_edges[i, j].item())
        # Optionally skip invalid targets (e.g., -1) if present
        if 0 <= target < num_nodes:
            G.add_edge(i, target)

# ===== HIGHLIGHTING =====  
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
        for pred in G.predecessors(highlight_index):
            node_colors[pred] = "yellow"
            indices_with_labels.add(pred)

# ===== PREPARE LABELS DICT (only labels for indices_with_labels)=====
if show_all_labels:
    labels_to_draw = {i: index_to_label[i] for i in range(num_nodes)}
else:
    labels_to_draw = {i: index_to_label[i] for i in indices_with_labels}

# ===== DRAW GRAPH =====
plt.figure(figsize=(18,12))
pos = nx.spring_layout(G, seed=42, k=0.1, iterations=100)

nx.draw_networkx_nodes(G, pos, node_color = node_colors, node_size = 450)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle = '->', arrowsize=16)

# Draw only the selected labels
if len(labels_to_draw) > 0:
    nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=10)

plt.title("Directed Graph (highlighted node + descendants labeled)")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"./imgs/glitch_{glitch_id}_{highlight_node_label}.png", dpi=300, bbox_inches="tight")
plt.show()