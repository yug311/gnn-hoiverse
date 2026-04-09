from collections import defaultdict
from io import BytesIO
import os
import pickle
import sys

# Ensure the project root is on the path so root-level modules are importable
_VG_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_VG_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch_geometric.data import Data

from GNN import DEVICE, DROPOUT, HIDDEN_DIM, NUM_LAYERS, GNNEdgeClassifier
from modules.vg_image_data import get_image_url, get_image_size

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

PROCESSED_FILE = os.path.join(_VG_DIR, "processed_data", "vg_processed")
CHECKPOINT     = os.path.join(_ROOT_DIR, "checkpoints", "best_gnn.pt")
OUTPUT_DIR     = os.path.join(_VG_DIR, "visualizations")

NUM_SAMPLES    = 50   # how many images to visualise

IMG_W, IMG_H   = 640, 480  # fallback if image_data lookup fails

# One colour per relation class (extend if you add more classes)
RELATION_COLORS = {
    "sitting":    "#e74c3c",
    "holding":    "#2ecc71",
    "i_on":       "#3498db",
    "i_left":     "#9b59b6",
    "i_right":    "#f39c12",
    "i_above":    "#1abc9c",
    "i_below":    "#e67e22",
    "i_touching": "#c0392b",
}

DEFAULT_COLOR = "#7f8c8d"

# ──────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────

def load_data():
    print(f"Loading data from {PROCESSED_FILE} ...")
    with open(PROCESSED_FILE, "rb") as f:
        return pickle.load(f)

# converts to single pyg -- no batching
def to_pyg_single(graph, norm_stats):
    """Normalise a raw graph dict and return a PyG Data object (no edge_label)."""
    node_mean = norm_stats["node_mean"].astype(np.float32)
    node_std  = norm_stats["node_std"].astype(np.float32)
    edge_mean = norm_stats["edge_mean"].astype(np.float32)
    edge_std  = norm_stats["edge_std"].astype(np.float32)

    x         = (graph["x"].astype(np.float32)        - node_mean) / node_std
    edge_attr = (graph["edge_attr"].astype(np.float32) - edge_mean) / edge_std

    return Data(
        x          = torch.tensor(x,                   dtype=torch.float32),
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr,           dtype=torch.float32),
    )


def recover_nodes(raw_x, num_categories, node_names, img_w, img_h):
    """
    Recover category name and pixel-space bbox (x1,y1,x2,y2) for every node
    from the raw (pre-stats-normalisation) feature matrix.

    Node feature layout (from preprocess.py):
        [0 .. num_categories-1]  one-hot category
        [num_categories+0]       cx / img_w
        [num_categories+1]       cy / img_h
        [num_categories+2]       w  / img_w
        [num_categories+3]       h  / img_h
        [num_categories+4]       area
        [num_categories+5]       aspect ratio
        [num_categories+6]       is_human flag
    """
    nodes = []
    for feat in raw_x:
        cat_id = int(np.argmax(feat[:num_categories]))
        cx     = feat[num_categories]     * img_w
        cy     = feat[num_categories + 1] * img_h
        w      = feat[num_categories + 2] * img_w
        h      = feat[num_categories + 3] * img_h
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        nodes.append({
            "cat_name": node_names[cat_id] if cat_id < len(node_names) else str(cat_id),
            "bbox":     (x1, y1, x2, y2),
            "center":   (cx, cy),
        })
    return nodes


def fetch_vg_image(image_id) -> np.ndarray | None:
    """Fetch the VG image from its URL; return an RGB ndarray or None on failure."""
    url = get_image_url(image_id)
    if url is None:
        return None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return np.array(Image.open(BytesIO(response.content)).convert("RGB"))
    except Exception as e:
        print(f"  Warning: could not fetch image {image_id}: {e}")
        return None

# ──────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────

def filter_edges(edges, edge_labels, max_edges_per_node=3):
    node_edge_count = defaultdict(int)
    filtered_edges, filtered_labels = [], []
    for (src, dst), label in zip(edges, edge_labels):
        if node_edge_count[src] < max_edges_per_node and node_edge_count[dst] < max_edges_per_node:
            filtered_edges.append((src, dst))
            filtered_labels.append(label)
            node_edge_count[src] += 1
            node_edge_count[dst] += 1
    return filtered_edges, filtered_labels


def draw_panel(ax, bg_image, nodes, edges, edge_labels, title, img_w, img_h,
               active_node_indices=None, max_edges_per_node=3):
    """
    Render one panel (before or after) on *ax*.

    Parameters
    ----------
    bg_image           : np.ndarray (H, W, 3) or None
    nodes              : list of dicts from recover_nodes()
    edges              : list of (src_idx, dst_idx)
    edge_labels        : list of str if labelled, else None for topology-only
    title              : panel title string
    active_node_indices: set of node indices to draw bboxes for (None = all)
    """
    if bg_image is not None:
        # Darken background so overlays pop.
        # Do NOT use extent/aspect="auto" — they stretch the image and cause
        # visual misalignment with bbox coordinates.
        darkened = (bg_image.astype(np.float32) * 0.45).astype(np.uint8)
        ax.imshow(darkened)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)   # y=0 at top, y increases downward
    ax.set_facecolor("#1a1a2e")

    # Which nodes to draw bboxes for
    draw_indices = active_node_indices if active_node_indices is not None else range(len(nodes))

    # ── Bounding boxes + node labels
    for ni in draw_indices:
        node = nodes[ni]
        x1, y1, x2, y2 = node["bbox"]
        w_box, h_box = x2 - x1, y2 - y1
        # White glow shadow
        ax.add_patch(patches.Rectangle(
            (x1, y1), w_box, h_box,
            linewidth=5, edgecolor="white", facecolor="none", zorder=3, alpha=0.4,
        ))
        # Main cyan box with faint fill
        ax.add_patch(patches.Rectangle(
            (x1, y1), w_box, h_box,
            linewidth=2, edgecolor="#00e5ff", facecolor=(0, 0.9, 1, 0.08), zorder=4,
        ))
        ax.text(
            x1 + 3, y1 + 13, node["cat_name"],
            fontsize=8, fontweight="bold", color="white", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#00557a",
                      edgecolor="#00e5ff", linewidth=1.2, alpha=0.92),
        )

    # ── Directed edges
    # Cap edges per node to avoid visual clutter
    if edge_labels is not None:
        edges, edge_labels = filter_edges(edges, edge_labels, max_edges_per_node)
    # Track label midpoints already used so we can offset collisions
    used_label_positions: list[tuple[float, float]] = []

    for idx, (src, dst) in enumerate(edges):
        if active_node_indices is not None and (
            src not in active_node_indices or dst not in active_node_indices
        ):
            continue
        cx_s, cy_s = nodes[src]["center"]
        cx_d, cy_d = nodes[dst]["center"]

        label = edge_labels[idx] if edge_labels else None
        color = RELATION_COLORS.get(label, DEFAULT_COLOR) if label else "#ffffff"

        # White shadow arrow for contrast
        ax.annotate(
            "",
            xy=(cx_d, cy_d), xytext=(cx_s, cy_s),
            arrowprops=dict(
                arrowstyle="-|>",
                color="white",
                lw=4.5,
                mutation_scale=20,
                connectionstyle="arc3,rad=0.12",
            ),
            zorder=4,
        )
        # Coloured arrow on top
        ax.annotate(
            "",
            xy=(cx_d, cy_d), xytext=(cx_s, cy_s),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.5,
                mutation_scale=18,
                connectionstyle="arc3,rad=0.12",
            ),
            zorder=5,
        )

        # ── S / O role badges along the arrow
        dx_v = cx_d - cx_s
        dy_v = cy_d - cy_s
        length = max(np.sqrt(dx_v**2 + dy_v**2), 1e-6)
        ux, uy = dx_v / length, dy_v / length
        badge_offset = 14
        ax.text(
            cx_s + ux * badge_offset, cy_s + uy * badge_offset, "S",
            fontsize=6, fontweight="bold", color="black",
            ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor=color,
                      edgecolor="white", linewidth=0.8, alpha=0.9),
        )
        ax.text(
            cx_d - ux * badge_offset, cy_d - uy * badge_offset, "O",
            fontsize=6, fontweight="bold", color="black",
            ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor=color,
                      edgecolor="white", linewidth=0.8, alpha=0.9),
        )

        if label:
            # Find a midpoint that doesn't overlap existing labels
            mx = (cx_s + cx_d) / 2
            my = (cy_s + cy_d) / 2
            # Perp direction for nudging
            px, py = -uy, ux
            nudge_step = 18
            for _ in range(8):
                too_close = any(
                    np.sqrt((mx - ox)**2 + (my - oy)**2) < nudge_step
                    for ox, oy in used_label_positions
                )
                if not too_close:
                    break
                mx += px * nudge_step
                my += py * nudge_step
            used_label_positions.append((mx, my))

            ax.text(
                mx, my, label,
                fontsize=8, fontweight="bold", color="white",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=color,
                          edgecolor="white", linewidth=1.2, alpha=0.95),
            )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="white")
    ax.axis("off")

# ──────────────────────────────────────────────
# MAIN VISUALISATION LOOP
# ──────────────────────────────────────────────

def visualize_predictions(model, graphs, image_ids, norm_stats, meta,
                           num_samples=NUM_SAMPLES):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_categories = len(meta["node_names"])
    node_names     = meta["node_names"]
    label_map      = meta["label_map"]

    model.eval()
    with torch.no_grad():
        for i, (graph, img_id) in enumerate(
            zip(graphs[:num_samples], image_ids[:num_samples])
        ):
            # ── Per-graph image dimensions (from vg_image_data for accuracy)
            img_w, img_h = get_image_size(img_id)

            # ── Run GNN
            pyg    = to_pyg_single(graph, norm_stats).to(DEVICE)
            logits = model(pyg)
            preds  = logits.argmax(dim=-1).cpu().numpy()

            pred_labels = [label_map[int(p)] for p in preds]

            # ── Recover human-readable node / edge info
            nodes = recover_nodes(graph["x"], num_categories, node_names, img_w, img_h)
            edges = [
                (int(graph["edge_index"][0, e]), int(graph["edge_index"][1, e]))
                for e in range(graph["edge_index"].shape[1])
            ]

            # ── Background image fetched from URL at native resolution
            bg_raw = fetch_vg_image(img_id)
            bg = np.array(Image.fromarray(bg_raw).resize((img_w, img_h))) if bg_raw is not None else None

            # ── Figure: before (left) | after (right)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=110)
            fig.suptitle(
                f"VG Image ID: {img_id}  |  {len(nodes)} nodes  {len(edges)} edges",
                fontsize=10, y=1.01,
            )

            # Nodes that participate in at least one edge (for after panel)
            edge_node_indices = set(
                idx
                for src, dst in edges
                for idx in (src, dst)
            )

            draw_panel(axes[0], bg, nodes, edges, edge_labels=None,
                       title="Before — topology only (no relation labels)",
                       img_w=img_w, img_h=img_h)
            draw_panel(axes[1], bg, nodes, edges, edge_labels=pred_labels,
                       title="After — GNN predicted relations",
                       img_w=img_w, img_h=img_h,
                       active_node_indices=edge_node_indices)

            # ── Legend for relation colours
            legend_handles = [
                plt.Line2D([0], [0], color=c, lw=2, label=rel)
                for rel, c in RELATION_COLORS.items()
            ]
            fig.legend(handles=legend_handles, loc="lower center",
                       ncol=len(RELATION_COLORS), fontsize=7,
                       frameon=True, bbox_to_anchor=(0.5, -0.04))

            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, f"{img_id}.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            print(f"[{i + 1}/{num_samples}] image_id={img_id}  "
                  f"predicted: {pred_labels}  → {out_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

data_processed = load_data()

norm_stats = data_processed["norm_stats"]
graphs     = data_processed["generalization_graphs"]
meta       = data_processed["meta"]
image_ids  = data_processed["image_ids"]

node_dim    = meta["node_feature_dim"]
edge_dim    = meta["edge_feature_dim"]
num_classes = meta["num_classes"]

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

gnn = GNNEdgeClassifier(
    node_dim    = node_dim,
    edge_dim    = edge_dim,
    hidden_dim  = HIDDEN_DIM,
    num_classes = num_classes,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT,
).to(DEVICE)

# load best gnn model
gnn.load_state_dict(checkpoint)
print("Checkpoint loaded.")

# visualize predictions
visualize_predictions(gnn, graphs, image_ids, norm_stats, meta, num_samples=NUM_SAMPLES)
print(f"\nDone. Figures saved to '{OUTPUT_DIR}/'")
