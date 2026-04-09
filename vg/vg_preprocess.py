"""
vg_preprocess.py
----------------
Converts Visual Genome JSON data into a generalisation split for the GNN.

Pipeline
--------
1. Stream vg/relationships.json → vg_relationships.jsonl  (one image per line)
2. Stream vg/objects.json       → vg_objects.jsonl
3. For every image in vg_relationships.jsonl:
     - Map VG synsets → Hoiverse category IDs via canonicalize()
     - Build node annotations and directed edges
     - Skip images with fewer than 2 valid nodes or 0 valid edges
4. Compute normalisation statistics (mean / std) over all processed graphs
5. Save a pickle containing graphs, metadata, norm_stats, and image IDs

Output: vg_processed  (pickle file, loaded by evaluate_vg.py)
"""

import ijson
import json
import os
import pickle
import sys

# Ensure the project root is on the path so root-level modules are importable
_VG_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_VG_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np
from tqdm import tqdm

from mappings import VG_TO_HOI
from modules.vg_image_data import get_image_size
from preprocess import (
    USEFUL_RELATIONS,
    compute_feature_stats,
    normalize_bbox,
    bbox_center,
    bbox_area,
    bbox_iou,
    one_hot,
    is_human,
    bbox_aspect_ratio,
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

VG_RELATIONSHIPS_JSON = os.path.join(_VG_DIR, "data", "relationships.json")
VG_OBJECTS_JSON       = os.path.join(_VG_DIR, "data", "objects.json")

RELATIONSHIPS_JSONL   = os.path.join(_VG_DIR, "data", "vg_relationships.jsonl")
OBJECTS_JSONL         = os.path.join(_VG_DIR, "data", "vg_objects.jsonl")

OUTPUT_FILE  = os.path.join(_VG_DIR, "processed_data", "vg_processed")

# Set to None to process all images; otherwise limit for quick testing.
MAX_IMAGES = None

# ──────────────────────────────────────────────
# VOCABULARY  (kept in sync with vg_helpers.py)
# ──────────────────────────────────────────────

NODE_NAMES = [
    'Can', 'FoodBag', 'FoodBox', 'FruitContainer', 'GrassTuft', 'Jar',
    'KitchenSpace', 'Tap', 'UNKNOWN', 'art', 'balloon', 'bathtub', 'bed',
    'blanket', 'book', 'bottle', 'bowl', 'can', 'ceiling', 'chair', 'cup',
    'cutlery', 'dishwasher', 'door', 'exterior', 'floor', 'food', 'fridge',
    'fruit', 'gnome', 'hardware', 'human', 'light', 'mattress', 'meshed',
    'microwave', 'mirror', 'oven', 'pan', 'pillar', 'pillow', 'plant',
    'plate', 'pot', 'rock', 'rug', 'screen', 'shelf', 'sink', 'skirtingboard',
    'sofa', 'table', 'toilet', 'towel', 'trinket', 'vase', 'wall', 'watch',
    'window', 'wineglass',
]

VOCAB          = {name: i for i, name in enumerate(NODE_NAMES)}
NUM_CATEGORIES = len(NODE_NAMES)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# VG-SPECIFIC FEATURE FUNCTIONS
# (dimension-aware copies; preprocess.py is left unchanged)
# ──────────────────────────────────────────────

def vg_bbox_to_features(bbox, img_w, img_h):
    """[cx_norm, cy_norm, w_norm, h_norm] normalised to actual image size."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    return np.array([cx / img_w, cy / img_h, w / img_w, h / img_h])


def vg_node_features(ann, num_categories, img_w, img_h):
    """Node feature vector using per-image dimensions."""
    cat_id = ann["category_id"]
    bbox   = normalize_bbox(ann["bbox"])

    onehot_cat = one_hot(cat_id, num_categories)
    bbox_feat  = vg_bbox_to_features(bbox, img_w, img_h)
    area       = bbox_area(bbox) / (img_w * img_h)
    aspect     = bbox_aspect_ratio(bbox)
    human_flag = is_human(cat_id)

    return np.concatenate([onehot_cat, bbox_feat, [area, aspect, human_flag]])


def vg_edge_geometry(node_i, node_j, img_w, img_h):
    """Edge geometry features normalised to per-image dimensions."""
    box_i = normalize_bbox(node_i["bbox"])
    box_j = normalize_bbox(node_j["bbox"])

    cx_i, cy_i = bbox_center(box_i)
    cx_j, cy_j = bbox_center(box_j)

    dx = (cx_j - cx_i) / img_w
    dy = (cy_j - cy_i) / img_h
    dist = np.sqrt(dx**2 + dy**2)

    iou = bbox_iou(box_i, box_j)

    area_i = bbox_area(box_i) + 1e-6
    area_j = bbox_area(box_j) + 1e-6
    size_ratio_ij = np.clip(area_j / area_i, 0.0, 100.0)
    size_ratio_ji = np.clip(area_i / area_j, 0.0, 100.0)

    return np.array([
        dx, dy, dist,
        1.0 if dx < 0 else 0.0,  # is_left
        1.0 if dx > 0 else 0.0,  # is_right
        1.0 if dy < 0 else 0.0,  # is_above
        1.0 if dy > 0 else 0.0,  # is_below
        iou,
        size_ratio_ij, size_ratio_ji,
    ], dtype=float)

def canonicalize(vg_synset: str):
    """Map a VG WordNet synset to a Hoiverse category name, or None if unknown."""
    return VG_TO_HOI.get(vg_synset, None)

# ──────────────────────────────────────────────
# STEP 1 – JSON → JSONL
# ──────────────────────────────────────────────

def convert_to_jsonl():
    """Stream both large VG JSON arrays to line-delimited JSONL files."""
    pairs = [
        (VG_RELATIONSHIPS_JSON, RELATIONSHIPS_JSONL, "relationships"),
        (VG_OBJECTS_JSON,       OBJECTS_JSONL,       "objects"),
    ]
    for src, dst, label in pairs:
        if os.path.exists(dst):
            print(f"  Skipping {dst} (already exists)")
            continue
        print(f"  Converting {src} → {dst} ...")
        with open(src, "r") as f_in, open(dst, "w") as f_out:
            for item in ijson.items(f_in, "item"):
                f_out.write(json.dumps(item) + "\n")
        print(f"    Done.")


# ──────────────────────────────────────────────
# STEP 2 – GRAPH CONSTRUCTION
# ──────────────────────────────────────────────

def process_image(line_dict: dict, img_w: int, img_h: int) -> tuple:
    """
    Build a graph dict from a single VG image entry.

    Returns (graph, image_id) on success, or (None, None) if the image
    does not have enough valid nodes / edges to be useful.
    """
    image_id  = line_dict["image_id"]
    relations = line_dict["relationships"]

    # ── Pass 1: collect unique mappable nodes ──────────────────────────────
    obj_id_to_ann: dict = {}

    for rel in relations:
        for entity in (rel["subject"], rel["object"]):
            eid     = entity["object_id"]
            if eid in obj_id_to_ann:
                continue

            synsets = entity.get("synsets", [])
            if not synsets:
                continue

            cat_name = canonicalize(synsets[0])
            if cat_name is None:
                continue

            cat_id = VOCAB[cat_name]
            obj_id_to_ann[eid] = {
                "category_id": cat_id,
                "bbox":        [entity["x"], entity["y"], entity["w"], entity["h"]],
            }

    if len(obj_id_to_ann) < 2:
        return None, None

    # Assign contiguous node indices in insertion order
    obj_id_to_idx = {oid: idx for idx, oid in enumerate(obj_id_to_ann)}
    nodes         = [obj_id_to_ann[oid] for oid in obj_id_to_ann]

    # ── Pass 2: collect valid edges ────────────────────────────────────────
    edge_list:  list = []
    edge_feats: list = []

    for rel in relations:
        s_id = rel["subject"]["object_id"]
        o_id = rel["object"]["object_id"]
        if s_id not in obj_id_to_idx or o_id not in obj_id_to_idx:
            continue

        i = obj_id_to_idx[s_id]
        j = obj_id_to_idx[o_id]
        feat = vg_edge_geometry(obj_id_to_ann[s_id], obj_id_to_ann[o_id], img_w, img_h)
        edge_list.append([i, j])
        edge_feats.append(feat)

    if not edge_list:
        return None, None

    # ── Build arrays ───────────────────────────────────────────────────────
    node_features = np.array(
        [vg_node_features(n, NUM_CATEGORIES, img_w, img_h) for n in nodes],
        dtype=np.float32,
    )
    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_attr  = np.array(edge_feats, dtype=np.float32)

    graph = {
        "x":          node_features,
        "edge_index": edge_index,
        "edge_attr":  edge_attr,
        "img_w":      img_w,
        "img_h":      img_h,
    }
    return graph, image_id


def build_graphs() -> tuple:
    """Iterate over every image in the JSONL file and build graph dicts."""
    processed_graphs: list = []
    valid_images:     list = []
    skipped = 0

    print(f"Building graphs from {RELATIONSHIPS_JSONL} ...")
    with open(RELATIONSHIPS_JSONL, "r") as f:
        lines = f.readlines() if MAX_IMAGES is None else f.readlines()[:MAX_IMAGES]

    for raw_line in tqdm(lines, desc="Processing images"):
        line_dict = json.loads(raw_line)
        image_id  = line_dict["image_id"]
        img_w, img_h = get_image_size(image_id)

        graph, image_id = process_image(line_dict, img_w, img_h)
        if graph is None:
            skipped += 1
            continue

        processed_graphs.append(graph)
        valid_images.append(image_id)

    print(f"  Built {len(processed_graphs)} graphs  |  Skipped {skipped} images")
    return processed_graphs, valid_images


# ──────────────────────────────────────────────
# STEP 3 – SAVE
# ──────────────────────────────────────────────

def save_output(processed_graphs: list, valid_images: list):
    feature_stats = compute_feature_stats(processed_graphs)

    label_map = {i: name for i, name in enumerate(USEFUL_RELATIONS)}

    output = {
        "generalization_graphs": processed_graphs,
        "meta": {
            "node_feature_dim": processed_graphs[0]["x"].shape[1],
            "edge_feature_dim": processed_graphs[0]["edge_attr"].shape[1],
            "num_classes":      len(USEFUL_RELATIONS),
            "node_names":       NODE_NAMES,
            "label_map":        label_map,
        },
        "norm_stats": feature_stats,
        "image_ids":  valid_images,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved → {OUTPUT_FILE}")
    print(f"  node_feature_dim : {output['meta']['node_feature_dim']}")
    print(f"  edge_feature_dim : {output['meta']['edge_feature_dim']}")
    print(f"  num_classes      : {output['meta']['num_classes']}")
    print(f"  label_map        : {label_map}")
    print(f"  images           : {len(valid_images)}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Convert JSON → JSONL ===")
    convert_to_jsonl()

    print("\n=== Step 2: Build graphs ===")
    graphs, images = build_graphs()

    if not graphs:
        raise RuntimeError("No valid graphs produced — check your VG data path and mappings.")

    print("\n=== Step 4: Save output ===")
    save_output(graphs, images)

    print("\nDone.")
