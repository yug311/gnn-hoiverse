import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
import os

INPUT_FILE = "hoiverse.pkl"
OUTPUT_FILE = "data/processed.pkl"

SEMANTIC_RELATIONS = [
    "sitting",
    "holding",
]

USEFUL_RELATIONS = [
    "sitting",
    "holding",
    "i_on",
    "i_left",
    "i_right",
    "i_above",
    "i_below",
    "i_touching",
]

IMG_W = 640
IMG_H = 480

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remainder (0.15)

BBOX_FORMAT = "xywh"

HUMAN_CATEGORY_ID = 31

def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec

def normalize_bbox(bbox):
    """
    Converts bbox to [x1, y1, x2, y2] regardless of input format.
    BBOX_FORMAT should be set to "xyxy" or "xywh" in CONFIG above.
    """
    if BBOX_FORMAT == "xywh":
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    return list(bbox)  # already xyxy

def bbox_to_features(bbox):
    """Returns [cx_norm, cy_norm, w_norm, h_norm] from an xyxy bbox."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return np.array([
        cx / IMG_W,
        cy / IMG_H,
        w / IMG_W,
        h / IMG_H,
    ])

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)

def bbox_aspect_ratio(bbox):
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return w / h

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy

def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_w = max(xi2 - xi1, 0)
    inter_h = max(yi2 - yi1, 0)
    inter_area = inter_w * inter_h

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union

def is_human(category_id):
    return 1.0 if category_id == HUMAN_CATEGORY_ID else 0.0

def node_features_from_annotation(ann, num_categories):
    cat_id = ann["category_id"]
    bbox = normalize_bbox(ann["bbox"])

    onehot_cat  = one_hot(cat_id, num_categories)
    bbox_feat   = bbox_to_features(bbox)
    area        = bbox_area(bbox) / (IMG_W * IMG_H)
    aspect      = bbox_aspect_ratio(bbox)
    human_flag  = is_human(cat_id)

    return np.concatenate([
        onehot_cat,
        bbox_feat,
        [area, aspect, human_flag],
    ])

def edge_geometry(node_i, node_j):
    box_i = normalize_bbox(node_i["bbox"])
    box_j = normalize_bbox(node_j["bbox"])

    cx_i, cy_i = bbox_center(box_i)
    cx_j, cy_j = bbox_center(box_j)

    dx = (cx_j - cx_i) / IMG_W
    dy = (cy_j - cy_i) / IMG_H
    dist = np.sqrt(dx**2 + dy**2)

    is_left  = 1.0 if dx < 0 else 0.0
    is_right = 1.0 if dx > 0 else 0.0
    is_above = 1.0 if dy < 0 else 0.0
    is_below = 1.0 if dy > 0 else 0.0

    iou = bbox_iou(box_i, box_j)

    area_i = bbox_area(box_i) + 1e-6
    area_j = bbox_area(box_j) + 1e-6
    # Clipped to prevent extreme outliers from dominating
    size_ratio_ij = np.clip(area_j / area_i, 0.0, 100.0)
    size_ratio_ji = np.clip(area_i / area_j, 0.0, 100.0)

    return np.array([
        dx, dy, dist,
        is_left, is_right, is_above, is_below,
        iou,
        size_ratio_ij, size_ratio_ji,
    ], dtype=float)

def build_edge_features(edge_list, nodes):
    return np.array([
        edge_geometry(nodes[s], nodes[t])
        for s, t in edge_list
    ], dtype=float)

def compute_feature_stats(graphs):
    """
    Computes mean and std over all node and edge features across the
    training split so they can be saved and reused at inference time.
    Only the continuous suffix of each feature vector is standardized;
    the one-hot prefix is left as-is (standardizing binary indicators
    tends to hurt rather than help).
    """
    all_node = np.concatenate([g["x"] for g in graphs], axis=0)
    all_edge = np.concatenate([g["edge_attr"] for g in graphs], axis=0)

    node_mean = all_node.mean(axis=0)
    node_std  = all_node.std(axis=0) + 1e-8
    edge_mean = all_edge.mean(axis=0)
    edge_std  = all_edge.std(axis=0) + 1e-8

    return {
        "node_mean": node_mean,
        "node_std":  node_std,
        "edge_mean": edge_mean,
        "edge_std":  edge_std,
    }

# =========================
# CLASS WEIGHTS
# =========================

def compute_class_weights(graphs, num_classes):
    """
    Inverse-frequency weights for use with nn.CrossEntropyLoss(weight=...).
    Helps handle the class imbalance common in HOI relation datasets.
    """
    all_labels = [lbl for g in graphs for lbl in g["edge_label"]]
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = np.array([
        total / (num_classes * counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=np.float32)
    return weights

def preprocess():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("Loading raw data...")
    with open(INPUT_FILE, "rb") as f:
        raw = pickle.load(f)

    data       = raw["data"]
    node_names = raw["node_names"]
    rel_names  = raw["rel_names"]

    num_node_types = len(node_names)

    rel_name_to_idx = {name: i for i, name in enumerate(rel_names)}

    # Only keep relations that exist in this dataset
    present_relations = [r for r in USEFUL_RELATIONS if r in rel_name_to_idx]
    if len(present_relations) < len(USEFUL_RELATIONS):
        missing = set(USEFUL_RELATIONS) - set(present_relations)
        print(f"  Warning: these relations not found in dataset and will be skipped: {missing}")

    # Raw relation IDs for the useful relations (may be non-contiguous)
    useful_raw_ids = [rel_name_to_idx[r] for r in present_relations]

    # --- FIX: remap raw IDs → contiguous [0, num_classes-1] ---
    # raw IDs [3, 7, 12] → labels [0, 1, 2]
    raw_id_to_label = {raw_id: new_label for new_label, raw_id in enumerate(useful_raw_ids)}
    useful_raw_id_set = set(useful_raw_ids)
    num_classes = len(present_relations)

    # Human-readable label map to save with output
    label_map = {new_label: name for new_label, name in enumerate(present_relations)}

    print(f"Relations: {label_map}")
    print(f"Node types: {num_node_types}  |  Edge classes: {num_classes}")
    print(f"BBox format: {BBOX_FORMAT}  |  Human category ID: {HUMAN_CATEGORY_ID}")
    print()

    processed_graphs = []
    skipped_no_edges  = 0

    print("Processing graphs...")
    for example in tqdm(data):
        annotations = example["annotations"]
        relations   = example["relations"]

        # --- Node features ---
        node_features = np.array([
            node_features_from_annotation(obj, num_node_types)
            for obj in annotations
        ], dtype=np.float32)

        # --- Edges + labels (filtered to useful relations only) ---
        # Group all useful relations by (subject, object) pair
        pair_to_raw_ids = {}
        for (s, o, r) in relations:
            if r in useful_raw_id_set:
                pair = (s, o)
                if pair not in pair_to_raw_ids:
                    pair_to_raw_ids[pair] = []
                pair_to_raw_ids[pair].append(r)

        # For each pair, prefer semantic relation over geometric if both exist
        edge_list   = []
        edge_labels = []

        for (s, o), raw_ids in pair_to_raw_ids.items():
            labels = [raw_id_to_label[r] for r in raw_ids]
            names  = [label_map[l] for l in labels]

            semantic = [(l, n) for l, n in zip(labels, names) if n in SEMANTIC_RELATIONS]
            chosen_label = semantic[0][0] if semantic else labels[0]

            edge_list.append([s, o])
            edge_labels.append(chosen_label)

        if len(edge_list) == 0:
            skipped_no_edges += 1
            continue

        edge_array  = np.array(edge_list, dtype=np.int64)
        edge_index  = edge_array.T
        edge_feats  = build_edge_features(edge_list, annotations)
        edge_labels = np.array(edge_labels, dtype=np.int64)

        graph = {
            "x":           node_features,  # [num_nodes, node_feat_dim]
            "edge_index":  edge_index,      # [2, num_edges]
            "edge_attr":   edge_feats,      # [num_edges, edge_feat_dim]
            "edge_label":  edge_labels,     # [num_edges]  — contiguous class IDs
        }
        processed_graphs.append(graph)

    print(f"\nTotal graphs: {len(processed_graphs)}  |  Skipped (no useful edges): {skipped_no_edges}")

    # --- Train / val / test split ---
    random.seed(RANDOM_SEED)
    random.shuffle(processed_graphs)
    n     = len(processed_graphs)
    n_tr  = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_graphs = processed_graphs[:n_tr]
    val_graphs   = processed_graphs[n_tr : n_tr + n_val]
    test_graphs  = processed_graphs[n_tr + n_val:]

    print(f"Split  →  train: {len(train_graphs)}  val: {len(val_graphs)}  test: {len(test_graphs)}")

    # --- Normalization stats (computed on train split only) ---
    norm_stats = compute_feature_stats(train_graphs)

    # --- Class weights (from train split only) ---
    class_weights = compute_class_weights(train_graphs, num_classes)
    print("\nClass distribution (train):")
    all_train_labels = [lbl for g in train_graphs for lbl in g["edge_label"]]
    counts = Counter(all_train_labels)
    for cls_id, name in label_map.items():
        print(f"  [{cls_id}] {name:<12}  count: {counts.get(cls_id, 0):>6}  weight: {class_weights[cls_id]:.4f}")

    # --- Save ---
    output = {
        "splits": {
            "train": train_graphs,
            "val":   val_graphs,
            "test":  test_graphs,
        },
        "meta": {
            "node_feature_dim": processed_graphs[0]["x"].shape[1],
            "edge_feature_dim": processed_graphs[0]["edge_attr"].shape[1],
            "num_classes":      num_classes,
            "label_map":        label_map,       # int → relation name
            "node_names":       node_names,
            "bbox_format_used": BBOX_FORMAT,
            "human_category_id": HUMAN_CATEGORY_ID,
        },
        "norm_stats":    norm_stats,    # fit on train; apply to all splits
        "class_weights": class_weights, # for nn.CrossEntropyLoss(weight=...)
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Node feature dim : {output['meta']['node_feature_dim']}")
    print(f"Edge feature dim : {output['meta']['edge_feature_dim']}")
    print(f"Num classes      : {output['meta']['num_classes']}")


if __name__ == "__main__":
    preprocess()