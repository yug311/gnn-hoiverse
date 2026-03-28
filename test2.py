import pickle
import numpy as np
from collections import Counter

# This script loads the processed dataset and performs a series of sanity checks and inspections to understand its structure and contents.



with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

# ── Top-level keys ──────────────────────────────────────────
print("Top-level keys:", list(data.keys()))

# ── Metadata ────────────────────────────────────────────────
meta = data["meta"]
print("\n── Meta ──")
for k, v in meta.items():
    print(f"  {k}: {v}")

# ── Split sizes ─────────────────────────────────────────────
splits = data["splits"]
print("\n── Split sizes ──")
for split_name, graphs in splits.items():
    print(f"  {split_name}: {len(graphs)} graphs")

# ── Single graph inspection ──────────────────────────────────
sample = splits["train"][0]
print("\n── Sample graph (train[0]) ──")
print(f"  x          (node features) : {sample['x'].shape}   dtype={sample['x'].dtype}")
print(f"  edge_index                 : {sample['edge_index'].shape}   dtype={sample['edge_index'].dtype}")
print(f"  edge_attr  (edge features) : {sample['edge_attr'].shape}   dtype={sample['edge_attr'].dtype}")
print(f"  edge_label                 : {sample['edge_label'].shape}   dtype={sample['edge_label'].dtype}")
print(f"  Labels in this graph       : {sample['edge_label']}")

# ── Feature value sanity check ───────────────────────────────
print("\n── Feature value ranges (train[0]) ──")
print(f"  node features  min={sample['x'].min():.3f}  max={sample['x'].max():.3f}")
print(f"  edge features  min={sample['edge_attr'].min():.3f}  max={sample['edge_attr'].max():.3f}")

# ── Label distribution across full train split ───────────────
label_map = meta["label_map"]
all_labels = [lbl for g in splits["train"] for lbl in g["edge_label"]]
counts = Counter(all_labels)
print("\n── Label distribution (train) ──")
for cls_id, name in label_map.items():
    print(f"  [{cls_id}] {name:<14}  {counts.get(cls_id, 0):>6} edges")

# ── Norm stats ───────────────────────────────────────────────
print("\n── Norm stats ──")
norm = data["norm_stats"]
print(f"  node_mean  shape={norm['node_mean'].shape}  first 5: {norm['node_mean'][:5].round(4)}")
print(f"  node_std   shape={norm['node_std'].shape}   first 5: {norm['node_std'][:5].round(4)}")
print(f"  edge_mean  shape={norm['edge_mean'].shape}  values: {norm['edge_mean'].round(4)}")
print(f"  edge_std   shape={norm['edge_std'].shape}   values: {norm['edge_std'].round(4)}")

# ── Class weights ────────────────────────────────────────────
print("\n── Class weights ──")
for cls_id, name in label_map.items():
    print(f"  [{cls_id}] {name:<14}  {data['class_weights'][cls_id]:.4f}")

# ── Edge index sanity check (no out-of-bounds indices) ───────
print("\n── Edge index bounds check (train) ──")
bad = 0
for g in splits["train"]:
    num_nodes = g["x"].shape[0]
    if g["edge_index"].max() >= num_nodes:
        bad += 1
print(f"  Graphs with out-of-bounds edge indices: {bad}  (should be 0)")

sample = splits["train"][0]
for i, row in enumerate(sample["edge_attr"]):
    if row.min() < -1.5:
        print(f"Edge {i}: {row}")
        print(f"  nodes: {sample['edge_index'][:, i]}")