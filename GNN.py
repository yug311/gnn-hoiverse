import pickle
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm
from sklearn.metrics import classification_report, f1_score

# =========================
# CONFIG
# =========================

PROCESSED_FILE  = "data/processed.pkl"
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "best_gnn.pt")

SEMANTIC_RELATIONS    = {"sitting", "holding"}
OVERSAMPLE_MULTIPLIER = 3

# Model
HIDDEN_DIM  = 128
NUM_LAYERS  = 3
DROPOUT     = 0.3

# Training
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 10      # early stopping patience (epochs)

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# UTILITIES
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    print(f"Loading data from {PROCESSED_FILE} ...")
    with open(PROCESSED_FILE, "rb") as f:
        data = pickle.load(f)
    return data


def oversample(graphs, label_map, multiplier):
    semantic_ids = {i for i, name in label_map.items() if name in SEMANTIC_RELATIONS}
    result = []
    for g in graphs:
        result.append(g)
        if any(lbl in semantic_ids for lbl in g["edge_label"]):
            for _ in range(multiplier - 1):
                result.append(g)
    return result


def to_pyg(graphs, norm_stats):
    node_mean = norm_stats["node_mean"].astype(np.float32)
    node_std  = norm_stats["node_std"].astype(np.float32)
    edge_mean = norm_stats["edge_mean"].astype(np.float32)
    edge_std  = norm_stats["edge_std"].astype(np.float32)

    pyg_list = []
    for g in graphs:
        x         = (g["x"].astype(np.float32)        - node_mean) / node_std
        edge_attr = (g["edge_attr"].astype(np.float32) - edge_mean) / edge_std

        pyg_list.append(Data(
            x          = torch.tensor(x,               dtype=torch.float32),
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,       dtype=torch.float32),
            edge_label = torch.tensor(g["edge_label"], dtype=torch.long),
        ))
    return pyg_list


# =========================
# MODELS
# =========================

class EdgeMLP(nn.Module):
    def __init__(self, node_dim, edge_dim, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        h = torch.cat([x[ei[0]], x[ei[1]], ea], dim=-1)
        return self.net(h)


class GNNEdgeClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_classes,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Project raw node features into hidden space
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Message-passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            # edge_dim tells GINEConv to project edge features to hidden_dim
            # before adding them to the node message
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(BatchNorm(hidden_dim))

        # Edge classification head
        head_in = hidden_dim * 2 + edge_dim
        self.edge_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        # Node embedding via message passing
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, ei, ea)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Edge-level prediction
        edge_repr = torch.cat([x[ei[0]], x[ei[1]], ea], dim=-1)
        return self.edge_head(edge_repr)


# =========================
# TRAIN / EVAL / TEST
# =========================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss  = 0.0
    total_edges = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        loss   = criterion(logits, batch.edge_label)
        loss.backward()

        # Gradient clipping prevents occasional loss spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = batch.edge_label.size(0)
        total_loss  += loss.item() * n
        total_edges += n

    return total_loss / total_edges


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_map, verbose=False):
    model.eval()
    total_loss  = 0.0
    total_edges = 0
    all_preds   = []
    all_true    = []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        loss   = criterion(logits, batch.edge_label)

        n = batch.edge_label.size(0)
        total_loss  += loss.item() * n
        total_edges += n

        all_preds.append(logits.argmax(dim=-1).cpu())
        all_true.append(batch.edge_label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()

    avg_loss = total_loss / total_edges
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    if verbose:
        target_names = [label_map[i] for i in range(len(label_map))]
        print(classification_report(all_true, all_preds,
                                    target_names=target_names, zero_division=0))

    return avg_loss, macro_f1


def train_model(model, model_name, train_loader, val_loader,
                criterion, label_map, checkpoint_path=None):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    print(f"{'='*55}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Halve LR when val macro F1 stops improving for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_f1  = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_f1    = evaluate(model, val_loader, criterion, DEVICE, label_map)
        scheduler.step(val_f1)

        improved = "*" if val_f1 > best_val_f1 else ""
        print(f"  Epoch {epoch:>3}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_macro_f1={val_f1:.4f}  {improved}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
                print(f"           Checkpoint saved → {checkpoint_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}  "
                      f"(best val macro F1={best_val_f1:.4f})")
                break

    return best_state


# =========================
# MAIN
# =========================

def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    # ── Load processed data
    raw         = load_data()
    meta        = raw["meta"]
    norm_stats  = raw["norm_stats"]
    splits      = raw["splits"]
    label_map   = meta["label_map"]
    node_dim    = meta["node_feature_dim"]
    edge_dim    = meta["edge_feature_dim"]
    num_classes = meta["num_classes"]

    class_weights = torch.tensor(raw["class_weights"], dtype=torch.float32).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Oversample train split
    train_graphs = oversample(splits["train"], label_map, OVERSAMPLE_MULTIPLIER)
    random.shuffle(train_graphs)
    print(f"Graphs → train: {len(train_graphs)}  val: {len(splits['val'])}  test: {len(splits['test'])}")

    train_loader = DataLoader(to_pyg(train_graphs,   norm_stats), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(to_pyg(splits["val"],  norm_stats), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(to_pyg(splits["test"], norm_stats), batch_size=BATCH_SIZE, shuffle=False)

    gnn = GNNEdgeClassifier(
        node_dim    = node_dim,
        edge_dim    = edge_dim,
        hidden_dim  = HIDDEN_DIM,
        num_classes = num_classes,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(DEVICE)

    mlp = EdgeMLP(
        node_dim    = node_dim,
        edge_dim    = edge_dim,
        num_classes = num_classes,
        hidden      = HIDDEN_DIM,
        dropout     = DROPOUT,
    ).to(DEVICE)

    print(f"\nGNN parameters : {sum(p.numel() for p in gnn.parameters()):,}")
    print(f"MLP parameters : {sum(p.numel() for p in mlp.parameters()):,}")

    # ── Train GNN
    gnn_state = train_model(
        gnn, "GNN (GINEConv)",
        train_loader, val_loader,
        criterion, label_map,
        checkpoint_path=CHECKPOINT_FILE,
    )

    # ── Train MLP baseline
    mlp_state = train_model(
        mlp, "MLP Baseline",
        train_loader, val_loader,
        criterion, label_map,
        checkpoint_path=None,
    )

    # ── Test both with best weights
    print(f"\n{'='*55}")
    print("  FINAL TEST RESULTS")
    print(f"{'='*55}")

    for model_name, model, state in [
        ("GNN (GINEConv)", gnn, gnn_state),
        ("MLP Baseline",   mlp, mlp_state),
    ]:
        model.load_state_dict(state)
        test_loss, test_f1 = evaluate(
            model, test_loader, criterion, DEVICE, label_map, verbose=True
        )
        print(f"{model_name}  →  test_loss={test_loss:.4f}  test_macro_f1={test_f1:.4f}\n")


if __name__ == "__main__":
    main()