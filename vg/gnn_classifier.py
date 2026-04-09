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