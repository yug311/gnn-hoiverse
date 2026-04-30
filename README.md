# Personalized Human-Object Interaction (HOI) Detection using GNNs

This project implements a sophisticated pipeline for detecting and classifying relationships between humans and objects in interior environments. By leveraging **Graph Neural Networks (GNNs)**, specifically the **GINEConv** architecture, the system moves beyond simple object detection to understand the contextual and spatial "topology" of a room.

## Overview

Traditional Neural Networks treat detected objects as independent entities. Our approach treats a scene as a **Graph**:
- **Nodes**: Objects detected (Person, Chair, TV, etc.).
- **Edges**: The spatial and semantic relationships between them.
- **Personalization**: Unlike generic models, our GNN uses specific geometric parameters (angles, relative distances, and IoU) to learn interaction signatures specific to individual users and layouts.

## The Pipeline

1. **Graph Construction (`preprocess.py`)**: Converts detections into a graph structure where edges represent 10-dimensional spatial vectors.
2. **Relationship Prediction (GNN)**: A Graph Isomorphic Network (GINEConv) performs message passing to predict interactions like *sitting*, *touching*, or *holding*.
3. **Smart Automation**: Predicted relationships can trigger IoT actions, such as setting the AC to 74°F when "sleeping" is detected.

## Performance

Our GNN architecture significantly outperforms standard MLP baselines:

| Model | Test Loss | Macro F1-Score | Accuracy |
| :--- | :--- | :--- | :--- |
| **GNN (GINEConv)** | **0.2677** | **0.8783** | **89%** |
| MLP Baseline | 0.4092 | 0.7767 | 81% |

## Project Structure


```
.
├── preprocess.py       # Processes hoiverse.pkl and hoiverse.zip into graph-ready data
├── GNN.py              # Model architecture (GINEConv) and training/evaluation loops
├── test.py             # Sanity checks for raw data
├── test2.py            # Sanity checks for processed data
├── data/               # Directory for processed pickle files
└── checkpoints/        # Stores best-performing model weights (best_gnn.pt)
```

## Dataset
The dataset for this project is available at the following link:
* [HoiVerse v1 Dataset](https://myweb.rz.uni-augsburg.de/~phatakmr/hoiverse/v1/)


## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yug311/gnn-hoiverse.git
   cd gnn-hoiverse
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** Make sure your `requirements.txt` includes `torch`, `torch-geometric`, `numpy`, `tqdm`, and `scikit-learn`.

## 🖥️ Usage

### 1. Preprocess the Data

Ensure `hoiverse.pkl` is in the root directory, then run:
```bash
python preprocess.py
```

### 2. Train the Model

This will train both the GNN and the MLP baseline for comparison:
```bash
python GNN.py
```

### 3. Inference & Inspection

Use the test scripts to verify data integrity:
```bash
python test.py
python test2.py
```

## 🧠 Key Methodology: GINEConv

The core of the model is the **Graph Isomorphism Network with Edge features**. It updates node embeddings using the following logic:

$$x_i^{(k)} = \text{MLP}^{(k)} \left( x_i^{(k-1)} + \sum_{j \in \mathcal{N}(i)} \text{ReLU}\left( x_j^{(k-1)} + e_{j,i} \right) \right)$$

This allows the "Person" node to aggregate information from the "Bed" node while specifically considering the edge parameters ($e_{j,i}$) like distance and angle.

## 🏠 Smart Home Applications

- **Contextual Comfort**: Detects "Sleeping" → Adjusts AC to 74°F.
- **Media Control**: Detects "Watching TV" → Dims lights and powers on system.
- **Personalization**: The model is sensitive to individual posture and distance, making it ideal for private household interiors.
