"""
Dataset / DataLoader bridge.
- Sequential 7:1 split (front 7/8 = train, back 1/8 = test), no shuffle.
- Chronological mapping in this dataset: **2018–2022 → train**, **2023 → test**.
- SubgraphEdgeDataset yields (edge_index, label)
- collate_subgraphs -> (List[edge_index], labels_tensor)
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader


class SubgraphEdgeDataset(Dataset):
    def __init__(self, subgraph_edges: List[torch.Tensor], labels: torch.Tensor):
        assert len(subgraph_edges) == len(labels), "length mismatch"
        self.edges = subgraph_edges
        self.labels = labels

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx: int):
        return self.edges[idx], self.labels[idx]


def collate_subgraphs(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    edges_list = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch], dim=0).float().unsqueeze(1)  # [B,1]
    return edges_list, labels


def sequential_split(subgraph_edges: List[torch.Tensor], labels: torch.Tensor, ratio: float = 7/8):
    N = len(subgraph_edges)
    split_idx = int(N * ratio)
    train_edges = subgraph_edges[:split_idx]
    test_edges = subgraph_edges[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    return (train_edges, train_labels), (test_edges, test_labels)


def make_loaders(
    subgraph_edges: List[torch.Tensor],
    labels: torch.Tensor,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    (tr_e, tr_y), (te_e, te_y) = sequential_split(subgraph_edges, labels, ratio=7/8)

    train_ds = SubgraphEdgeDataset(tr_e, tr_y)
    test_ds = SubgraphEdgeDataset(te_e, te_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_subgraphs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_subgraphs,
    )
    return train_loader, test_loader
