"""
Graph construction module.
- Harmonizes IDs, dedups undirected edges, constructs a unified heterogeneous graph.
- Produces:
    * PyG-compatible tensors on device for global edges and node features
    * list of subgraph edge_index tensors (connected components)
    * subgraph labels (torch.LongTensor)
    * node_mapping dict[str -> int]
"""
from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


def _dedup_undirected_edges(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "node1": df[["node1", "node2"]].min(axis=1),
                "node2": df[["node1", "node2"]].max(axis=1),
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_graph(
    water: pd.DataFrame,
    small_edge: pd.DataFrame,
    middle_edge: pd.DataFrame,
    bio: pd.DataFrame,
    *,
    block_size: int = 183,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Normalize & dedup
    small_edge = _dedup_undirected_edges(small_edge)
    middle_edge = _dedup_undirected_edges(middle_edge)

    # Prefixing
    water = water.copy()
    water["observation_point_id"] = "ob_" + water["observation_point_id"].astype(str)
    water["observation_point"] = "ob_" + water["observation_point"].astype(str)
    water["sub_basin_id"] = "sub_" + water["sub_basin_id"].astype(str)
    water["sub_basin"] = "sub_" + water["sub_basin"].astype(str)

    small_edge = small_edge.copy()
    small_edge["node1"] = "ob_" + small_edge["node1"].astype(str)
    small_edge["node2"] = "ob_" + small_edge["node2"].astype(str)

    middle_edge = middle_edge.copy()
    middle_edge["node1"] = "sub_" + middle_edge["node1"].astype(str)
    middle_edge["node2"] = "sub_" + middle_edge["node2"].astype(str)

    # Bio table ordering and naming
    bio = bio.sort_values(["sub_basin", "year", "collection_round"]).reset_index(drop=True)
    _exclude = ['Samcheokosipcheon',
                'Sapgyocheon',
                'Yeonggang',
                'Yocheon',
                'Yongdamdaemharyu',
                'Wicheon',
                'Imhadaem',
                'Chogang',
                'Chungjudaemharyu',
                'Tamjingang',
                'Hangang_Seohae',
                'Hantangang',
                'Hyeongsangang',
                'Hongcheongang',
                'Hwangryonggang',
                'Hoeyagang',
                'Hoecheon']
    bio = bio.loc[~bio["sub_basin"].isin(_exclude)].reset_index(drop=True)
    bio["sub_basin"] = "sub_" + bio["sub_basin"].astype(str)

    # Drop unused cols if exist
    for c in ["latitude", "longitude", "month", "day", "year"]:
        if c in water.columns:
            water = water.drop(columns=[c])

    # Half-year key
    bio["year"] = bio["year"] + (bio["collection_round"] - 1) * 0.5

    mid_bio_edge = water[["sub_basin_id", "sub_basin"]].reset_index(drop=True)

    # Suffixing rules
    years = ["_2018.0", "_2018.5", "_2019.0", "_2019.5", "_2020.0", "_2020.5",
             "_2021.0", "_2021.5", "_2022.0", "_2022.5", "_2023.0", "_2023.5"]

    rep = 12
    mid_bio_edge = mid_bio_edge.assign(tmp=list(range(rep)) * (len(mid_bio_edge) // rep or 1))
    mid_bio_edge["tmp"] = mid_bio_edge["tmp"].astype(str)
    mid_bio_edge["sub_basin"] = "bio_" + (mid_bio_edge["sub_basin"] + "_" + mid_bio_edge["tmp"])
    mid_bio_edge = mid_bio_edge.drop(columns=["tmp"]).reset_index(drop=True)

    mid_bio_edge = mid_bio_edge.assign(tmp=(years * (len(mid_bio_edge) // len(years) + 1))[: len(mid_bio_edge)])
    mid_bio_edge["sub_basin"] = mid_bio_edge["sub_basin"] + mid_bio_edge["tmp"]
    mid_bio_edge = mid_bio_edge.drop(columns=["tmp"]).reset_index(drop=True)

    bio = bio.assign(tmp=(years * (len(bio) // len(years) + 1))[: len(bio)])
    bio["sub_basin"] = "bio_" + (bio["sub_basin"] + "_" + bio["tmp"].astype(str))
    bio = bio.drop(columns=["tmp"]).reset_index(drop=True)

    # Build graph
    G = nx.Graph()

    # middle nodes with feature means
    middle_df = (
        water.groupby("sub_basin_id")
        .agg(
            {
                "water_temp": "mean",
                "BOD": "mean",
                "COD": "mean",
                "TN": "mean",
                "TP": "mean",
                "pH": "mean",
                "EC": "mean",
                "salinity": "mean",
                "turbidity": "mean",
            }
        )
        .reset_index()
    )
    for _, row in middle_df.iterrows():
        G.add_node(row["sub_basin_id"], attr=row.values.tolist()[1:])

    # middle directed edges over contiguous windows
    num_blocks = max(1, len(water) // block_size)
    directed_edges_middle = []
    for i in range(num_blocks):
        tmp = water.iloc[i * block_size : (i + 1) * block_size]
        prev = None
        for node_id in tmp["sub_basin_id"]:
            if prev is not None and prev in G and node_id in G:
                G.add_edge(prev, node_id)
                directed_edges_middle.append((prev, node_id))
            prev = node_id

    # middle undirected edges replicated over time
    tmp_middle_edge = middle_edge.copy()
    tmp_middle_edge["node1"] += "_0"
    tmp_middle_edge["node2"] += "_0"
    for i in range(1, num_blocks):
        df = middle_edge.copy()
        df["node1"] += f"_{i}"
        df["node2"] += f"_{i}"
        tmp_middle_edge = pd.concat([tmp_middle_edge, df], ignore_index=True)

    undirected_edges_middle = []
    for _, row in tmp_middle_edge.iterrows():
        if row["node1"] in G and row["node2"] in G:
            G.add_edge(row["node1"], row["node2"])
            undirected_edges_middle.append((row["node1"], row["node2"]))

    # bio nodes & labels (dummy attr)
    subgraph_labels = []
    for _, row in bio.iterrows():
        node_attributes = [0] * 9
        label_value = row.iloc[-1]
        G.add_node(row["sub_basin"], attr=node_attributes, label=label_value)
        subgraph_labels.append(label_value)

    # middle-bio links
    undirected_middle_bio = []
    for _, row in mid_bio_edge.iterrows():
        if row["sub_basin_id"] in G and row["sub_basin"] in G:
            G.add_edge(row["sub_basin_id"], row["sub_basin"])
            undirected_middle_bio.append((row["sub_basin_id"], row["sub_basin"]))

    # remove bio dummy nodes before adding small layers
    to_remove = [n for n, d in G.nodes(data=True) if "attr" in d and d["attr"] == [0] * 9]
    G.remove_nodes_from(to_remove)

    # small nodes (features)
    for _, row in water.iterrows():
        node_attr = row.values.tolist()[4:13]
        G.add_node(row["observation_point_id"], attr=node_attr)

    # small directed edges
    directed_edges_small = []
    for i in range(num_blocks):
        tmp = water.iloc[i * block_size : (i + 1) * block_size]
        prev = None
        for node_id in tmp["observation_point_id"]:
            if prev is not None and prev in G and node_id in G:
                G.add_edge(prev, node_id)
                directed_edges_small.append((prev, node_id))
            prev = node_id

    # small undirected edges replicated
    tmp_small_edge = small_edge.copy()
    tmp_small_edge["node1"] += "_0"
    tmp_small_edge["node2"] += "_0"
    for i in range(1, num_blocks):
        df = small_edge.copy()
        df["node1"] += f"_{i}"
        df["node2"] += f"_{i}"
        tmp_small_edge = pd.concat([tmp_small_edge, df], ignore_index=True)

    undirected_edges_small = []
    for _, row in tmp_small_edge.iterrows():
        if row["node1"] in G and row["node2"] in G:
            G.add_edge(row["node1"], row["node2"])
            undirected_edges_small.append((row["node1"], row["node2"]))

    # small-middle links
    small_middle_edge = water[["observation_point_id", "sub_basin_id"]]
    undirected_edges_small_middle = []
    for _, row in small_middle_edge.iterrows():
        if row["observation_point_id"] in G and row["sub_basin_id"] in G:
            G.add_edge(row["observation_point_id"], row["sub_basin_id"])
            undirected_edges_small_middle.append((row["observation_point_id"], row["sub_basin_id"]))

    # connected components as subgraphs
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    # deterministic ordering by a simple key
    subgraph_keys = []
    for sg in subgraphs:
        mids = [n for n in sg.nodes if str(n).startswith("sub_")]
        key = mids[0] if len(mids) > 0 else list(sg.nodes())[0]
        subgraph_keys.append(str(key))

    order = sorted(range(len(subgraphs)), key=lambda i: subgraph_keys[i])

    node_mapping: Dict[str, int] = {node: i for i, node in enumerate(G.nodes())}

    def _map_edges(edge_list: List[Tuple[str, str]]):
        return [(node_mapping[u], node_mapping[v]) for u, v in edge_list]

    data = Data()
    data.x = torch.tensor([G.nodes[n]["attr"] for n in G.nodes], dtype=torch.float, device=device)

    data.undirected_edges_mapped_small = torch.tensor(np.array(_map_edges(undirected_edges_small)).T, dtype=torch.long, device=device)
    data.directed_edges_mapped_small = torch.tensor(np.array(_map_edges(directed_edges_small)).T, dtype=torch.long, device=device)
    data.undirected_edges_mapped_small_middle = torch.tensor(np.array(_map_edges(undirected_edges_small_middle)).T, dtype=torch.long, device=device)
    data.undirected_edges_mapped_middle = torch.tensor(np.array(_map_edges(undirected_edges_middle)).T, dtype=torch.long, device=device)
    data.directed_edges_mapped_middle = torch.tensor(np.array(_map_edges(directed_edges_middle)).T, dtype=torch.long, device=device)

    # subgraph edge tensors (ordered)
    subgraph_edges = []
    for i in order:
        sg = subgraphs[i]
        edges = [(node_mapping[u], node_mapping[v]) for u, v in sg.edges()]
        if len(edges) == 0:
            nodes = list(sg.nodes())
            nid = node_mapping[nodes[0]]
            edges = [(nid, nid)]
        subgraph_edges.append(torch.tensor(np.array(edges).T, dtype=torch.long, device=device))

    # labels
    labels = torch.tensor(subgraph_labels, dtype=torch.long, device=device)
    min_len = min(len(labels), len(subgraph_edges))
    labels = labels[:min_len]
    subgraph_edges = subgraph_edges[:min_len]

    # pack
    packed = {
        "data": data,
        "subgraph_edges": subgraph_edges,
        "labels": labels,
        "node_mapping": node_mapping,
        "stats": {
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "num_subgraphs": len(subgraphs),
            "num_blocks": num_blocks,
        },
    }
    return packed
