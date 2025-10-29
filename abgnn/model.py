import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ObservationPointNodeEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.obs_space = GATConv(in_channels, in_channels, heads=2, concat=True)
        self.obs_time = GATConv(in_channels * 2, in_channels * 2, heads=2, concat=True)
        self.attention_scores = {}

    def forward(self, x, undirected_edges_small, directed_edges_small):
        x, (edge_index_s, attn_space) = self.obs_space(
            x, undirected_edges_small, return_attention_weights=True
        )
        self.attention_scores["obs_space"] = (edge_index_s, attn_space)

        x, (edge_index_t, attn_time) = self.obs_time(
            x, directed_edges_small, return_attention_weights=True
        )
        self.attention_scores["obs_time"] = (edge_index_t, attn_time)
        return x


class ObservationPointToSubBasinAggregation(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.obs_to_sub = GATConv(in_channels, in_channels, heads=2, concat=True)
        self.attention_scores = {}

    def forward(self, x, undirected_edges_small_middle):
        x, (edge_index_m, attn_obs2sub) = self.obs_to_sub(
            x, undirected_edges_small_middle, return_attention_weights=True
        )
        self.attention_scores["obs_to_sub"] = (edge_index_m, attn_obs2sub)
        return x


class SubBasinNodeEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.sub_space = GATConv(in_channels, in_channels, heads=2, concat=True)
        self.sub_time = GATConv(in_channels * 2, in_channels * 2, heads=2, concat=True)
        self.attention_scores = {}

    def forward(self, x, undirected_edges_middle, directed_edges_middle):
        x, (edge_index_s, attn_sub_space) = self.sub_space(
            x, undirected_edges_middle, return_attention_weights=True
        )
        self.attention_scores["sub_space"] = (edge_index_s, attn_sub_space)

        x, (edge_index_t, attn_sub_time) = self.sub_time(
            x, directed_edges_middle, return_attention_weights=True
        )
        self.attention_scores["sub_time"] = (edge_index_t, attn_sub_time)
        return x


class ABGNN(torch.nn.Module):
    """
    GAT stack over:
      - observation points (space/time),
      - obs→sub aggregation,
      - sub-basin (space/time),
    then subgraph mean pooling → MLP → scalar regression.

    forward(..., subgraph_edges): subgraph_edges is a List[edge_index Tensor].
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.observation_point_node_embedding = ObservationPointNodeEmbedding(in_channels)
        self.observation_point_to_sub_basin_aggregation = ObservationPointToSubBasinAggregation(in_channels * 4)
        self.sub_basin_node_embedding = SubBasinNodeEmbedding(in_channels * 8)

        self.fc1 = torch.nn.Linear(in_channels * 32, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)
        self.regression_biodiversity = torch.nn.Linear(16, 1)

        self.attention_scores = {}
        self.best_attn_scores = {}

    def forward(
        self,
        x,
        undirected_edges_small,
        directed_edges_small,
        undirected_edges_small_middle,
        undirected_edges_middle,
        directed_edges_middle,
        subgraph_edges,
    ):
        x = self.observation_point_node_embedding(x, undirected_edges_small, directed_edges_small)
        self.attention_scores["observation_point"] = self.observation_point_node_embedding.attention_scores
        x = F.relu(x)

        x = self.observation_point_to_sub_basin_aggregation(x, undirected_edges_small_middle)
        self.attention_scores["obs_to_sub"] = self.observation_point_to_sub_basin_aggregation.attention_scores
        x = F.relu(x)

        x = self.sub_basin_node_embedding(x, undirected_edges_middle, directed_edges_middle)
        self.attention_scores["sub_basin"] = self.sub_basin_node_embedding.attention_scores
        x = F.relu(x)

        # subgraph mean pooling
        subgraph_embeddings = []
        for sub_edges in subgraph_edges:
            subgraph_nodes = torch.unique(sub_edges)
            subgraph_embedding = x[subgraph_nodes].mean(dim=0)
            subgraph_embeddings.append(subgraph_embedding)
        subgraph_embeddings = torch.stack(subgraph_embeddings)

        subgraph_embeddings = F.relu(self.fc1(subgraph_embeddings))
        subgraph_embeddings = F.relu(self.fc2(subgraph_embeddings))
        subgraph_embeddings = F.relu(self.fc3(subgraph_embeddings))
        subgraph_embeddings = F.relu(self.fc4(subgraph_embeddings))
        out = self.regression_biodiversity(subgraph_embeddings)
        return out

    def save_best_attention(self):
        def _dc(t):
            return t.detach().cpu() if torch.is_tensor(t) else t

        self.best_attn_scores = {
            "obs_space": tuple(_dc(t) for t in self.attention_scores["observation_point"]["obs_space"]),
            "obs_time": tuple(_dc(t) for t in self.attention_scores["observation_point"]["obs_time"]),
            "obs_to_sub": tuple(_dc(t) for t in self.attention_scores["obs_to_sub"]["obs_to_sub"]),
            "sub_space": tuple(_dc(t) for t in self.attention_scores["sub_basin"]["sub_space"]),
            "sub_time": tuple(_dc(t) for t in self.attention_scores["sub_basin"]["sub_time"]),
        }
