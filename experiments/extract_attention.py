"""
Extract attention scores from a trained checkpoint.

Usage:
  python experiments/extract_attention.py \
    --data_dir ./data \
    --ckpt ./checkpoints/abgnn_best.pt \
    --out ./artifacts/attention_scores.pt
"""
import argparse
import torch

from abgnn.model import ABGNN
from abgnn.io import load_tables
from abgnn.graph_builder import build_graph
from abgnn.utils import ensure_dir


def forward_once_and_dump_attention(model, data, subgraph_edges, out_path: str):
    # A single forward pass (on all subgraphs at once) to populate attention buffers
    model.eval()
    with torch.no_grad():
        _ = model(
            data.x,
            data.undirected_edges_mapped_small,
            data.directed_edges_mapped_small,
            data.undirected_edges_mapped_small_middle,
            data.undirected_edges_mapped_middle,
            data.directed_edges_mapped_middle,
            subgraph_edges,
        )
        model.save_best_attention()  # detach+cpu

    torch.save(model.best_attn_scores, out_path)
    print(f"[Saved] attention scores → {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Rebuild graph/tensors on the same preprocessed CSVs
    water, small_edge, middle_edge, bio = load_tables(args.data_dir, encoding=args.encoding)
    packed = build_graph(
        water, small_edge, middle_edge, bio, block_size=args.block_size, device=device
    )
    data = packed["data"]
    subgraph_edges = packed["subgraph_edges"]  # all subgraphs
    in_channels = data.x.shape[1]

    # Load model
    model = ABGNN(in_channels=in_channels).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    ensure_dir(args.out_dir)
    out_path = f"{args.out_dir}/attention_scores.pt"
    forward_once_and_dump_attention(model, data, subgraph_edges, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--encoding", type=str, default="euc-kr")
    parser.add_argument("--block_size", type=int, default=183)
    parser.add_argument("--ckpt", type=str, default="./checkpoints/abgnn_best.pt")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
