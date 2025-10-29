"""
Training script
- Uses abgnn.yaml (Conda/Poetry/etc) as your environment spec (not handled here).
- Saves ONLY model weights (no attention) to checkpoints/.
"""
import argparse
import json
from tqdm import tqdm

import torch

from abgnn.model import ABGNN
from abgnn.utils import seed_everything, ensure_dir, save_node_mapping
from abgnn.io import load_tables
from abgnn.graph_builder import build_graph
from abgnn.datasets import make_loaders


def evaluate(model, data, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for edges_list, labels in loader:
            out = model(
                data.x,
                data.undirected_edges_mapped_small,
                data.directed_edges_mapped_small,
                data.undirected_edges_mapped_small_middle,
                data.undirected_edges_mapped_middle,
                data.directed_edges_mapped_middle,
                edges_list,
            )
            loss = criterion(out, labels.to(device))
            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)
    return total_loss / max(1, n)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # I/O
    water, small_edge, middle_edge, bio = load_tables(args.data_dir, encoding=args.encoding)
    packed = build_graph(
        water,
        small_edge,
        middle_edge,
        bio,
        block_size=args.block_size,
        device=device,
    )
    data = packed["data"]
    subgraph_edges = packed["subgraph_edges"]
    labels = packed["labels"]
    stats = packed["stats"]
    node_mapping = packed["node_mapping"]

    # DataLoaders
    train_loader, test_loader = make_loaders(
        subgraph_edges, labels, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type=="cuda")
    )

    # Model/optim
    model = ABGNN(in_channels=data.x.shape[1]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ensure_dir(args.ckpt_dir)
    ensure_dir(args.out_dir)

    best_test = float("inf")
    best_epoch = -1

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        for edges_list, labels_batch in train_loader:
            optimizer.zero_grad()
            out = model(
                data.x,
                data.undirected_edges_mapped_small,
                data.directed_edges_mapped_small,
                data.undirected_edges_mapped_small_middle,
                data.undirected_edges_mapped_middle,
                data.directed_edges_mapped_middle,
                edges_list,
            )
            loss = criterion(out, labels_batch.to(device))
            loss.backward()
            optimizer.step()

        if (epoch % args.eval_every) == 0 or epoch == args.epochs:
            test_loss = evaluate(model, data, test_loader, criterion, device)
            if test_loss < best_test:
                best_test = test_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f"{args.ckpt_dir}/abgnn_best.pt")

            torch.save(model.state_dict(), f"{args.ckpt_dir}/abgnn_last.pt")
            print(f"[Epoch {epoch:4d}] test_loss={test_loss:.6f} | best={best_test:.6f} (epoch {best_epoch})")

    # Save metadata
    with open(f"{args.out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_test_loss": best_test, "best_epoch": best_epoch, "stats": stats},
            f,
            indent=2,
            ensure_ascii=False,
        )
    save_node_mapping(node_mapping, f"{args.out_dir}/node_mapping.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--encoding", type=str, default="euc-kr")
    parser.add_argument("--block_size", type=int, default=183)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    args = parser.parse_args()
    main(args)
