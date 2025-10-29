# README.md

# A Graph Neural Network Approach for Aquatic Biodiversity Prediction Leveraging Water System Interconnections

**Status:** Under review at *Ecological Informatics*
**Goal:** Build a heterogeneous water-system graph (observation points ↔ sub-basins ↔ biodiversity) and predict aquatic biodiversity with a GNN that models spatial/temporal relations.

---

## Repository Structure

```text
ABGNN/
├─ abgnn/
│  ├─ __init__.py
│  ├─ model.py            # ABGNN architecture (GAT stack + subgraph mean pooling)
│  ├─ utils.py            # seeding & small helpers
│  ├─ io.py               # load preprocessed CSVs (EUC-KR)
│  ├─ graph_builder.py    # build heterogeneous graph + tensors
│  └─ dataset.py          # DataLoader bridge (sequential 7:1 split; 2018–2022=train, 2023=test)
│
├─ experiments/
│  ├─ train.py            # training + save weights (best/last)
│  └─ extract_attention.py# forward once + dump attention tensors
│
├─ data/                  # (included) preprocessed CSVs
│  ├─ new_water.csv
│  ├─ small_edge.csv
│  ├─ middle_edge.csv
│  └─ bio_final.csv
│
├─ .gitignore             # keep caches/ckpts out of git; CSVs remain tracked
├─ LICENSE                # MIT license
├─ README.md
└─ abgnn.yaml             # conda environment spec
```

---

## Environment

Use the provided **conda** environment:

```bash
conda env create -f abgnn.yaml
conda activate abgnn
```

**Author’s setup (recommended):** Python **3.10.14**, PyTorch **2.5.1**, PyTorch Geometric **2.6.1**
Hardware used: Intel Core **i9-14900K** CPU + **2× RTX 4090** GPUs.

---

## Data

* **Included:** preprocessed CSVs under `./data` (encoding: **euc-kr**).
* **Not shared:** raw/original data (license/privacy constraints).

**Terminology / 용어**
* 측정소 → *observation-point*
* 중권역 → *sub-basin*

**CSV expectations (KOR/ENG)**
* `new_water.csv`:  
  columns include  
  `id`, `측정소명 (observation-point name)`, `중권역_id (sub-basin ID)`, `중권역명 (sub-basin name)`, and 9 water-quality features:  
  • `수온 (water temperature)`  
  • `용존산소량 (dissolved oxygen)`  
  • `화학적산소요구량 (chemical oxygen demand; COD)`  
  • `총질소 (total nitrogen)`  
  • `총인 (total phosphorus)`  
  • `수소이온농도 (pH)`  
  • `전기전도도 (electrical conductivity)`  
  • `염분 (salinity)`  
  • `탁도 (turbidity)`
* `small_edge.csv`, `middle_edge.csv`: undirected edges with `node1`,`node2`
  (the code canonically normalizes to `(min, max)` and deduplicates)
* `bio_final.csv`:  
  `중권역 명 (sub-basin name)`, `년 (year)`, `회 (period; biannual: 1=H1, 2=H2)`, … and **target** (biodiversity index) in the last column

---

## Training

Sequential **7:1** split (first 7/8 = train, last 1/8 = test; 2018–2022=train, 2023=test), no shuffle.

```bash
python experiments/train.py \
  --data_dir ./data \
  --encoding euc-kr \
  --block_size 183 \
  --epochs 1000 \
  --eval_every 100 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --batch_size 64
```

**Outputs**

* `checkpoints/abgnn_best.pt`, `checkpoints/abgnn_last.pt`
* `artifacts/metrics.json` (best_test_loss, best_epoch, graph stats)
* `artifacts/node_mapping.json` (node → index)

---

## Extract Attention Scores

```bash
python experiments/extract_attention.py \
  --data_dir ./data \
  --encoding euc-kr \
  --block_size 183 \
  --ckpt ./checkpoints/abgnn_best.pt \
  --out_dir ./artifacts
```

**Output**

* `artifacts/attention_scores.pt`
  Keys: `obs_space`, `obs_time`, `obs_to_sub`, `sub_space`, `sub_time`
  Each value is a tuple `(edge_index, attn)` stored as CPU tensors.

---

## Model Overview

* **Layers**

  * Observation points (소권역): `GATConv` for **space** and **time**
  * Obs → Sub-basin aggregation: `GATConv`
  * Sub-basin (중권역): `GATConv` for **space** and **time**
* **Pooling:** connected-component (subgraph) **mean pooling**
* **Head:** MLP → scalar regression (biodiversity)
* **Loss:** MSE

---

## Reproducibility Notes

* Seed fixed to `42` (numpy/torch/cuda).
* Undirected edges normalized to `(min, max)` then deduplicated.
* Time replication uses `--block_size` (default `183`, half year).
* Subgraph ordering made deterministic via a middle-region–based key before the sequential split.

---

## Citation

If you find this repository useful, please cite:

```
A Graph Neural Network Approach for Aquatic Biodiversity Prediction Leveraging Water System Interconnections.
Under review at Ecological Informatics (Elsevier).
```

BibTeX (update authors/years upon acceptance):

```bibtex
@article{abgnn_biodiversity_under_review,
  title   = {A Graph Neural Network Approach for Aquatic Biodiversity Prediction Leveraging Water System Interconnections},
  author  = {Byeongkeun Kwon and Dain Lee and Hyunjun Ko and Hanbin Lee and Hyeonjun Hwang and Suhyeon Kim},
  journal = {Ecological Informatics},
  note    = {Under review. Equal contribution: Byeongkeun Kwon and Dain Lee. Co-corresponding authors: Hyeonjun Hwang and Suhyeon Kim.},
  year    = {2025}
}
```

---

## License

Released under the **MIT License**. See [LICENSE](#license-1) for details.

---

## Contact

For questions and collaborations, please open an issue or contact the authors.

---

# LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 ABGNN Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

# .gitignore

> **What is this?** Files listed here are **not** tracked by Git. It prevents committing caches, virtual envs, temporary outputs, etc. Preprocessed CSVs in `data/` are intentionally kept **tracked**.

```gitignore
# --- Python common ---
__pycache__/
*.py[cod]
*.so
*.pyd
*.pyo
*.pkl
*.egg-info/
.eggs/
build/
dist/
.coverage
.pytest_cache/

# --- Environments ---
.env
.venv/
venv/
env/

# --- OS / IDE ---
.DS_Store
*.swp
*.swo
.vscode/
.idea/

# --- Experiment outputs ---
checkpoints/
artifacts/
runs/
logs/
lightning_logs/

# --- Important: keep preprocessed CSVs tracked ---
# data/*.csv  <-- DO NOT ignore; included in the repo
# If you later add large raw files, ignore like:
# data/raw/
# data/*.zip
```
