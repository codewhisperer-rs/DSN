import os
import math
import json
import argparse
from typing import Dict

import numpy as np
import pandas as pd

# Local minimal copies to avoid requiring full project deps
import random as _py_random


def set_random_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    np.random.seed(seed)
    _py_random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, num_nodes, seed=None):
        self.seed = None
        self.num_node = num_nodes
        self.src_list = np.concatenate(src_list)
        self.dst_list = np.concatenate(dst_list)
        self.src_list_uni = np.unique(self.src_list)
        self.dst_list_uni = np.unique(self.dst_list)
        self.edge = list(zip(self.src_list, self.dst_list))
        try:
            import torch
            self.edge_list = torch.stack([torch.Tensor(self.src_list), torch.Tensor(self.dst_list)])
        except Exception:
            self.edge_list = None

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list_uni), size)
            dst_index = np.random.randint(0, len(self.dst_list_uni), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list_uni), size)
            dst_index = self.random_state.randint(0, len(self.dst_list_uni), size)
        return self.src_list_uni[src_index], self.dst_list_uni[dst_index]

    def sample_semba(self, size):
        # Strict negative sampling (if torch_geometric is available)
        try:
            import torch
            from torch_geometric.utils import negative_sampling
            if self.edge_list is None:
                self.edge_list = torch.stack([torch.Tensor(self.src_list), torch.Tensor(self.dst_list)])
            null_ei_batch = negative_sampling(self.edge_list, num_nodes=self.num_node, num_neg_samples=size)
            while ((null_ei_batch == 0).any()) == True:
                null_ei_batch = negative_sampling(self.edge_list, num_nodes=self.num_node, num_neg_samples=size)
            return np.array(null_ei_batch[0]), np.array(null_ei_batch[1])
        except Exception:
            # Fallback to uniform if torch_geometric unavailable
            return self.sample(size)


def parse_args():
    p = argparse.ArgumentParser("Export fixed dataset splits and optional negatives to CSV")
    p.add_argument("--data", "-d", type=str, required=True,
                   choices=["wiki-RfA", "epinions", "bitcoinalpha", "bitcoinotc"],
                   help="Dataset name used in DynamicData/weight/ml_<data>.csv")
    p.add_argument("--seed", type=int, default=0, help="Random seed for split and mask sampling")
    p.add_argument("--val_quantile", type=float, default=0.70, help="Quantile for validation split time")
    p.add_argument("--test_quantile", type=float, default=0.85, help="Quantile for test split time")
    p.add_argument("--mask_frac", type=float, default=0.10, help="Fraction of nodes to mask from training (sampled from post-val nodes)")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory for CSVs; default DynamicData/splits/<data>_seed<seed>")
    p.add_argument("--export_neg", action="store_true", help="Also export fixed negative samples for train/val/test/trans/induc")
    p.add_argument("--neg_method", type=str, default="uniform", choices=["uniform", "semba"],
                   help="Negative sampling method: uniform uses RandEdgeSampler.sample; semba uses torch_geometric.negative_sampling")
    p.add_argument("--neg_per_pos", type=int, default=1, help="Number of negatives per positive when exporting negatives")
    p.add_argument("--mask_nodes_path", type=str, default=None, help="Optional path to pre-defined mask_nodes CSV (single column 'node') to reproduce exact split")
    p.add_argument("--save_mask_nodes", action="store_true", help="Save the sampled mask_nodes to mask_nodes.csv in out_dir")
    return p.parse_args()


def load_dataset_csv(dataset: str) -> pd.DataFrame:
    # Build path relative to this file location
    here = os.path.dirname(__file__)
    root = os.path.dirname(here)  # DSN/PolarDSN
    csv_path = os.path.join(root, "DynamicData", "weight", f"ml_{dataset}.csv")
    if not os.path.exists(csv_path):
        # Try relative to current working directory as a fallback
        alt_path = os.path.join("..", "DynamicData", "weight", f"ml_{dataset}.csv")
        csv_path = alt_path if os.path.exists(alt_path) else csv_path
    df = pd.read_csv(csv_path)
    # Normalize column names if the first index column is unnamed
    if df.columns[0] == df.columns[0] and (df.columns[0] == "" or "Unnamed" in df.columns[0]):
        df = df.drop(columns=[df.columns[0]])
    return df


def compute_splits(g_df: pd.DataFrame, seed: int, val_q: float, test_q: float, mask_frac: float, mask_nodes_override: np.ndarray = None) -> Dict[str, np.ndarray]:
    # Required columns: u, i, ts, label, weight, idx
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    sign_l = g_df.label.values
    ts_l = g_df.ts.values
    weight_l = g_df.weight.values

    set_random_seed(seed)

    max_idx = max(src_l.max(), dst_l.max())
    _ = max_idx  # unused but maintained for parity

    # Time thresholds
    val_time, test_time = list(np.quantile(g_df.ts, [val_q, test_q]))

    total_node_set = set(np.unique(np.hstack([src_l, dst_l])))
    num_total_unique_nodes = len(total_node_set)

    # Sample mask nodes only from nodes appearing after val_time
    post_val_nodes = set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))
    num_mask = int(mask_frac * num_total_unique_nodes)
    if num_mask > len(post_val_nodes):
        num_mask = len(post_val_nodes)
    if mask_nodes_override is not None:
        # Use provided mask nodes exactly (intersect to be safe)
        mask_node_set = set([int(x) for x in mask_nodes_override]) & post_val_nodes
    else:
        # Match main.py behavior: Python's random.sample on a set
        import random as _rnd
        # set_random_seed already seeded _rnd
        # Python 3.12 requires a sequence; use sorted set for stability
        pop = sorted(post_val_nodes)
        mask_node_set = set(_rnd.sample(pop, num_mask)) if num_mask > 0 else set()

    mask_src_flag = np.array([u in mask_node_set for u in src_l])
    mask_dst_flag = np.array([v in mask_node_set for v in dst_l])
    none_node_flag = (1 - mask_src_flag.astype(int)) * (1 - mask_dst_flag.astype(int))

    train_flag = (ts_l <= val_time) * (none_node_flag > 0)

    train_src_l = src_l[train_flag]
    train_dst_l = dst_l[train_flag]
    train_ts_l = ts_l[train_flag]
    train_e_idx_l = e_idx_l[train_flag]
    train_label_l = sign_l[train_flag]
    train_weight_l = weight_l[train_flag]

    train_node_set = set(train_src_l).union(train_dst_l)

    new_node_set = total_node_set - train_node_set

    val_flag = (ts_l <= test_time) * (ts_l > val_time)
    test_flag = ts_l > test_time

    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
    is_seen_node_edge = np.array([(a in train_node_set and b in train_node_set) for a, b in zip(src_l, dst_l)])

    tr_test_flag = test_flag * is_seen_node_edge  # transductive
    nn_test_flag = test_flag * is_new_node_edge   # inductive

    # Package all views as index masks so we can slice consistently later
    return {
        "train_mask": train_flag,
        "val_mask": val_flag,
        "test_mask": test_flag,
        "test_seen_mask": tr_test_flag,
        "test_unseen_mask": nn_test_flag,
        "val_time": np.array([val_time]),
        "test_time": np.array([test_time]),
        "mask_nodes": np.array(sorted(list(mask_node_set))),
    }


def random_sample_deterministic(population, k):
    # Kept for compatibility; not used for mask anymore.
    if k <= 0:
        return []
    if k >= len(population):
        return population
    idx = np.random.choice(len(population), size=k, replace=False)
    return [population[i] for i in idx]


def df_from_mask(g_df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    cols = ["u", "i", "ts", "label", "weight", "idx"]
    return g_df.loc[mask, cols].reset_index(drop=True)


def export_positives(g_df: pd.DataFrame, masks: Dict[str, np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Always export train/val/test
    base_map = {
        "train_mask": "train",
        "val_mask": "val",
        "test_mask": "test",
    }
    for key, name in base_map.items():
        df = df_from_mask(g_df, masks[key])
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    # Export only canonical names for the two test subsets
    trans_df = df_from_mask(g_df, masks["test_seen_mask"])   # transductive
    trans_df.to_csv(os.path.join(out_dir, "transductive.csv"), index=False)
    induc_df = df_from_mask(g_df, masks["test_unseen_mask"]) # inductive
    induc_df.to_csv(os.path.join(out_dir, "inductive.csv"), index=False)


def export_negatives(g_df: pd.DataFrame, masks: Dict[str, np.ndarray], out_dir: str,
                     neg_method: str = "uniform", neg_per_pos: int = 1,
                     seeds: Dict[str, int] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Build samplers consistent with main.py
    # src/dst lists for samplers
    src_l = g_df.u.values
    dst_l = g_df.i.values
    num_total_unique_nodes = len(set(np.unique(np.hstack([src_l, dst_l]))))

    # Train/val/test splits
    train_df = df_from_mask(g_df, masks["train_mask"])  # used for samplers
    val_df = df_from_mask(g_df, masks["val_mask"])      # used for samplers
    test_df = df_from_mask(g_df, masks["test_mask"])    # used for samplers

    def build_sampler(arr_src, arr_dst, seed=None):
        sampler = RandEdgeSampler((arr_src,), (arr_dst,), num_total_unique_nodes, seed=seed)
        return sampler

    # Seeds aligned with main.py usage
    if seeds is None:
        seeds = {
            "val": 0,
            "test": 1,
            "transductive": 2,
            "inductive": 3,
        }

    # samplers for different phases
    train_sampler = build_sampler(train_df.u.values, train_df.i.values, seed=None)
    val_sampler = RandEdgeSampler((train_df.u.values, val_df.u.values), (train_df.i.values, val_df.i.values), num_total_unique_nodes, seed=seeds["val"]) 
    test_sampler = RandEdgeSampler((train_df.u.values, val_df.u.values, test_df.u.values), (train_df.i.values, val_df.i.values, test_df.i.values), num_total_unique_nodes, seed=seeds["test"]) 
    trans_sampler = RandEdgeSampler((train_df.u.values, val_df.u.values, test_df.u.values), (train_df.i.values, val_df.i.values, test_df.i.values), num_total_unique_nodes, seed=seeds["transductive"]) 
    induc_sampler = RandEdgeSampler((train_df.u.values, val_df.u.values, test_df.u.values), (train_df.i.values, val_df.i.values, test_df.i.values), num_total_unique_nodes, seed=seeds["inductive"]) 

    def sample_neg(dst_sampler: RandEdgeSampler, size: int):
        if neg_method == "uniform":
            # Only dst part is used downstream; src from sampler is discarded in eval
            _, dst_fake = dst_sampler.sample(size)
            return dst_fake
        elif neg_method == "semba":
            # Use strict negative sampling that avoids existing edges, but not src-aligned
            src_fake, dst_fake = dst_sampler.sample_semba(size)
            return dst_fake
        else:
            raise ValueError("Unknown neg_method")

    def export_for_split(name: str, mask_key: str, sampler: RandEdgeSampler):
        pos_df = df_from_mask(g_df, masks[mask_key])
        if len(pos_df) == 0:
            # Still write an empty file for consistency
            neg_path = os.path.join(out_dir, f"{name}_neg.csv")
            pd.DataFrame(columns=["u", "i", "ts", "label", "weight", "idx"]).to_csv(neg_path, index=False)
            return

        # Build negatives per positive by sampling dst and pairing with original src
        total_negs = len(pos_df) * max(1, int(neg_per_pos))
        dst_fake = sample_neg(sampler, total_negs)
        # Repeat each src/ts/weight/idx accordingly
        src_rep = np.repeat(pos_df.u.values, neg_per_pos)
        ts_rep = np.repeat(pos_df.ts.values, neg_per_pos)
        idx_base = pos_df.idx.values.max() if len(pos_df) > 0 else 0
        # Assign new idx for negatives following positives in that split
        neg_idx = np.arange(idx_base + 1, idx_base + 1 + total_negs)
        neg_df = pd.DataFrame({
            "u": src_rep,
            "i": dst_fake,
            "ts": ts_rep,
            "label": np.zeros(total_negs, dtype=int),  # 0 denotes negative link; for link_sign treat as class 2 during usage
            "weight": np.zeros(total_negs, dtype=float),
            "idx": neg_idx,
        })
        neg_df.to_csv(os.path.join(out_dir, f"{name}_neg.csv"), index=False)

    export_for_split("train", "train_mask", train_sampler)
    export_for_split("val", "val_mask", val_sampler)
    export_for_split("test", "test_mask", test_sampler)
    # Only canonical names for negatives
    export_for_split("transductive", "test_seen_mask", trans_sampler)
    export_for_split("inductive", "test_unseen_mask", induc_sampler)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    g_df = load_dataset_csv(args.data)

    # Compute masks according to main.py logic
    mask_nodes_override = None
    if args.mask_nodes_path and os.path.exists(args.mask_nodes_path):
        mask_nodes_override = pd.read_csv(args.mask_nodes_path).iloc[:,0].values
    masks = compute_splits(g_df, seed=args.seed, val_q=args.val_quantile, test_q=args.test_quantile, mask_frac=args.mask_frac, mask_nodes_override=mask_nodes_override)

    # Output directory
    if args.out_dir is None:
        here = os.path.dirname(__file__)
        root = os.path.dirname(here)
        default_out = os.path.join(root, "DynamicData", "splits", f"{args.data}_seed{args.seed}")
        out_dir = default_out
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Export positive splits
    export_positives(g_df, masks, out_dir)

    # Export negatives if requested
    if args.export_neg:
        export_negatives(g_df, masks, out_dir, neg_method=args.neg_method, neg_per_pos=args.neg_per_pos)

    # Write metadata for reproducibility
    meta = {
        "seed": args.seed,
        "val_quantile": args.val_quantile,
        "test_quantile": args.test_quantile,
        "mask_frac": args.mask_frac,
        "val_time": float(masks["val_time"][0]),
        "test_time": float(masks["test_time"][0]),
        "num_mask_nodes": int(len(masks["mask_nodes"]))
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_mask_nodes:
        pd.DataFrame({"node": masks["mask_nodes"]}).to_csv(os.path.join(out_dir, "mask_nodes.csv"), index=False)


if __name__ == "__main__":
    main()
