import pickle as pkl
import os.path as osp
import os
import torch
import numpy as np
import random
from utils import get_data
from parser import parse_args
import sys

def save_multiple_splits(args, n_splits=5):
    dataset_path = osp.join('./data', args.dataset)
    save_dir = osp.join('./splits', args.dataset)   # 存放目录
    os.makedirs(save_dir, exist_ok=True)

    for split_id in range(n_splits):
        # 修改随机种子
        seed = args.seed + split_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 调用原来的划分函数
        data, train_data, val_data, test_data = get_data(
            args.dataset, dataset_path, args.device, 
            val_ratio=args.val_ratio, test_ratio=args.test_ratio
        )

        # 保存到 pkl
        split_file = osp.join(save_dir, f"split_{split_id}.pkl")
        with open(split_file, "wb") as f:
            pkl.dump({
                "seed": seed,
                "train": train_data,
                "val": val_data,
                "test": test_data,
                "all": data
            }, f)

        print(f"[OK] Saved split {split_id} with seed={seed} to {split_file}")


if __name__ == "__main__":
    # 使用训练脚本同样的参数解析器
    args = parse_args(sys.argv[1:])

    # 如果 parser 里没有 n_splits，就默认 5；若用户通过命令行传了 --n_splits，我们也兼容
    try:
      n_splits = getattr(args, 'n_splits')
      if n_splits is None:
          n_splits = 5
    except Exception:
      n_splits = 5

    print(f"[INFO] Generating {n_splits} data splits for dataset={args.dataset}, val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, seed_base={args.seed}")
    save_multiple_splits(args, n_splits=n_splits)