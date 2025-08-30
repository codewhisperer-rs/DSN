import os
import pandas as pd
import torch
from torch_geometric.data import TemporalData

def load_polardsn_csv(csv_dir: str, name: str) -> TemporalData:
    """Load a PolarDSN-formatted CSV into a TemporalData object for Semba.

    Expected columns in CSV: u, i, ts, label, weight, idx
    - u, i: 1-based node ids (will be converted to 0-based)
    - ts: unix timestamp (float or int)
    - label: signed label in {-1, 1}
    - weight: edge weight (float)
    - idx: edge id (ignored here but must exist)

    The data is sorted by timestamp ascending for sequential processing.
    """

    # Resolve file path and load only needed columns to avoid stray index cols
    fname = f"ml_{name}.csv"
    fpath = os.path.join(csv_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"PolarDSN CSV not found: {fpath}")

    df = pd.read_csv(fpath, usecols=["u", "i", "ts", "label", "weight", "idx"])  # type: ignore

    # Ensure deterministic order by time then idx
    df = df.sort_values(["ts", "idx"]).reset_index(drop=True)

    # Convert 1-based -> 0-based ids for PyG consistency
    u = torch.tensor(df["u"].values, dtype=torch.long)
    i = torch.tensor(df["i"].values, dtype=torch.long)
    # Shift to a single 0-based id space
    base = min(int(u.min()), int(i.min()))
    if base != 0:
        u = u - base
        i = i - base

    # Time and features
    t = torch.tensor(df["ts"].astype(int).values, dtype=torch.long)
    # msg: edge weight as a 1D feature
    msg = torch.tensor(df["weight"].values, dtype=torch.float32).unsqueeze(1)
    # y: binary label expected by Semba (pos=1, neg=0)
    y = torch.tensor((df["label"].values > 0).astype(int), dtype=torch.long)

    data = TemporalData(src=u, dst=i, t=t, msg=msg, y=y)
    return data
