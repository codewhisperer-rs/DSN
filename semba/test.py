import pickle

# 替换成你自己的文件路径
pkl_path = '/home/houyikang/sign_link/semba/splits/wikirfa/split_0.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 打印对象类型
print("数据类型:", type(data))

# 如果是字典，打印前几个键值
if isinstance(data, dict):
    for k in list(data.keys())[:5]:
        print(f"{k} : {type(data[k])}")

# 如果是列表或其他序列类型
elif isinstance(data, (list, tuple)):
    print("长度:", len(data))
    for i, item in enumerate(data[:5]):
        print(f"[{i}] 类型: {type(item)}")

# 如果是 PyTorch Tensor 数据
try:
    import torch
    if isinstance(data, torch.Tensor):
        print("Tensor 形状:", data.shape)
except ImportError:
    pass
from torch_geometric.data.temporal import TemporalData

td = data["all"]  # 或 val/test/all

print("ALL 是 TemporalData")
print("src:", td.src.shape)
print("dst:", td.dst.shape)
print("edge_time:", td.t.shape if hasattr(td, 't') else td.edge_time.shape)
print("msg:", td.msg.shape if td.msg is not None else "无")
print("y:", td.y.shape if td.y is not None else "无")

td = data["train"]  # 或 val/test/all

for i in range(10,100):
    print(f"\n第 {i} 条边：")
    print(f"src: {td.src[i].item()}")
    print(f"dst: {td.dst[i].item()}")
    print(f"time: {td.t[i].item() if hasattr(td, 't') else td.edge_time[i].item()}")
    print(f"msg: {td.msg[i].item()}")
    print(f"y: {td.y[i].item()}")
