# import os
# import torch
# import os.path as osp

# from typing import Optional, Callable

# from torch_geometric.data import InMemoryDataset, Data, download_url, extract_gz, TemporalData
# from datetime import datetime

# class tgn_wikirfa(InMemoryDataset):

#     def __init__(self, root: str, edge_window_size: int = 10,
#                  name = 'wikirfa',
#                  transform: Optional[Callable] = None,
#                  pre_transform: Optional[Callable] = None):
#         self.edge_window_size = edge_window_size
#         self.name = name
#         if self.name == 'wikirfa':
#             self.url = 'https://snap.stanford.edu/data/wiki-RfA.txt.gz'

#         super().__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
        
#     @property
#     def raw_file_names(self) -> str:
#         if self.name == 'wikirfa':
#             # return 'wiki-RfA.txt'
#             return 'wikirfa.csv'
        
#     @property
#     def processed_file_names(self) -> str:
#         return 'data.pt'

#     @property
#     def num_nodes(self) -> int:
#         return self.data.edge_index.max().item() + 1

#     def download(self):
#         path = download_url(self.url, self.raw_dir)
#         print(path)
#         extract_gz(path, self.raw_dir)
#         os.unlink(path)

#     def process(self):
#         with open(self.raw_paths[0], 'r') as f:
#             data = f.read().split('\n')
#             data = [data[i:i + 7] for i in range(0, len(data), 8)][:-1]            
#             data = [line for line in data if line[5][4:] != '']   #Removing dates with no entries
#             data = [line for line in data if line[2][4:] != '0']  #Removing null votes
            
#             #Sample entry
#             #['SRC:Steel1943', 'TGT:BDD', 'VOT:1', 'RES:1', 'YEA:2013', 'DAT:23:13, 19 April 2013', "TXT:abc"]
#             temp_data = []
#             for line in data:
#                 try:
#                     line[0] = line[0][4:]   #Removing 'SRC' from 'SRC:Steel1943'
#                     line[1] = line[1][4:]   #Removing 'DST' from 'DST:BDD'
#                     line[2] = line[2][4:]   #Removing 'VOT' from 'VOT:1'
                    
#                     line[5] = (datetime.strptime(line[5][4:], "%H:%M, %d %B %Y") - datetime(1970, 1, 1)).total_seconds() #Converting time to total seconds from origin
                    
#                     temp_data.append(line)
#                 except:
#                     pass
            
#             data = temp_data

#             signs = [int(line[2]) for line in data]              
#             edge_index = [[line[0], line[1]] for line in data]
#             node_names = set()
            
#             #Mapping node names to integers
#             for src, dst in edge_index:
#                 node_names.add(src)
#                 node_names.add(dst)

#             nodes = list(range(len(node_names)))
#             mapping = {}
#             for node, name in zip(nodes, node_names):
#                 mapping[name] = node

#             for edge_id in range(len(edge_index)):
#                 edge_index[edge_id][0] = mapping[edge_index[edge_id][0]]
#                 edge_index[edge_id][1] = mapping[edge_index[edge_id][1]]

#             edge_index = torch.tensor(edge_index, dtype=torch.long).t()
#             edge_index = edge_index - edge_index.min()
            
# #           edge_attr = torch.tensor(edge_attr, dtype=torch.long)
#             signs = torch.tensor(signs, dtype=torch.long)
#             signs = signs > 0
#             signs = signs * 1

#             stamps_raw = [int(line[5]) for line in data]
#             t = torch.tensor(stamps_raw).to(torch.long)
#             t_sorted, ix = t.sort(descending=True) #Sort by descending, since the earliest date has most seconds passed

#             edge_index = edge_index[:, ix]
#             signs = signs[ix]
            
#             src = edge_index[0]
#             dst = edge_index[1]
#             dst += int(src.max()) + 1
#             msg = torch.ones(src.size(0), 1) #Set to 1, maybe changed to some edge weight
#             t = t_sorted
#             y = signs
            
#             assert sorted(t.cpu().tolist(), reverse=True) == t.cpu().tolist()
            
# #             print(edge_index.size())
# #             print(signs.size())
# #             print(t.size())
# #             print(msg.size())

#         data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
# #         data.mapping = mapping

#         if self.pre_transform is not None:
#             data = self.pre_transform(data)

#         torch.save(self.collate([data]), self.processed_paths[0])


import os
import torch
import os.path as osp

from typing import Optional, Callable
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from torch_geometric.data import TemporalData
from datetime import datetime

class tgn_wikirfa(InMemoryDataset):
    def __init__(self, root: str, edge_window_size: int = 10,
                 name='wikirfa',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        if self.name == 'wikirfa':
            self.url = 'https://snap.stanford.edu/data/wiki-RfA.txt.gz'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        if self.name == 'wikirfa':
            # 原始解压后文件名（你可以用 txt 或 csv）
            return 'wiki-RfA.txt'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            raw_lines = f.read().split('\n')
            # 按照每 8 行为一个记录分段
            data_blocks = [raw_lines[i:i + 7] for i in range(0, len(raw_lines), 8)][:-1]

            # 过滤无效数据（无日期或投票为0）
            valid_blocks = [
                block for block in data_blocks
                if len(block) >= 6 and block[5][4:] != '' and block[2][4:] != '0'
            ]

            parsed_data = []
            for line in valid_blocks:
                try:
                    src = line[0][4:]  # Remove 'SRC:'
                    tgt = line[1][4:]  # Remove 'TGT:'
                    vot = int(line[2][4:])  # Remove 'VOT:' and convert
                    timestamp = datetime.strptime(line[5][4:], "%H:%M, %d %B %Y")
                    timestamp_sec = int((timestamp - datetime(1970, 1, 1)).total_seconds())
                    parsed_data.append((src, tgt, vot, timestamp_sec))
                except Exception:
                    continue

        # 构造节点编号映射
        edge_list = [(src, tgt) for src, tgt, _, _ in parsed_data]
        node_names = set([n for edge in edge_list for n in edge])
        name2id = {name: i for i, name in enumerate(node_names)}

        # 编码边和标签
        edge_index = [[name2id[src], name2id[tgt]] for src, tgt, _, _ in parsed_data]
        y = torch.tensor([vot for _, _, vot, _ in parsed_data], dtype=torch.long)
        msg = y.clone().unsqueeze(1).float()
        t = torch.tensor([ts for _, _, _, ts in parsed_data], dtype=torch.long)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_index = edge_index - edge_index.min()

        # 时间倒序排序
        t_sorted, idx = t.sort(descending=True)
        edge_index = edge_index[:, idx]
        y = y[idx]
        msg = msg[idx]
        t = t_sorted

        # dst 偏移，避免 TGN 中 src 与 dst 冲突
        src = edge_index[0]
        dst = edge_index[1] + int(src.max()) + 1

        # 构建 TemporalData 对象
        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
if __name__ == "__main__":
    print("🔍 加载 Wiki-RfA 数据集中...")

    dataset = tgn_wikirfa(root='data/wikirfa')
    data = dataset[0]

    print("✅ 数据加载完成")
    print(f"边数: {data.src.size(0)}")
    print(f"src 示例: {data.src[:5].tolist()}")
    print(f"dst 示例: {data.dst[:5].tolist()}")
    print(f"time 示例: {data.t[:5].tolist()}")
    print(f"msg (VOT 投票): {data.msg[:5].view(-1).tolist()}")
    print(f"y   (标签): {data.y[:5].tolist()}")