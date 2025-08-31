"""
固定数据集划分加载器
支持semba和polardsn使用相同的数据集划分
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# 尝试导入torch和torch_geometric，如果没有安装则使用简化版本
try:
    import torch
    from torch_geometric.data import TemporalData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: torch或torch_geometric未安装，将使用简化版本")


class FixedSplitLoader:
    """固定数据集划分加载器"""
    
    def __init__(self, fixed_splits_dir: str = "./fixed_splits"):
        """
        初始化加载器
        
        Args:
            fixed_splits_dir: 固定划分文件存储目录
        """
        self.fixed_splits_dir = fixed_splits_dir
        self.supported_datasets = ["bitcoinalpha", "bitcoinotc", "epinions", "wiki-RfA"]
    
    def load_split_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载固定划分数据
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            包含数据和划分信息的字典
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {self.supported_datasets}")
        
        split_file = os.path.join(self.fixed_splits_dir, dataset_name, "unified_split.pkl")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"找不到划分文件: {split_file}")
        
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        
        return split_data
    
    def get_semba_format(self, dataset_name: str, device: str = "cpu") -> Tuple[Any, Dict[str, np.ndarray]]:
        """
        获取Semba格式的数据
        
        Args:
            dataset_name: 数据集名称
            device: 设备 ("cpu" 或 "cuda")
            
        Returns:
            (TemporalData对象或数据字典, 划分索引字典)
        """
        split_data = self.load_split_data(dataset_name)
        
        # 获取原始数据
        df = split_data['data']
        splits = split_data['splits']
        
        # 确保数据按时间排序
        df = df.sort_values(['ts', 'idx']).reset_index(drop=True)
        
        if TORCH_AVAILABLE:
            # 转换为0-based节点ID
            u = torch.tensor(df['u'].values, dtype=torch.long)
            i = torch.tensor(df['i'].values, dtype=torch.long)
            
            # 统一到0-based ID空间
            base = min(int(u.min()), int(i.min()))
            if base != 0:
                u = u - base
                i = i - base
            
            # 时间和特征
            t = torch.tensor(df['ts'].values, dtype=torch.long)
            
            # 边权重作为消息特征
            msg = torch.tensor(df['weight'].values, dtype=torch.float32).unsqueeze(1)
            
            # 标签 (positive=1, negative=0)
            y = torch.tensor((df['label'].values > 0).astype(int), dtype=torch.long)
            
            # 为了与TGN兼容，dst需要偏移避免与src冲突
            src = u
            dst = i + int(src.max()) + 1
            
            # 创建TemporalData对象
            data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y).to(device)
        else:
            # 简化版本，返回numpy数组字典
            u = df['u'].values
            i = df['i'].values
            
            # 统一到0-based ID空间
            base = min(u.min(), i.min())
            if base != 0:
                u = u - base
                i = i - base
            
            # 为了与TGN兼容，dst需要偏移避免与src冲突
            src = u
            dst = i + int(src.max()) + 1
            
            data = {
                'src': src,
                'dst': dst,
                't': df['ts'].values,
                'msg': df['weight'].values.reshape(-1, 1),
                'y': (df['label'].values > 0).astype(int)
            }
        
        # 准备划分信息 - 使用固定的划分标志
        train_flag = splits['train_flag']
        val_flag = splits['val_flag'] 
        test_flag = splits['test_flag']
        
        # 转换为索引
        train_indices = np.where(train_flag)[0]
        val_indices = np.where(val_flag)[0]
        test_indices = np.where(test_flag)[0]
        
        split_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'val_time': splits['val_time'],
            'test_time': splits['test_time']
        }
        
        return data, split_indices
    
    def get_polardsn_format(self, dataset_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        获取PolarDSN格式的数据
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            (数据字典, 划分信息字典)
        """
        split_data = self.load_split_data(dataset_name)
        
        # 获取原始数据
        df = split_data['data']
        splits = split_data['splits']
        
        # 确保数据按时间排序
        df = df.sort_values(['ts', 'idx']).reset_index(drop=True)
        
        # 获取划分标志
        train_flag = splits['train_flag']
        val_flag = splits['val_flag']
        test_flag = splits['test_flag']
        
        # 全部数据
        all_data = {
            'src_l': df['u'].values,
            'dst_l': df['i'].values,
            'ts_l': df['ts'].values,
            'e_idx_l': df['idx'].values,
            'sign_l': df['label'].values,
            'weight_l': df['weight'].values
        }
        
        # 训练集数据
        train_data = {
            'train_src_l': df.loc[train_flag, 'u'].values,
            'train_dst_l': df.loc[train_flag, 'i'].values,
            'train_ts_l': df.loc[train_flag, 'ts'].values,
            'train_e_idx_l': df.loc[train_flag, 'idx'].values,
            'train_label_l': df.loc[train_flag, 'label'].values,
            'train_weight_l': df.loc[train_flag, 'weight'].values
        }
        
        # 验证集数据
        val_data = {
            'val_src_l': df.loc[val_flag, 'u'].values,
            'val_dst_l': df.loc[val_flag, 'i'].values,
            'val_ts_l': df.loc[val_flag, 'ts'].values,
            'val_e_idx_l': df.loc[val_flag, 'idx'].values,
            'val_label_l': df.loc[val_flag, 'label'].values,
            'val_weight_l': df.loc[val_flag, 'weight'].values
        }
        
        # 测试集数据
        test_data = {
            'test_src_l': df.loc[test_flag, 'u'].values,
            'test_dst_l': df.loc[test_flag, 'i'].values,
            'test_ts_l': df.loc[test_flag, 'ts'].values,
            'test_e_idx_l': df.loc[test_flag, 'idx'].values,
            'test_label_l': df.loc[test_flag, 'label'].values,
            'test_weight_l': df.loc[test_flag, 'weight'].values
        }
        
        # 合并所有数据
        polardsn_data = {**all_data, **train_data, **val_data, **test_data}
        
        # 统计信息
        total_node_set = set(np.unique(np.hstack([df['u'].values, df['i'].values])))
        train_node_set = set(train_data['train_src_l']).union(set(train_data['train_dst_l']))
        
        split_info = {
            'val_time': splits['val_time'],
            'test_time': splits['test_time'],
            'total_node_set': total_node_set,
            'train_node_set': train_node_set,
            'num_total_unique_nodes': len(total_node_set),
            'max_idx': max(df['u'].max(), df['i'].max()),
            'train_flag': train_flag,
            'val_flag': val_flag,
            'test_flag': test_flag,
            'seed': splits['seed']
        }
        
        return polardsn_data, split_info


def load_fixed_split_for_semba(dataset_name: str, device: str = "cpu", 
                              fixed_splits_dir: str = "./fixed_splits") -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    为Semba加载固定划分的数据集
    
    Args:
        dataset_name: 数据集名称 ("bitcoinalpha", "bitcoinotc", "epinions", "wiki-RfA")
        device: 设备
        fixed_splits_dir: 固定划分目录
        
    Returns:
        (TemporalData对象或数据字典, 划分索引字典)
    """
    loader = FixedSplitLoader(fixed_splits_dir)
    return loader.get_semba_format(dataset_name, device)


def load_fixed_split_for_polardsn(dataset_name: str, 
                                  fixed_splits_dir: str = "./fixed_splits") -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    为PolarDSN加载固定划分的数据集
    
    Args:
        dataset_name: 数据集名称 ("bitcoinalpha", "bitcoinotc", "epinions", "wiki-RfA")
        fixed_splits_dir: 固定划分目录
        
    Returns:
        (数据字典, 划分信息字典)
    """
    loader = FixedSplitLoader(fixed_splits_dir)
    return loader.get_polardsn_format(dataset_name)


if __name__ == "__main__":
    # 测试加载器
    print("测试固定划分加载器...")
    
    for dataset_name in ["bitcoinalpha", "bitcoinotc", "epinions", "wiki-RfA"]:
        print(f"\n=== 测试 {dataset_name} ===")
        
        try:
            # 测试Semba格式
            data, split_indices = load_fixed_split_for_semba(dataset_name)
            if TORCH_AVAILABLE:
                print(f"Semba格式 - 边数: {data.src.size(0)}")
            else:
                print(f"Semba格式 - 边数: {len(data['src'])}")
            print(f"  训练集: {len(split_indices['train'])} 边")
            print(f"  验证集: {len(split_indices['val'])} 边")
            print(f"  测试集: {len(split_indices['test'])} 边")
            
            # 测试PolarDSN格式
            polardsn_data, split_info = load_fixed_split_for_polardsn(dataset_name)
            print(f"PolarDSN格式 - 总边数: {len(polardsn_data['src_l'])}")
            print(f"  训练集: {len(polardsn_data['train_src_l'])} 边")
            print(f"  验证集: {len(polardsn_data['val_src_l'])} 边")
            print(f"  测试集: {len(polardsn_data['test_src_l'])} 边")
            print(f"  总节点数: {split_info['num_total_unique_nodes']}")
            
        except Exception as e:
            print(f"错误: {e}")
