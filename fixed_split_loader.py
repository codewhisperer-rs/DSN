"""
统一数据集划分加载器
一套数据，支持PolarDSN和SEMBA两种使用方式
"""

import pickle as pkl
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os


class UnifiedSplitLoader:
    """加载统一数据集划分的接口"""
    
    def __init__(self, splits_root='/home/houyikang/sign/DSN/fixed_splits'):
        self.splits_root = Path(splits_root)
    
    def load_split(self, dataset_name):
        """
        加载统一格式的数据划分
        返回: (g_df, splits_dict)
        """
        split_file = self.splits_root / dataset_name / 'unified_split.pkl'
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'rb') as f:
            split_data = pkl.load(f)
        
        return split_data['data'], split_data['splits']
    
    def load_for_polardsn(self, dataset_name):
        """
        为PolarDSN加载数据（直接返回原始格式）
        """
        return self.load_split(dataset_name)
    
    def load_for_semba(self, dataset_name, device='cpu'):
        """
        为SEMBA加载数据（转换为其期望的格式）
        """
        g_df, splits = self.load_split(dataset_name)
        
        # 检查节点ID是否从0开始，如果不是则需要重新映射
        all_nodes = set(g_df.u.tolist() + g_df.i.tolist())
        min_node = min(all_nodes)
        max_node = max(all_nodes)
        
        # 如果节点不是从0开始，需要重新映射
        if min_node != 0:
            print(f"[INFO] Remapping nodes from range [{min_node}, {max_node}] to [0, {len(all_nodes)-1}]")
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_nodes))}
            
            # 重新映射DataFrame中的节点ID
            g_df_mapped = g_df.copy()
            g_df_mapped['u'] = g_df_mapped['u'].map(node_mapping)
            g_df_mapped['i'] = g_df_mapped['i'].map(node_mapping)
        else:
            g_df_mapped = g_df
        
        # 创建SEMBA兼容的数据类
        class SembaData:
            def __init__(self, src, dst, t, y, msg, num_nodes, num_events):
                self.src = torch.tensor(src, dtype=torch.long, device=device)
                self.dst = torch.tensor(dst, dtype=torch.long, device=device)
                self.t = torch.tensor(t, dtype=torch.long, device=device)  # 使用long类型
                self.y = torch.tensor(y, dtype=torch.long, device=device)
                # 确保msg是二维的
                if len(msg.shape) == 1:
                    msg = msg.reshape(-1, 1)
                self.msg = torch.tensor(msg, dtype=torch.float, device=device)
                self.num_nodes = num_nodes
                self.num_events = num_events
            
            def __getitem__(self, idx):
                """支持索引操作"""
                return SembaData(
                    src=self.src[idx].cpu().numpy(),
                    dst=self.dst[idx].cpu().numpy(),
                    t=self.t[idx].cpu().numpy(),
                    y=self.y[idx].cpu().numpy(),
                    msg=self.msg[idx].cpu().numpy(),
                    num_nodes=self.num_nodes,
                    num_events=len(self.src[idx])
                )
            
            def seq_batches(self, batch_size=128):
                """批处理迭代器"""
                curr_idx = 0
                while curr_idx < len(self.src):
                    yield self[curr_idx:curr_idx+batch_size]
                    curr_idx += batch_size
        
        def create_data_subset(flag):
            subset_df = g_df_mapped[flag]
            # 转换标签：将-1转换为0，保持1不变
            labels = subset_df.label.values
            labels = np.where(labels == -1, 0, labels)
            
            # 确保时间是整数类型
            times = subset_df.ts.values.astype(np.int64)
            
            # 确保权重是二维的
            weights = subset_df.weight.values
            if len(weights.shape) == 1:
                weights = weights.reshape(-1, 1)
            
            return SembaData(
                src=subset_df.u.values,
                dst=subset_df.i.values,
                t=times,  # 使用整数时间
                y=labels,  # 使用转换后的标签
                msg=weights,  # 使用二维权重
                num_nodes=len(all_nodes),  # 使用实际的节点数量
                num_events=len(subset_df)
            )
        
        # 创建all data
        all_labels = g_df_mapped.label.values
        all_labels = np.where(all_labels == -1, 0, all_labels)
        all_times = g_df_mapped.ts.values.astype(np.int64)
        all_weights = g_df_mapped.weight.values
        if len(all_weights.shape) == 1:
            all_weights = all_weights.reshape(-1, 1)
        
        all_data = SembaData(
            src=g_df_mapped.u.values,
            dst=g_df_mapped.i.values,
            t=all_times,  # 使用整数时间
            y=all_labels,  # 使用转换后的标签
            msg=all_weights,  # 使用二维权重
            num_nodes=len(all_nodes),  # 使用实际的节点数量
            num_events=len(g_df_mapped)
        )
        
        train_data = create_data_subset(splits['train_flag'])
        val_data = create_data_subset(splits['val_flag'])
        test_data = create_data_subset(splits['test_flag'])
        
        return all_data, train_data, val_data, test_data
    
    def list_available_datasets(self):
        """列出所有可用的数据集"""
        if not self.splits_root.exists():
            return []
        
        datasets = []
        for item in self.splits_root.iterdir():
            if item.is_dir() and (item / 'unified_split.pkl').exists():
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def get_split_stats(self, dataset_name):
        """获取数据集划分统计信息"""
        stats_file = self.splits_root / dataset_name / 'split_stats.txt'
        
        if not stats_file.exists():
            return None
        
        with open(stats_file, 'r') as f:
            return f.read()


def create_polardsn_data_arrays(g_df, splits):
    """
    从统一划分创建PolarDSN需要的数据数组
    """
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    sign_l = g_df.label.values
    ts_l = g_df.ts.values
    weight_l = g_df.weight.values
    
    # 使用统一划分的flags
    train_flag = splits['train_flag']
    val_flag = splits['val_flag']
    test_flag = splits['test_flag']
    tr_test_flag = splits['tr_test_flag']
    nn_test_flag = splits['nn_test_flag']
    
    # Train data
    train_src_l = src_l[train_flag]
    train_dst_l = dst_l[train_flag]
    train_ts_l = ts_l[train_flag]
    train_e_idx_l = e_idx_l[train_flag]
    train_label_l = sign_l[train_flag]
    train_weight_l = weight_l[train_flag]
    
    # Val data
    val_src_l = src_l[val_flag]
    val_dst_l = dst_l[val_flag]
    val_ts_l = ts_l[val_flag]
    val_e_idx_l = e_idx_l[val_flag]
    val_label_l = sign_l[val_flag]
    val_weight_l = weight_l[val_flag]
    
    # Test data
    test_src_l = src_l[test_flag]
    test_dst_l = dst_l[test_flag]
    test_ts_l = ts_l[test_flag]
    test_e_idx_l = e_idx_l[test_flag]
    test_label_l = sign_l[test_flag]
    test_weight_l = weight_l[test_flag]
    
    # Transductive test data
    tr_test_src_l = src_l[tr_test_flag]
    tr_test_dst_l = dst_l[tr_test_flag]
    tr_test_ts_l = ts_l[tr_test_flag]
    tr_test_e_idx_l = e_idx_l[tr_test_flag]
    tr_test_label_l = sign_l[tr_test_flag]
    tr_test_weight_l = weight_l[tr_test_flag]
    
    # Inductive test data
    nn_test_src_l = src_l[nn_test_flag]
    nn_test_dst_l = dst_l[nn_test_flag]
    nn_test_ts_l = ts_l[nn_test_flag]
    nn_test_e_idx_l = e_idx_l[nn_test_flag]
    nn_test_label_l = sign_l[nn_test_flag]
    nn_test_weight_l = weight_l[nn_test_flag]
    
    return {
        # Original arrays
        'src_l': src_l, 'dst_l': dst_l, 'e_idx_l': e_idx_l,
        'sign_l': sign_l, 'ts_l': ts_l, 'weight_l': weight_l,
        
        # Train
        'train_src_l': train_src_l, 'train_dst_l': train_dst_l,
        'train_ts_l': train_ts_l, 'train_e_idx_l': train_e_idx_l,
        'train_label_l': train_label_l, 'train_weight_l': train_weight_l,
        
        # Val
        'val_src_l': val_src_l, 'val_dst_l': val_dst_l,
        'val_ts_l': val_ts_l, 'val_e_idx_l': val_e_idx_l,
        'val_label_l': val_label_l, 'val_weight_l': val_weight_l,
        
        # Test
        'test_src_l': test_src_l, 'test_dst_l': test_dst_l,
        'test_ts_l': test_ts_l, 'test_e_idx_l': test_e_idx_l,
        'test_label_l': test_label_l, 'test_weight_l': test_weight_l,
        
        # Transductive test
        'tr_test_src_l': tr_test_src_l, 'tr_test_dst_l': tr_test_dst_l,
        'tr_test_ts_l': tr_test_ts_l, 'tr_test_e_idx_l': tr_test_e_idx_l,
        'tr_test_label_l': tr_test_label_l, 'tr_test_weight_l': tr_test_weight_l,
        
        # Inductive test
        'nn_test_src_l': nn_test_src_l, 'nn_test_dst_l': nn_test_dst_l,
        'nn_test_ts_l': nn_test_ts_l, 'nn_test_e_idx_l': nn_test_e_idx_l,
        'nn_test_label_l': nn_test_label_l, 'nn_test_weight_l': nn_test_weight_l,
        
        # Split info
        'train_node_set': splits['train_node_set'],
        'mask_node_set': splits['mask_node_set'],
        'new_node_set': splits['new_node_set'],
        'num_total_unique_nodes': splits['num_total_unique_nodes']
    }


# 便利函数供外部使用
def load_polardsn_fixed_split(dataset_name, splits_root=None):
    """便利函数：加载PolarDSN格式的统一划分数据"""
    if splits_root is None:
        splits_root = '/home/houyikang/sign/DSN/fixed_splits'
    
    loader = UnifiedSplitLoader(splits_root)
    g_df, splits = loader.load_for_polardsn(dataset_name)
    data_arrays = create_polardsn_data_arrays(g_df, splits)
    
    return g_df, splits, data_arrays


def load_semba_fixed_split(dataset_name, device='cpu', splits_root=None):
    """便利函数：加载SEMBA格式的统一划分数据"""
    if splits_root is None:
        splits_root = '/home/houyikang/sign/DSN/fixed_splits'
    
    loader = UnifiedSplitLoader(splits_root)
    return loader.load_for_semba(dataset_name, device)
