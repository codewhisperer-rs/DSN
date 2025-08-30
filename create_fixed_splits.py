#!/usr/bin/env python3
"""
创建固定数据集划分脚本
按照PolarDSN的划分方法，为所有数据集创建统一的train/val/test划分
生成的划分文件可以被PolarDSN和SEMBA两个项目使用
"""

import pandas as pd
import numpy as np
import pickle as pkl
import os
import random
import argparse
from pathlib import Path


class FixedDataSplitter:
    def __init__(self, seed=0, val_ratio=0.15, test_ratio=0.15):
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
    def polardsn_split(self, g_df):
        """
        使用PolarDSN的时间划分方法和masked node逻辑
        """
        # 设置随机种子确保可重现
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        src_l = g_df.u.values
        dst_l = g_df.i.values
        e_idx_l = g_df.idx.values
        sign_l = g_df.label.values
        ts_l = g_df.ts.values
        weight_l = g_df.weight.values
        
        # 使用时间分位点划分 (70% train, 15% val, 15% test)
        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
        
        # 获取所有节点
        total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
        num_total_unique_nodes = len(total_node_set)
        
        # 创建masked node set (10%的validation+test时间段的节点用于inductive测试)
        future_nodes = set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))
        mask_node_set = set(random.sample(
            sorted(future_nodes), 
            int(0.1 * num_total_unique_nodes)
        ))
        
        # 创建flags
        mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
        mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
        none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
        
        # Train split: 时间<=val_time 且不包含masked nodes
        train_flag = (ts_l <= val_time) * (none_node_flag > 0)
        
        # Val split: val_time < 时间 <= test_time
        val_flag = (ts_l <= test_time) * (ts_l > val_time)
        
        # Test split: 时间 > test_time
        test_flag = ts_l > test_time
        
        # 获取train nodes
        train_src_l = src_l[train_flag]
        train_dst_l = dst_l[train_flag]
        train_node_set = set(train_src_l).union(train_dst_l)
        
        # 计算new node sets
        new_node_set = total_node_set - train_node_set
        is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
        is_seen_node_edge = np.array([(a in train_node_set and b in train_node_set) for a, b in zip(src_l, dst_l)])
        
        # Transductive and Inductive test splits
        tr_test_flag = test_flag * is_seen_node_edge  # transductive: both nodes seen in train
        nn_test_flag = test_flag * is_new_node_edge   # inductive: at least one new node
        
        splits = {
            'train_flag': train_flag,
            'val_flag': val_flag,
            'test_flag': test_flag,
            'tr_test_flag': tr_test_flag,  # transductive test
            'nn_test_flag': nn_test_flag,  # inductive test
            'mask_node_set': mask_node_set,
            'train_node_set': train_node_set,
            'new_node_set': new_node_set,
            'val_time': val_time,
            'test_time': test_time,
            'num_total_unique_nodes': num_total_unique_nodes
        }
        
        return splits
    
    def create_splits_for_dataset(self, dataset_name, csv_path, output_dir):
        """为单个数据集创建统一划分"""
        print(f"Processing {dataset_name}...")
        
        # 读取CSV
        g_df = pd.read_csv(csv_path)
        print(f"  Loaded {len(g_df)} edges")
        
        # 执行PolarDSN划分
        splits = self.polardsn_split(g_df)
        
        # 创建输出目录
        dataset_output_dir = Path(output_dir) / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存统一的划分信息（供两个项目共同使用）
        unified_split = {
            'dataset_name': dataset_name,
            'seed': self.seed,
            'data': g_df,
            'splits': splits,
            'format_version': 1.0
        }
        
        with open(dataset_output_dir / 'unified_split.pkl', 'wb') as f:
            pkl.dump(unified_split, f)
        
        # 保存统计信息
        stats = self.compute_split_stats(g_df, splits)
        with open(dataset_output_dir / 'split_stats.txt', 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Total edges: {len(g_df)}\n")
            f.write(f"Total nodes: {splits['num_total_unique_nodes']}\n")
            f.write(f"Train edges: {stats['train_edges']}\n")
            f.write(f"Val edges: {stats['val_edges']}\n")
            f.write(f"Test edges: {stats['test_edges']}\n")
            f.write(f"Transductive test edges: {stats['tr_test_edges']}\n")
            f.write(f"Inductive test edges: {stats['nn_test_edges']}\n")
            f.write(f"Masked nodes: {len(splits['mask_node_set'])}\n")
            f.write(f"Train nodes: {len(splits['train_node_set'])}\n")
            f.write(f"New nodes: {len(splits['new_node_set'])}\n")
            
        print(f"  Saved unified split to {dataset_output_dir}")
        return stats
    
    def compute_split_stats(self, g_df, splits):
        """计算划分统计信息"""
        return {
            'train_edges': splits['train_flag'].sum(),
            'val_edges': splits['val_flag'].sum(),
            'test_edges': splits['test_flag'].sum(),
            'tr_test_edges': splits['tr_test_flag'].sum(),
            'nn_test_edges': splits['nn_test_flag'].sum()
        }


def main():
    parser = argparse.ArgumentParser(description='Create fixed dataset splits')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/houyikang/sign/DSN/PolarDSN/DynamicData/weight',
                       help='Directory containing weight CSV files')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/houyikang/sign/DSN/fixed_splits',
                       help='Output directory for splits')
    parser.add_argument('--datasets', nargs='+', 
                       default=['bitcoinalpha', 'bitcoinotc', 'epinions', 'wiki-RfA'],
                       help='Dataset names to process')
    
    args = parser.parse_args()
    
    # 创建分割器
    splitter = FixedDataSplitter(seed=args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    # 处理每个数据集
    for dataset in args.datasets:
        csv_file = Path(args.data_dir) / f'ml_{dataset}.csv'
        if csv_file.exists():
            stats = splitter.create_splits_for_dataset(dataset, csv_file, args.output_dir)
            all_stats[dataset] = stats
        else:
            print(f"Warning: {csv_file} not found, skipping {dataset}")
    
    # 保存总体统计
    summary_file = Path(args.output_dir) / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Fixed Dataset Splits Summary\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Processed datasets: {list(all_stats.keys())}\n\n")
        
        for dataset, stats in all_stats.items():
            f.write(f"{dataset}:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\nAll splits created successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary: {summary_file}")


if __name__ == '__main__':
    main()
