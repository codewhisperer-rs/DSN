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
        
    def simple_time_split(self, g_df):
        """
        简单的基于时间的train/val/test划分
        """
        # 设置随机种子确保可重现
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        ts_l = g_df.ts.values
        
        # 使用时间分位点划分 (70% train, 15% val, 15% test)
        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
        
        # 基于时间的简单划分
        train_flag = ts_l <= val_time
        val_flag = (ts_l > val_time) & (ts_l <= test_time)  
        test_flag = ts_l > test_time
        
        # 获取所有节点统计信息
        total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
        num_total_unique_nodes = len(total_node_set)
        
        splits = {
            'train_flag': train_flag,
            'val_flag': val_flag,
            'test_flag': test_flag,
            'val_time': val_time,
            'test_time': test_time,
            'num_total_unique_nodes': num_total_unique_nodes,
            'seed': self.seed
        }
        
        return splits
    
    def create_splits_for_dataset(self, dataset_name, csv_path, output_dir):
        """为单个数据集创建统一划分"""
        print(f"Processing {dataset_name}...")
        
        # 读取CSV
        g_df = pd.read_csv(csv_path)
        print(f"  Loaded {len(g_df)} edges")
        
        # 执行简单时间划分
        splits = self.simple_time_split(g_df)
        
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
            f.write(f"Val time threshold: {splits['val_time']}\n")
            f.write(f"Test time threshold: {splits['test_time']}\n")
            
        print(f"  Saved unified split to {dataset_output_dir}")
        return stats
    
    def compute_split_stats(self, g_df, splits):
        """计算划分统计信息"""
        return {
            'train_edges': splits['train_flag'].sum(),
            'val_edges': splits['val_flag'].sum(),
            'test_edges': splits['test_flag'].sum()
        }


def main():
    parser = argparse.ArgumentParser(description='Create fixed dataset splits')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/houyikang/DSN/PolarDSN/DynamicData/weight',
                       help='Directory containing weight CSV files')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/houyikang/DSN/fixed_splits',
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
