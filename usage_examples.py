#!/usr/bin/env python3
"""
使用固定数据集划分的示例代码

这个文件展示了如何在semba和polardsn中使用fixed_split_loader.py
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
sys.path.append(os.getcwd())

from fixed_split_loader import load_fixed_split_for_semba, load_fixed_split_for_polardsn

def example_semba_usage():
    """展示如何在Semba中使用固定划分"""
    print("=== Semba使用示例 ===")
    
    dataset_name = "bitcoinalpha"
    device = "cpu"
    
    # 加载固定划分的数据
    data, split_indices = load_fixed_split_for_semba(dataset_name, device)
    
    print(f"数据集: {dataset_name}")
    if hasattr(data, 'src'):  # torch版本
        print(f"总边数: {data.src.size(0)}")
        print(f"节点数: {max(data.src.max(), data.dst.max()) + 1}")
    else:  # numpy版本
        print(f"总边数: {len(data['src'])}")
        print(f"节点数: {max(data['src'].max(), data['dst'].max()) + 1}")
    
    print(f"训练集边数: {len(split_indices['train'])}")
    print(f"验证集边数: {len(split_indices['val'])}")
    print(f"测试集边数: {len(split_indices['test'])}")
    print(f"验证时间阈值: {split_indices['val_time']}")
    print(f"测试时间阈值: {split_indices['test_time']}")
    
    # 在Semba中如何使用这些索引
    train_indices = split_indices['train']
    val_indices = split_indices['val']
    test_indices = split_indices['test']
    
    print(f"\\n训练集索引范围: {train_indices.min()} - {train_indices.max()}")
    print(f"验证集索引范围: {val_indices.min()} - {val_indices.max()}")
    print(f"测试集索引范围: {test_indices.min()} - {test_indices.max()}")

def example_polardsn_usage():
    """展示如何在PolarDSN中使用固定划分"""
    print("\\n=== PolarDSN使用示例 ===")
    
    dataset_name = "bitcoinalpha"
    
    # 加载固定划分的数据
    polardsn_data, split_info = load_fixed_split_for_polardsn(dataset_name)
    
    print(f"数据集: {dataset_name}")
    print(f"总边数: {len(polardsn_data['src_l'])}")
    print(f"总节点数: {split_info['num_total_unique_nodes']}")
    print(f"最大节点ID: {split_info['max_idx']}")
    
    # 训练集数据
    print(f"\\n训练集:")
    print(f"  边数: {len(polardsn_data['train_src_l'])}")
    print(f"  时间范围: {polardsn_data['train_ts_l'].min()} - {polardsn_data['train_ts_l'].max()}")
    
    # 验证集数据  
    print(f"验证集:")
    print(f"  边数: {len(polardsn_data['val_src_l'])}")
    print(f"  时间范围: {polardsn_data['val_ts_l'].min()} - {polardsn_data['val_ts_l'].max()}")
    
    # 测试集数据
    print(f"测试集:")
    print(f"  边数: {len(polardsn_data['test_src_l'])}")
    print(f"  时间范围: {polardsn_data['test_ts_l'].min()} - {polardsn_data['test_ts_l'].max()}")
    
    # 时间阈值
    print(f"\\n时间阈值:")
    print(f"  验证时间: {split_info['val_time']}")
    print(f"  测试时间: {split_info['test_time']}")
    print(f"  种子: {split_info['seed']}")

def show_integration_instructions():
    """展示如何集成到现有代码中"""
    print("\\n=== 集成说明 ===")
    
    print("\\n1. 在Semba中集成:")
    print("   修改 semba/utils.py 中的 get_data() 函数:")
    print("   ```python")
    print("   from fixed_split_loader import load_fixed_split_for_semba")
    print("   ")
    print("   def get_data(NAME, path, device, use_fixed_split=True):")
    print("       if use_fixed_split and NAME in ['bitcoinalpha', 'bitcoinotc', 'epinions', 'wiki-RfA']:")
    print("           # 使用固定划分")
    print("           data, split_indices = load_fixed_split_for_semba(NAME, device)")
    print("           return data, split_indices")
    print("       else:")
    print("           # 原有的随机划分逻辑...")
    print("   ```")
    
    print("\\n2. 在PolarDSN中集成:")
    print("   修改 PolarDSN/PolarDSN/main.py:")
    print("   ```python")
    print("   from fixed_split_loader import load_fixed_split_for_polardsn")
    print("   ")
    print("   # 替换原有的数据加载和划分代码")
    print("   if args.use_fixed_split:")
    print("       polardsn_data, split_info = load_fixed_split_for_polardsn(DATA)")
    print("       ")
    print("       # 直接使用划分好的数据")
    print("       src_l = polardsn_data['src_l']")
    print("       dst_l = polardsn_data['dst_l']")
    print("       ts_l = polardsn_data['ts_l']")
    print("       # ... 等等")
    print("       ")
    print("       # 使用预计算的划分")
    print("       train_src_l = polardsn_data['train_src_l']")
    print("       val_src_l = polardsn_data['val_src_l']")
    print("       test_src_l = polardsn_data['test_src_l']")
    print("       # ... 等等")
    print("   else:")
    print("       # 原有的动态划分逻辑...")
    print("   ```")

def compare_splits():
    """比较不同数据集的划分统计"""
    print("\\n=== 划分统计比较 ===")
    
    datasets = ["bitcoinalpha", "bitcoinotc", "epinions", "wiki-RfA"]
    
    print(f"{'数据集':<15} {'总边数':<10} {'训练':<10} {'验证':<10} {'测试':<10} {'节点数':<10}")
    print("-" * 75)
    
    for dataset in datasets:
        try:
            _, split_indices = load_fixed_split_for_semba(dataset)
            _, split_info = load_fixed_split_for_polardsn(dataset)
            
            total_edges = len(split_indices['train']) + len(split_indices['val']) + len(split_indices['test'])
            
            print(f"{dataset:<15} {total_edges:<10} {len(split_indices['train']):<10} "
                  f"{len(split_indices['val']):<10} {len(split_indices['test']):<10} "
                  f"{split_info['num_total_unique_nodes']:<10}")
        except Exception as e:
            print(f"{dataset:<15} 错误: {e}")

if __name__ == "__main__":
    example_semba_usage()
    example_polardsn_usage()
    show_integration_instructions()
    compare_splits()
