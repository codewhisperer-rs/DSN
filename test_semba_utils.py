"""
测试更新后的 semba/utils.py 中的 get_data 函数
"""

import sys
import os
sys.path.append('/home/houyikang/DSN/semba')

def test_new_get_data():
    """测试新的get_data函数"""
    try:
        from utils import get_data
        print("✅ 成功导入 get_data 函数")
        
        # 测试固定划分
        print("\n=== 测试固定划分 ===")
        data, train_data, val_data, test_data = get_data(
            NAME='bitcoinalpha',
            device='cpu',
            use_fixed_split=True
        )
        
        print(f"数据加载成功:")
        if hasattr(data, 'src'):
            print(f"  完整数据: {data.src.size(0)} 条边")
            print(f"  训练数据: {train_data.src.size(0)} 条边")
            print(f"  验证数据: {val_data.src.size(0)} 条边")
            print(f"  测试数据: {test_data.src.size(0)} 条边")
        else:
            print(f"  完整数据: {len(data.src)} 条边")
            print(f"  训练数据: {len(train_data.src)} 条边")
            print(f"  验证数据: {len(val_data.src)} 条边")
            print(f"  测试数据: {len(test_data.src)} 条边")
            
        # 测试兼容性（原有格式）
        print("\n=== 测试兼容性（原有格式） ===")
        try:
            data2, train_data2, val_data2, test_data2 = get_data(
                NAME='BitcoinAlpha-1',  # 原有格式
                device='cpu',
                use_fixed_split=True
            )
            print("✅ 原有格式兼容性测试通过")
        except Exception as e:
            print(f"⚠️ 原有格式测试失败: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """测试数据一致性"""
    print("\n=== 测试数据一致性 ===")
    
    try:
        from utils import get_data
        
        # 多次加载同一数据集，验证固定划分的一致性
        results = []
        for i in range(3):
            data, train_data, val_data, test_data = get_data(
                NAME='bitcoinalpha',
                device='cpu',
                use_fixed_split=True
            )
            
            if hasattr(train_data, 'src'):
                train_size = train_data.src.size(0)
                val_size = val_data.src.size(0)
                test_size = test_data.src.size(0)
            else:
                train_size = len(train_data.src)
                val_size = len(val_data.src)
                test_size = len(test_data.src)
            
            results.append((train_size, val_size, test_size))
            print(f"第{i+1}次: 训练={train_size}, 验证={val_size}, 测试={test_size}")
        
        # 检查一致性
        if len(set(results)) == 1:
            print("✅ 固定划分一致性测试通过")
            return True
        else:
            print("❌ 固定划分不一致")
            return False
            
    except Exception as e:
        print(f"❌ 一致性测试失败: {e}")
        return False

def show_usage_examples():
    """展示使用示例"""
    print("\n=== 使用示例 ===")
    
    print("1. 使用固定划分（推荐）:")
    print("```python")
    print("from utils import get_data")
    print("data, train, val, test = get_data('bitcoinalpha', device='cpu')")
    print("```")
    
    print("\n2. 强制使用随机划分:")
    print("```python")
    print("data, train, val, test = get_data('bitcoinalpha', device='cpu', use_fixed_split=False)")
    print("```")
    
    print("\n3. 兼容原有格式:")
    print("```python")
    print("data, train, val, test = get_data('BitcoinAlpha-1', path='data/', device='cpu')")
    print("```")
    
    print("\n4. 支持的数据集:")
    print("  - 固定划分: 'bitcoinalpha', 'bitcoinotc', 'epinions', 'wiki-RfA'")
    print("  - 原有格式: 'BitcoinOTC-1', 'BitcoinAlpha-1', 'wikirfa'")

if __name__ == "__main__":
    print("🧪 测试更新后的 semba get_data 函数")
    
    # 运行测试
    test1 = test_new_get_data()
    test2 = test_data_consistency()
    
    if test1 and test2:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查实现")
    
    show_usage_examples()
