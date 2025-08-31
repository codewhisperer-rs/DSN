# 固定数据集划分加载器

这个项目提供了一个统一的数据集加载器，让semba和polardsn可以使用相同的数据集划分，确保实验结果的可重现性。

## 文件说明

- `fixed_split_loader.py`: 主要的加载器实现
- `usage_examples.py`: 使用示例
- `fixed_splits/`: 存储预计算的数据集划分

## 支持的数据集

- `bitcoinalpha`: Bitcoin Alpha信任网络
- `bitcoinotc`: Bitcoin OTC信任网络  
- `epinions`: Epinions评分网络
- `wiki-RfA`: Wikipedia Request for Adminship网络

## 快速开始

### 1. 基本使用

```python
from fixed_split_loader import load_fixed_split_for_semba, load_fixed_split_for_polardsn

# 为Semba加载数据
data, split_indices = load_fixed_split_for_semba("bitcoinalpha", device="cpu")

# 为PolarDSN加载数据
polardsn_data, split_info = load_fixed_split_for_polardsn("bitcoinalpha")
```

### 2. 在Semba中集成

修改 `semba/utils.py` 中的 `get_data()` 函数：

```python
from fixed_split_loader import load_fixed_split_for_semba

def get_data(NAME, path, device, use_fixed_split=True):
    if use_fixed_split and NAME in ['bitcoinalpha', 'bitcoinotc', 'epinions', 'wiki-RfA']:
        # 使用固定划分
        data, split_indices = load_fixed_split_for_semba(NAME, device)
        
        # 提取训练/验证/测试数据
        train_indices = split_indices['train']
        val_indices = split_indices['val'] 
        test_indices = split_indices['test']
        
        return data, train_indices, val_indices, test_indices
    else:
        # 原有的随机划分逻辑...
```

### 3. 在PolarDSN中集成

修改 `PolarDSN/PolarDSN/main.py`：

```python
# 在文件开头添加导入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fixed_split_loader import load_fixed_split_for_polardsn

# 在参数解析后添加固定划分选项
args, sys_argv = get_args()
# 可以在args中添加use_fixed_split参数

# 替换原有的数据加载代码
if getattr(args, 'use_fixed_split', True):  # 默认使用固定划分
    polardsn_data, split_info = load_fixed_split_for_polardsn(DATA)
    
    # 直接使用划分好的数据
    src_l = polardsn_data['src_l']
    dst_l = polardsn_data['dst_l']
    ts_l = polardsn_data['ts_l']
    e_idx_l = polardsn_data['e_idx_l']
    sign_l = polardsn_data['sign_l']
    weight_l = polardsn_data['weight_l']
    
    # 使用预计算的划分
    train_src_l = polardsn_data['train_src_l']
    train_dst_l = polardsn_data['train_dst_l']
    train_ts_l = polardsn_data['train_ts_l']
    train_e_idx_l = polardsn_data['train_e_idx_l']
    train_label_l = polardsn_data['train_label_l']
    train_weight_l = polardsn_data['train_weight_l']
    
    val_src_l = polardsn_data['val_src_l']
    val_dst_l = polardsn_data['val_dst_l']
    val_ts_l = polardsn_data['val_ts_l']
    val_e_idx_l = polardsn_data['val_e_idx_l']
    val_label_l = polardsn_data['val_label_l']
    val_weight_l = polardsn_data['val_weight_l']
    
    test_src_l = polardsn_data['test_src_l']
    test_dst_l = polardsn_data['test_dst_l']
    test_ts_l = polardsn_data['test_ts_l']
    test_e_idx_l = polardsn_data['test_e_idx_l']
    test_label_l = polardsn_data['test_label_l']
    test_weight_l = polardsn_data['test_weight_l']
    
    # 其他需要的变量
    val_time = split_info['val_time']
    test_time = split_info['test_time']
    total_node_set = split_info['total_node_set']
    num_total_unique_nodes = split_info['num_total_unique_nodes']
    max_idx = split_info['max_idx']
    
    # 删除或注释掉原有的数据加载和划分代码
    # g_df = pd.read_csv('...')
    # val_time, test_time = list(np.quantile(...))
    # ...等等
else:
    # 原有的动态划分逻辑...
```

## API文档

### FixedSplitLoader类

主要的加载器类，提供统一的数据访问接口。

#### 方法

- `load_split_data(dataset_name)`: 加载原始的划分数据
- `get_semba_format(dataset_name, device)`: 获取Semba格式的数据
- `get_polardsn_format(dataset_name)`: 获取PolarDSN格式的数据

### 便利函数

- `load_fixed_split_for_semba(dataset_name, device, fixed_splits_dir)`: 为Semba加载数据
- `load_fixed_split_for_polardsn(dataset_name, fixed_splits_dir)`: 为PolarDSN加载数据

## 数据格式

### Semba格式

返回一个TemporalData对象（如果安装了torch_geometric）或数据字典，包含：
- `src`: 源节点
- `dst`: 目标节点（为避免冲突已偏移）
- `t`: 时间戳
- `msg`: 边权重特征
- `y`: 二进制标签（1为正，0为负）

以及划分索引字典：
- `train`: 训练集索引
- `val`: 验证集索引
- `test`: 测试集索引
- `val_time`: 验证时间阈值
- `test_time`: 测试时间阈值

### PolarDSN格式

返回两个字典：

1. 数据字典包含：
   - 全部数据：`src_l`, `dst_l`, `ts_l`, `e_idx_l`, `sign_l`, `weight_l`
   - 训练集：`train_src_l`, `train_dst_l`, `train_ts_l`, `train_e_idx_l`, `train_label_l`, `train_weight_l`
   - 验证集：`val_src_l`, `val_dst_l`, `val_ts_l`, `val_e_idx_l`, `val_label_l`, `val_weight_l`
   - 测试集：`test_src_l`, `test_dst_l`, `test_ts_l`, `test_e_idx_l`, `test_label_l`, `test_weight_l`

2. 划分信息字典包含：
   - `val_time`, `test_time`: 时间阈值
   - `total_node_set`, `train_node_set`: 节点集合
   - `num_total_unique_nodes`: 总节点数
   - `max_idx`: 最大节点ID
   - `train_flag`, `val_flag`, `test_flag`: 划分标志数组
   - `seed`: 随机种子

## 数据划分统计

| 数据集 | 总边数 | 训练集 | 验证集 | 测试集 | 节点数 |
|--------|--------|--------|--------|--------|--------|
| bitcoinalpha | 24,186 | 16,940 | 3,628 | 3,618 | 3,783 |
| bitcoinotc | 35,592 | 24,914 | 5,339 | 5,339 | 5,881 |
| epinions | 841,372 | 589,082 | 126,150 | 126,140 | 131,828 |
| wiki-RfA | 170,499 | 119,349 | 25,575 | 25,575 | 10,595 |

所有划分都使用70/15/15的比例，基于时间顺序进行划分，使用相同的随机种子(0)确保可重现性。

## 运行示例

```bash
# 测试加载器
python fixed_split_loader.py

# 查看使用示例
python usage_examples.py
```

## 注意事项

1. 确保`fixed_splits/`目录存在且包含所需的数据集文件
2. 如果torch_geometric未安装，Semba格式将返回numpy数组字典而不是TemporalData对象
3. 所有数据集都使用相同的划分策略，确保公平比较
4. 节点ID已经标准化为0-based，适合PyTorch Geometric使用
