```python
#取负样本命令
python export_splits.py -d bitcoinalpha --export_neg --neg_per_pos 1 --seed 0
```


默认1:1采样

如果不需要进行mask，把mask_frac设置为0，mask是为了归纳场景（inductive）
- 可控地保证有足够数量的“新节点”用于归纳评估（仅靠时间切分，有时新节点会很少或不稳定）
- 避免信息泄漏：被 mask 的节点在训练期完全不可见，测试才能真正衡量“新节点泛化”
- 难度可调：mask_frac（默认 0.1）控制新节点比例