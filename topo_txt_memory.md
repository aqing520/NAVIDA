# Topo Text Memory 实验总结

## 目标

我们尝试在不重新训练模型的情况下，在推理阶段给 Qwen3-VL-4B DAgger 模型加入一个 episode 内的拓扑记忆。

原始输入仍然保留当前的历史图像方案：历史帧 + 当前帧。Topo memory 额外提供一小段文本，告诉模型当前视角在过去轨迹中的拓扑关系，例如是否回到了之前见过的位置、最近路径是否形成回环。

这个实验主要验证：只靠推理阶段维护记忆图，是否能改善历史长程信息利用。

## 记忆设计

### 1. topo_text：抽象拓扑节点记忆

每个 episode 内维护一张临时拓扑图，不跨 episode 共享。

每次需要调用 VLM 决策时：

1. 用 SigLIP 对当前 RGB 视角编码成视觉 embedding。
2. 和已有节点 embedding 计算相似度。
3. 如果相似度超过阈值，则认为当前视角回到了已有节点。
4. 否则创建一个新节点。
5. 如果当前节点和上一次决策节点不同，则记录一条边。
6. 把当前拓扑状态转成短文本，插入到导航 prompt 中。

每个节点保留的信息：

- node id
- 第一次出现的 step
- 最近一次出现的 step
- visit 次数
- SigLIP 图像 embedding

每条边保留的信息：

- 起点 node
- 终点 node
- transition 次数
- 最近一次转移时的动作摘要

插入给模型的文本大致是：

```text
Topological memory: 5 places recorded; current place is 2.
Current view revisits place 2, first seen 18 decisions ago, visits=2, similarity=0.87.
Recent route: place 0 -> place 1 -> place 2 -> place 4 -> place 2.
Avoid repeating the same route unless the instruction requires returning.
```

这版记忆没有语义，只告诉模型“place id”和回环关系。

### 2. topo_semantic_text：尝试语义化节点

后续我们尝试把抽象节点改成语义节点：每个新节点创建时，让 VLM 给当前视角生成一个简短地点描述，再把这个描述作为 node label。

理想形式是：

```text
Semantic topological memory: 4 places recorded.
Current view matches place 2 (open hallway with sofa and doorway), first seen 12 decisions ago.
Recent route: place 0 (start hallway) -> place 1 (room entrance) -> place 2 (open hallway).
```

但实际发现，当前导航模型已经被强烈训练成输出动作，要求它描述图像时，经常仍然输出：

```text
forward 25 cm, turn right 15 degree, forward 25 cm
```

这类文本不是视觉描述，会污染记忆。因此我们加入过滤规则：如果 caption 中包含 `forward / turn / stop / degree / cm` 等动作词，就丢弃该 caption。

过滤后，很多节点没有语义 caption，最终退化成更干净但仍然抽象的 `place 0 / place 1` 记忆。

## 最终结果

### topo_text 阶段性结果

`topo_text` 因为速度较慢，没有跑完整量，停止时完成 265 个 episode。下面是相同 episode id 上的对比：

| 方法 | SR | OSR | SPL | DTG | Path |
|---|---:|---:|---:|---:|---:|
| baseline | 59.62 | 66.04 | 54.24 | 4.626 | 10.187 |
| uniform | 57.74 | 63.40 | 52.34 | 5.165 | 9.977 |
| topo_text | 59.25 | 63.02 | 52.50 | 5.051 | 11.169 |

结论：`topo_text` 没有稳定提升。相比 baseline，SR 接近，但 OSR、SPL、DTG 都更差；路径明显更长。

### topo_semantic_text 全量结果

`topo_semantic_text` 使用 SigLIP server 加速后跑完整量 1839 个 episode。

| 方法 | SR | OSR | SPL | DTG | Path |
|---|---:|---:|---:|---:|---:|
| baseline 11000 | 56.50 | 63.62 | 51.40 | 4.776 | 10.367 |
| uniform 11000 | 57.59 | 64.65 | 52.42 | 4.875 | 10.259 |
| topo_semantic_text | 56.28 | 63.40 | 50.41 | 5.089 | 11.171 |

相对 baseline：

- SR：-0.22
- OSR：-0.22
- SPL：-0.99
- DTG：+0.31
- Path：+0.80

相对 uniform：

- SR：-1.31
- OSR：-1.25
- SPL：-2.01

最终结论：这版 topo memory 没有带来提升，反而略有下降。

## 结果分析

### 1. 抽象 node id 对模型帮助有限

`place 0 / place 1 / place 2` 这种拓扑编号对人来说能表达路径结构，但对当前模型来说很难和 instruction 里的 landmark 对齐。

例如 instruction 说“走到沙发旁边的门”，但 memory 只告诉它“当前是 place 2，之前访问过”，模型并不知道 place 2 是否和“沙发”“门”有关。

所以抽象拓扑关系本身不足以显著改善导航。

### 2. 强策略提示可能带来负面影响

`topo_text` 里加入了类似：

```text
Avoid repeating the same route unless the instruction requires returning.
```

这类提示可能让模型过度避免回头或重复路径。但 VLN 任务中，有时回到之前位置、调整方向、重新观察是合理行为。

结果上也能看到 topo 版本 path 更长，说明它没有形成更高效的导航，反而可能让轨迹更绕。

### 3. 语义化方案受限于 caption 质量

`topo_semantic_text` 的原始想法是正确的：节点应该有可和指令对齐的语义描述。

但当前直接用导航模型生成 caption 不可行。它经常把 caption 任务也回答成动作序列，说明模型的输出分布已经强烈偏向导航动作。

过滤动作 caption 后，虽然避免了错误记忆，但大量节点失去语义，最终没有真正实现“语义拓扑记忆”。

### 4. 训练分布不匹配

当前模型训练时没有见过 topo memory text。推理时突然加入额外文本，会造成 prompt 分布变化。

即使 memory 内容本身是合理的，模型也未必知道如何利用它。尤其是这种结构化文本和视觉历史混合输入，可能需要在 SFT 或 DAgger 阶段一起训练。

### 5. 当前实验的主要价值

这组实验说明：

- 推理阶段维护 episode 内拓扑图是可行的。
- SigLIP 相似度可以用于判断视觉 revisit。
- 仅加入抽象 topo text 不足以提升性能。
- 直接用导航模型生成节点语义 caption 不可靠。
- 如果继续做 topo memory，需要把“语义节点”和“训练对齐”一起考虑。

## 后续判断

简单的 eval-only topo text 方案意义已经不大。后面如果继续做记忆图，更值得做的是：

1. 使用独立 caption / VLM 模型生成稳定的视觉 landmark 描述，而不是让导航模型自己 caption。
2. 去掉强策略性提示，只提供事实型记忆。
3. 在 SFT / DAgger 训练样本中同步加入同格式 memory text，让模型学习如何使用记忆。
4. 更进一步，可以从文本拓扑图扩展到空间图记忆，例如维护节点、边、相对方向、可通行关系，再和导航决策联合训练。
