# Qwen3-VL-4B V2 DAgger

## 概览

- 目标：
  - 基于第二版最好模型 `models/qwen3vl4b_ckpt12765`
  - 额外加入一轮 `episode-level` DAgger 数据
  - 观察在 `R2R + RxR` 混合训练下是否还能继续提升
- 相关说明：
  - 详细采集链路见 `dagger.md`
  - 本文件只记录这次实际数据落盘和训练情况

## DAgger 数据收集

- 基础模型：
  - `models/qwen3vl4b_ckpt12765`
- collector：
  - `src/data/dagger_collect_episode.py`
- prepare：
  - `src/data/dagger_prepare_episode.py`
- 主要输出目录：
  - `data/dagger/r2r_dagger_v1`
  - `data/dagger/rxr_dagger_v1`

### prepared 数据规模

- `R2R`：
  - `data/dagger/r2r_dagger_v1/prepared/train_full.jsonl`
  - `311,604` 条
- `RxR`：
  - `data/dagger/rxr_dagger_v1/prepared/train_full.jsonl`
  - `603,657` 条

### 最终混合训练集

- 原始训练集：
  - `data/train_r2r_rxr_qwen3vl4b_full.jsonl`
  - `816,897` 条
- 加入 DAgger 后：
  - `data/train_r2r_rxr_qwen3vl4b_full_with_dagger.jsonl`
  - `1,732,158` 条
- 增量：
  - `+915,261` 条
  - 约为原始训练集的 `2.12x`

## DAgger 训练

- 脚本：
  - `scripts/train_qwen3_r2r_rxr_dagger.sh`
- 环境：
  - `navida_wzy`
- 初始化模型：
  - `models/qwen3vl4b_ckpt12765`
- 输出目录：
  - `result/qwen3vl4b_r2r_rxr_dagger_zero2`
- GPU：
  - `1,2,3,4`
- `master_port=25436`

### 主要训练参数

- `num_train_epochs=1`
- `bf16=True`
- `torch_dtype=bfloat16`
- `attn_implementation=flash_attention_2`
- `lr_scheduler_type=cosine`
- `gradient_checkpointing=True`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `global_batch_size = 2 x 8 x 4 = 64`
- `dataloader_num_workers=0`
- `learning_rate=1e-5`
- `logging_steps=5`
- `save_steps=1000`
- `save_total_limit=5`
- `eval_strategy=no`

### 和 V2 base train 的差异

- 训练起点从 `Qwen3-VL-4B-Instruct` 变成：
  - `qwen3vl4b_ckpt12765`
- 数据从原始混合集变成：
  - `train_r2r_rxr_qwen3vl4b_full_with_dagger.jsonl`
- 学习率从 `2e-5` 降到：
  - `1e-5`
- `save_steps` 从 `200` 改成：
  - `1000`

## 训练状态

- 训练启动时间：
  - `2026-05-19 09:16`
- 已确认落盘：
  - `result/qwen3vl4b_r2r_rxr_dagger_zero2/checkpoint-1000`
  - `result/qwen3vl4b_r2r_rxr_dagger_zero2/checkpoint-2000`
  - `result/qwen3vl4b_r2r_rxr_dagger_zero2/checkpoint-3000`
  - `result/qwen3vl4b_r2r_rxr_dagger_zero2/checkpoint-4000`
  - `result/qwen3vl4b_r2r_rxr_dagger_zero2/checkpoint-5000`
- `checkpoint-1000` 状态：
  - `global_step=1000`
  - `max_steps=27065`
  - `epoch=0.03695`
- 说明：
  - 当前只跑了约 `3.7%`
  - 仍属于非常早期 checkpoint
  - 后续在 `checkpoint-5000` 落盘后，已人工停止训练，不再继续

## 早期评测备注

- 已用 `checkpoint-1000` 起过一次 `vLLM + Habitat worker` 的 `val_unseen`
- 结果目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_dagger_ckpt1000_r2r_valunseen_vllm_gpu0_baseline_8w`
- 注意：
  - 中途看到的 `37/110` 一类数字不是最终 `1839` 条汇总
  - 该 checkpoint 过早，不能据此直接判断 DAgger 是否有效

## 当前判断

- 这轮 DAgger 的主要变量更像是：
  - 数据分布变化
  - DAgger 样本占比过高
  - checkpoint 评测过早
- 目前没有证据表明：
  - 是因为学习率过大导致性能下降


### `val_unseen_DAgger`

#### checkpoint-1000

```bash
Success rate: 705/1839 (0.383)
Oracle success rate: 1217/1839 (0.662)
SPL: 553.145/1839 (0.301)
Distance to goal: 7.649
Path length: 26.573
```

- 现象：
  - `OSR` 高于 base，但 `SR / SPL` 明显下降
  - `Path length` 明显偏长，表现出“到过目标附近，但最终停点变差”的倾向

#### checkpoint-5000（中途停止）

- 结果目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_dagger_ckpt5000_r2r_valunseen_vllm_gpu0_baseline_8w`
- 评测说明：
  - 评测未跑完整个 `1839` 条 `val_unseen`
  - 在观测到趋势继续变差后，手动停止评测

```bash
Success rate: 42/164 (0.256)
Oracle success rate: 107/164 (0.652)
SPL: 21.003/164 (0.128)
Distance to goal: 9.267
Path length: 32.548
```

- 现象：
  - 相比 `checkpoint-1000`，`SR / SPL` 继续下降
  - `OSR` 仍维持偏高，但 `Path length` 进一步拉长
  - 初步说明问题更像是 DAgger 数据分布导致的“过冲 / stop 变差”，而不是单纯训练步数不够

## 当前结论

- 这轮 DAgger 没有看到正向收益
- `checkpoint-1000` 与 `checkpoint-5000` 都表现出：
  - `OSR` 偏高
  - `SR / SPL` 偏低
  - `Path length` 明显偏长
- 训练与 `checkpoint-5000` 评测已人工停止
- 下一步更可能需要：
  - 重新构造 DAgger 数据
  - 降低 rescue / 长轨迹样本占比
  - 强化接近目标后的 stop 监督
