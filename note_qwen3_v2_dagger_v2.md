# Qwen3-VL-4B DAgger V2

## 概览

- 目标：
  - 不重新采集 DAgger 数据
  - 直接从现有 `v1` 保留数据中提取更干净的监督
  - 保持 `HPAC + IDM` 训练格式不变
- 输入来源：
  - `data/dagger/r2r_dagger_v1`
  - `data/dagger/rxr_dagger_v1`

## V2 数据定义

### 1. `alt_path`

- 含义：
  - rollout step 数与原始 expert 差距不大
  - 但动作序列不同于原始参考轨迹
  - 视为“分布内附近的有效扩充”
- 当前近似规则：
  - `num_rescue_events == 0`
  - `actions != reference_actions`
  - `abs(len(actions) - len(reference_actions)) <= max_abs_step_diff`
  - `step_diff_ratio <= max_step_diff_ratio`

### 2. `correction`

- 含义：
  - 从错误位置开始
  - 只保留 `expert rescue span`
  - 监督目标是“如何从偏离状态拉回正确分布”
- 当前提取方式：
  - 基于 `kept_annotations.json` 中的 `action_source`
  - 提取连续 `expert` span
  - 每个 span 作为一条 correction 样本

### 3. 不额外补 `stop_window`

- 当前先不单独构 stop-only 数据
- 先观察：
  - `alt_path + correction`
  - 能产出多少数据
  - 以及后续训练效果

## 代码改动

- `src/data/dagger_prepare_episode.py`
  - 默认输入目录改回 `v1` collector 输出
  - 默认输出到 `prepared_daggerv2`
  - 直接从旧 `kept_annotations.json` 提取 `alt_path` 和 `correction`
  - 保留 `train_vln.jsonl / train_idm.jsonl / train_full.jsonl`
- `scripts/train_qwen3_r2r_rxr_daggerv2.sh`
  - 保留独立训练入口
  - 默认 mixed train file 名称为 `train_r2r_rxr_qwen3vl4b_full_with_daggerv2.jsonl`

## 新的 prepare 行为

### 默认目录

- `R2R input`：
  - `data/dagger/r2r_dagger_v1`
- `RxR input`：
  - `data/dagger/rxr_dagger_v1`
- `prepared output`：
  - `<input_dir>/prepared_daggerv2`

### 新参数

- `--prepared-subdir`
  - 默认 `prepared_daggerv2`
- `--max-abs-step-diff`
  - 默认 `6`
- `--max-step-diff-ratio`
  - 默认 `0.2`
- `--min-correction-len`
  - 默认 `1`
- `--seed`
  - 默认 `41`

## 输出格式

- `prepared_daggerv2/sub_dataset.jsonl`
  - 每条样本带：
    - `segment_type`
    - `segment_id`
    - `frame_start_idx`
    - `frame_end_idx`
- `prepared_daggerv2/train_vln.jsonl`
- `prepared_daggerv2/train_idm.jsonl`
- `prepared_daggerv2/train_full.jsonl`
- `prepared_daggerv2/summary.json`

## 当前提取结果

### R2R

- 输入：
  - `data/dagger/r2r_dagger_v1`
- 输出：
  - `data/dagger/r2r_dagger_v1/prepared_daggerv2`
- 统计：
  - `num_input_annotations = 4984`
  - `num_alt_path_rows = 3`
  - `num_correction_rows = 36149`
  - `num_sub_dataset_rows = 36152`
  - `num_vln_rows = 39109`
  - `num_idm_rows = 34734`
  - `num_full_rows = 73843`

### RxR

- 输入：
  - `data/dagger/rxr_dagger_v1`
- 输出：
  - `data/dagger/rxr_dagger_v1/prepared_daggerv2`
- 统计：
  - `num_input_annotations = 3465`
  - `num_alt_path_rows = 0`
  - `num_correction_rows = 35377`
  - `num_sub_dataset_rows = 35377`
  - `num_vln_rows = 115485`
  - `num_idm_rows = 35618`
  - `num_full_rows = 151103`

### 当前观察

- 现有 `v1` 数据里，`alt_path` 几乎没有
- `daggerv2` 目前本质上主要抽到了：
  - `correction` supervision
- 说明 `v1 kept data` 的主体不是“近似 expert 但路径不同”的 clean alternative path
  - 而是大量 rescue/correction 事件

## Mixed 训练集

- base：
  - `data/train_r2r_rxr_qwen3vl4b_full.jsonl`
  - `816,897` 条
- daggerv2：
  - `R2R prepared_daggerv2/train_full.jsonl = 73,843`
  - `RxR prepared_daggerv2/train_full.jsonl = 151,103`
- mixed：
  - `data/train_r2r_rxr_qwen3vl4b_full_with_daggerv2.jsonl`
  - `1,041,843` 条

## 训练启动

- 脚本：
  - `scripts/train_qwen3_r2r_rxr_daggerv2.sh`
- 输出目录：
  - `result/qwen3vl4b_r2r_rxr_daggerv2_zero2`
- GPU：
  - `1,3,4,5`
- `master_port`：
  - `25437`
- tmux session：
  - `navida_daggerv2_train`

## 当前设计意图

- 不再把整条 rescued episode 当训练主体
- 把 DAgger 监督聚焦到两类信息：
  - 接近 expert 分布但路径不同的可用扩充
  - 真实发生偏离后的 expert 修正片段
- 继续复用现有 `HPAC + IDM` 训练链路，不改 trainer

## `checkpoint-5000` 评测

### 评测配置

- checkpoint：
  - `result/qwen3vl4b_r2r_rxr_daggerv2_zero2/checkpoint-5000`
- vLLM：
  - `GPU 7`
  - port `8201`
- eval：
  - `GPU 2`
  - `16` workers
- 输出目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv2_ckpt5000_r2r_valunseen_vllm_gpu7_baseline_16w`

### 最终结果

- `Success rate = 736 / 1839 = 0.400`
- `Oracle success rate = 838 / 1839 = 0.456`
- `SPL = 662.228 / 1839 = 0.360`
- `Distance to goal = 6.975`
- `Path length = 9.403`
- `ndtw = 0.000`

### 结果解读

- 这次不是 `v1` 那种“`OSR` 高、`SR` 低、路径过长、停不住”
- `OSR - SR = 102 / 1839 = 0.055`
  - `oracle_only` 占比不高
  - 主要问题不是“经过目标但没停好”
- 主要失败类型是 `hard_fail`
  - `1001 / 1839`
  - 既没有 `success`，也没有 `oracle_success`
- `hard_fail` 的平均 `path length` 只有 `8.885`
  - 低于成功样本的 `9.597`
  - 说明不是走得过长，而是覆盖不足、偏保守
- `hard_fail` 平均起点 `DTG = 9.297`
- `hard_fail` 平均终点 `DTG = 11.392`
  - 终点反而比起点更远
  - 说明不少 episode 根本没推进到目标附近
- `443` 条 `hard_fail` 样本中：
  - 模型输出过 `stop`
  - 但最终 `DTG > 10`
  - 更像“过早结束 / 错误位置 stop”，不是“停不住”

### 当前结论

- `daggerv2` 基本压住了 `v1` 那种“长轨迹 rescue/stop 失真”问题
- 但当前新增数据几乎全是 `correction`
  - `alt_path` 几乎没有
- 结果是模型更会做局部修正
  - 但全局路径覆盖能力没有被补强
- 对应现象就是：
  - `OSR` 和 `SR` 一起下降
  - 主因是“没走到目标附近”，不是“到了附近停不住”

### 进程状态

- 本次 `checkpoint-5000` 评测已结束
- `GPU 2` worker 已退出
- `GPU 7` 上的 `vLLM` 已停止
