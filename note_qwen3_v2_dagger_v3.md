# Qwen3-VL-4B DAgger V3

## 目标

- 停掉 `daggerv2`
- 保留“从偏离状态恢复”的监督目标
- 修正 `daggerv2` 的两个主要问题：
  - `correction` 只截 expert rescue span，本身太短
  - `correction` 里混入不少 terminal `stop`

## V3 思路

- `alt_path`
  - 暂时保留原逻辑
  - 不是当前主要矛盾
- `correction`
  - 仍从 `action_source` 中的连续 `expert` span 提取
  - 但不再只保留 rescue span 本身
  - 改为：
    - 从 rescue 起点开始
    - 向后接一小段 suffix action
    - 默认把 segment primitive action 长度补到 `5`
  - 同时默认去掉 correction 末尾的 terminal `stop`

## 代码改动

- `src/data/dagger_prepare_episode.py`
  - 默认输出目录改为 `prepared_daggerv3`
  - 新增 `--correction-target-primitive-len`
    - 默认 `5`
  - 新增 `--keep-correction-terminal-stop`
    - 默认不保留 correction terminal `stop`
  - `correction` 提取逻辑改为：
    - `expert span + short suffix`
    - 并裁掉末尾 `stop`
- `scripts/train_qwen3_r2r_rxr_daggerv3.sh`
  - 新增独立训练入口
  - 默认 mixed train file：
    - `data/train_r2r_rxr_qwen3vl4b_full_with_daggerv3.jsonl`
  - 默认输出目录：
    - `result/qwen3vl4b_r2r_rxr_daggerv3_zero2`

## 当前状态

- `daggerv2` 训练已停止
- `daggerv3` 代码已落地
- 已重新生成：
  - `data/dagger/r2r_dagger_v1/prepared_daggerv3`
  - `data/dagger/rxr_dagger_v1/prepared_daggerv3`

## 当前提取结果

### R2R

- `num_alt_path_rows = 3`
- `num_correction_spans = 36149`
- `num_empty_after_trim_rows = 4203`
  - 主要是 `len=1` 的 pure-stop correction，被直接删掉
- `num_trimmed_terminal_stop_rows = 817`
- `num_suffix_extended_rows = 31197`
- `num_correction_rows = 31946`
- `num_full_rows = 79142`

### RxR

- `num_alt_path_rows = 0`
- `num_correction_spans = 35377`
- `num_empty_after_trim_rows = 2737`
  - 主要是 `len=1` 的 pure-stop correction，被直接删掉
- `num_trimmed_terminal_stop_rows = 677`
- `num_suffix_extended_rows = 31986`
- `num_correction_rows = 32640`
- `num_full_rows = 163074`

## 和 V2 的关键差异

- `sub_dataset` 的 correction 平均长度已经从 `3.6~3.8` 提到接近 `5`
  - `R2R avg_len = 4.95`
  - `RxR avg_len = 4.96`
- `train_vln / train_idm` 的平均 primitive action 长度也回到了接近 base HPAC 的水平
  - `R2R VLN avg_primitive = 3.99`
  - `R2R IDM avg_primitive = 4.00`
  - `RxR VLN avg_primitive = 3.97`
  - `RxR IDM avg_primitive = 3.96`
- pure-stop 训练样本基本被清掉
  - `R2R VLN` 只剩 `1` 条 pure-stop
  - 其余集合为 `0`
- correction 末尾 `stop` 基本不再保留
  - `R2R sub_dataset` 仅剩 `3` 条 end-stop
  - `RxR sub_dataset` 为 `0`

## Mixed 训练集

- base：
  - `data/train_r2r_rxr_qwen3vl4b_full.jsonl`
  - `816,897` 条
- daggerv3：
  - `R2R prepared_daggerv3/train_full.jsonl = 79,142`
  - `RxR prepared_daggerv3/train_full.jsonl = 163,074`
- mixed：
  - `data/train_r2r_rxr_qwen3vl4b_full_with_daggerv3.jsonl`
  - `1,059,113` 条

## 训练启动

- 脚本：
  - `scripts/train_qwen3_r2r_rxr_daggerv3.sh`
- 输出目录：
  - `result/qwen3vl4b_r2r_rxr_daggerv3_zero2`
- GPU：
  - `1,3,4,5`
- `master_port`：
  - `25438`
- `learning_rate`：
  - `5e-6`
- tmux session：
  - `navida_daggerv3_train`

## `checkpoint-1000` 评测

### 配置

- checkpoint：
  - `result/qwen3vl4b_r2r_rxr_daggerv3_zero2/checkpoint-1000`
- vLLM：
  - `GPU 7`
- eval：
  - `GPU 2`
  - `16` workers
- 输出目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv3_ckpt1000_r2r_valunseen_vllm_gpu7_baseline_16w`

### 中途结果

- 当前观察到的中途统计：
  - `Success rate = 146 / 344 = 0.424`
  - `Oracle success rate = 168 / 344 = 0.488`
  - `SPL = 136.323 / 344 = 0.396`
  - `Distance to goal = 6.271`
  - `Path length = 9.172`

### 当前判断

- `v3` 虽然修掉了 `v2` 的 pure-stop 和过短片段问题
- 但效果依然明显差于基座
- 当前更怀疑的问题不是基座能力，也不是单纯 stop 比例
- 而是：
  - dagger 状态分布与 base expert 分布不一致
  - `v3 correction` 在向后补 suffix 时，可能混入了 `model` 动作
  - 导致新增 supervision 不是纯 expert label

### 进程状态

- 本次 `daggerv3 checkpoint-1000` 评测已手动停止
- 对应 `vLLM` 也已停止
