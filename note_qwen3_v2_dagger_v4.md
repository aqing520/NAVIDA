# Qwen3-VL-4B DAgger V4

## 保留下来的数据

### 保留方式

- 基本单元：`episode-level` 的整条 mixed-policy trajectory
- 收集方式：
  - `model` 主跑
  - 出错后由 `expert rescue`
  - 最后按整条 episode 决定是否保留
- 保留规则：
  - `rescued and relative_pl > 0.93`
  - `clean and relative_pl > 0.85`
- 后处理：
  - 对 merged 结果再执行一次同样阈值的 post-filter
  - 删除不满足条件的样本及对应图像目录

### R2R 最终保留数据

- merged 目录：
  - `data/daggerv4/r2r_merged`
- 原始总 episode：
  - `10819`
- 最终 kept：
  - `4035`
- keep_rate：
  - `0.3729`
- kept 分布：
  - `long_rescue = 2874`
  - `short_rescue = 1049`
  - `clean = 112`
- kept 质量：
  - `avg_actions = 59.50`
  - `avg_path_length = 8.8887`
  - `avg_pl = 0.9797`
  - `avg_distance_to_goal = 0.1637`
  - `avg_rescue_events = 2.0731`

### RxR 最终保留数据

- merged 目录：
  - `data/daggerv4/rxr_merged`
- 原始总 episode：
  - `6634`
- 最终 kept：
  - `1112`
- keep_rate：
  - `0.1676`
- kept 分布：
  - `long_rescue = 647`
  - `short_rescue = 337`
  - `clean = 128`
- kept 质量：
  - `avg_actions = 69.13`
  - `avg_path_length = 9.6804`
  - `avg_pl = 0.9772`
  - `avg_distance_to_goal = 0.1796`
  - `avg_rescue_events = 2.6286`

## 训练数据输出

- `dagger-only`：
  - `data/dagger_train_rxr_r2r_v4.jsonl`
  - `293102` 条
- `base + daggerv4`：
  - `data/train_r2r_rxr_qwen3vl4b_full_with_daggerv4.jsonl`
  - `1109999` 条
  - 其中：
    - `base = 816897`
    - `daggerv4 = 293102`

## 检查点评测记录

### 基线

- 模型：
  - `models/qwen3vl4b_ckpt12765`
- 结果目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_ckpt12765_r2r_valunseen_vllm_gpu4_baseline_16w`
- 最终结果：
  - `SR = 0.5557`
  - `OSR = 0.6063`
  - `SPL = 0.5083`
  - `DTG = 5.3920`
  - `PL = 10.4757`

### checkpoint-1000

- checkpoint：
  - `result/qwen3vl4b_r2r_rxr_daggerv4_zero2/checkpoint-1000`
- 模型目录：
  - `models/qwen3vl4b_daggerv4_ckpt1000`
- 评测目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt1000_r2r_valunseen_direct_gpu67_2w_baseline`
- 最终结果：
  - `SR = 0.5525`
  - `OSR = 0.6308`
  - `SPL = 0.4972`
  - `DTG = 5.4081`
  - `PL = 11.2954`

### checkpoint-7000

- checkpoint：
  - `result/qwen3vl4b_r2r_rxr_daggerv4_zero2/checkpoint-7000`
- 模型目录：
  - `models/qwen3vl4b_daggerv4_ckpt7000`
- 评测目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt7000_r2r_valunseen_direct_gpu6_2w_baseline`
- 最终结果：
  - `SR = 0.5623`
  - `OSR = 0.6433`
  - `SPL = 0.5028`
  - `DTG = 4.7436`
  - `PL = 10.6084`

### vLLM 同 ID 对齐记录（中期）

说明：
- 下面是 `baseline / 1000 / 7000 / 11000 / 12000` 五边共同 `episode_id` 的对齐结果
- 当前公共交集：`1581` 条
- `7000 / 11000 / 12000` 当时都还不是完整 `1839` 条，所以这里只作为中期记录

- `baseline`
  - `SR = 0.5572`
  - `OSR = 0.6078`
  - `SPL = 0.5107`
  - `DTG = 5.3747`
  - `PL = 10.4465`
- `ckpt1000`
  - `SR = 0.5389`
  - `OSR = 0.6110`
  - `SPL = 0.4908`
  - `DTG = 5.6729`
  - `PL = 11.1216`
- `ckpt7000`
  - `SR = 0.5661`
  - `OSR = 0.6331`
  - `SPL = 0.5103`
  - `DTG = 4.8418`
  - `PL = 10.4247`
- `ckpt11000`
  - `SR = 0.5724`
  - `OSR = 0.6439`
  - `SPL = 0.5215`
  - `DTG = 4.7838`
  - `PL = 10.3529`
- `ckpt12000`
  - `SR = 0.5591`
  - `OSR = 0.6173`
  - `SPL = 0.5107`
  - `DTG = 4.9230`
  - `PL = 9.7412`
