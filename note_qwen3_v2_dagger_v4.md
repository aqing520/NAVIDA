# Qwen3-VL-4B DAgger V4

## 目标

- 回到更接近 `StreamVLN / v1` 的 `episode-level` DAgger 路线
- 不再走 `v2/v3` 的 `correction-only` 切片方案
- 修正 `v1` 的两个关键问题：
  - `relative_pl` 保留方向写反，错误偏向长轨迹
  - `model stop` 被无条件改写成 `expert continue`

## V4 思路

- collector 继续使用：
  - `model` 主跑
  - 出错后 `expert rescue`
  - 最终按 `episode-level` 决定是否保留整条轨迹
- prepare / train 方向：
  - 仍以整条 kept mixed-policy trajectory 为基本单元
  - 不再从 `action_source` 中切 `correction span`

## 代码改动

- `src/data/dagger_collect_episode.py`
  - 删掉了这段无条件 stop 改写：
    - `if action == 0 and not force_episode_end: action = expert`
  - keep 逻辑最终修正为：
    - `rescued and relative_pl > 0.93`
    - `clean and relative_pl > 0.85`
    - 且 `rescued` 分支与 `clean` 分支显式区分，避免 `or (relative_pl > 0.85)` 覆盖 rescued 阈值
- `src/data/dagger_post_filter.py`
  - 新增：
    - `--rescued-pl-threshold`
    - `--clean-pl-threshold`
  - 可对已收集 merged 数据做原地二次过滤
  - 会同步删除被过滤掉样本对应的 `images/<episode_id>/`

## 输出目录

- R2R:
  - shard:
    - `data/daggerv4/r2r/shards/r2r_s00 ... r2r_s19`
  - merged:
    - `data/daggerv4/r2r_merged`
- RxR:
  - shard:
    - `data/daggerv4/rxr/shards/rxr_s00 ... rxr_s15`

## 收集配置

### R2R

- 使用模型：
  - `models/qwen3vl4b_ckpt12765`
- 总分片：
  - `split-num = 20`
- 设备：
  - `GPU 1/2/3/4/5`
- 并行方式：
  - 每张卡 `4` 个 collector

### RxR

- 使用模型：
  - `models/qwen3vl4b_ckpt12765`
- 子集规模：
  - `candidate_episode_count = 6634`
- 总分片：
  - `split-num = 16`
- 设备：
  - `GPU 1/2/3/4`
- 并行方式：
  - 每张卡 `4` 个 collector

## R2R 当前结果

### 完成状态

- `R2R` 已全部收集完成
- 已合并到：
  - `data/daggerv4/r2r_merged`

### merged summary（第一次 merge）

- `num_episodes = 10819`
- `num_kept = 5113`
- `keep_rate = 0.4726`
- `success_rate = 0.7998`
- `avg_distance_to_goal = 1.2316`
- `avg_pl = 0.8548`
- `avg_num_rescue_events = 4.4707`
- `avg_kept_pl = 0.9621`
- `avg_kept_distance_to_goal = 0.1623`

来源：
- `data/daggerv4/r2r_merged/result_summary.json`

### 后处理修正

- 对 `data/daggerv4/r2r_merged` 额外执行：
  - `python src/data/dagger_post_filter.py --input-dir data/daggerv4/r2r_merged --rescued-pl-threshold 0.93 --clean-pl-threshold 0.85`
- 作用：
  - 删掉由于 keep 逻辑 `or` 覆盖而错误保留的 rescued 样本
  - 同步删除对应图像目录

后处理结果：
- `original_kept = 5113`
- `filtered_kept = 4035`
- `removed_kept = 1078`
- `removed_image_dirs = 1078`

### kept 轨迹分布（修正后最终结果）

- `kept = 4035`
- `rescue_tier`:
  - `long_rescue = 2874`
  - `short_rescue = 1049`
  - `clean = 112`
- `model_success`:
  - `False = 3923`
  - `True = 112`

### kept 轨迹长度与质量（修正后最终结果）

- `avg_actions = 59.50`
- `median num_steps = 56`
- `avg num_steps = 59.50`
- `p90 num_steps = 85`
- `p95 num_steps = 96`
- `p99 num_steps = 123`
- `max num_steps = 497`
- `avg path_length = 8.8887`
- `median path_length = 8.2856`
- `avg pl = 0.9797`
- `median pl = 0.9857`
- `avg distance_to_goal = 0.1637`
- `avg rescue_events = 2.0731`

## 当前判断

- `v4` 的核心修复已经生效：
  - 保留下来的样本不再偏向长轨迹
  - `kept` 集合的 `pl` 明显更健康
- 当前 `R2R merged` 的主要特征是：
  - 数据质量比旧 `v1` 明显更正常
  - 但仍然是 `rescued-heavy`
  - `long_rescue` 占比很高
  - 仍然存在 `step` 长尾，但比第一次 merge 已明显收紧
- 训练前更值得关注的点：
  - 是否要控制 `long_rescue` 占比
  - 是否要单独保留一部分 `clean` / `short_rescue` 配比更高的版本

## RxR 最终结果

### 完成状态

- `RxR` 已全部收集完成
- 已合并到：
  - `data/daggerv4/rxr_merged`
- 对 `data/daggerv4/rxr_merged` 额外执行：
  - `python src/data/dagger_post_filter.py --input-dir data/daggerv4/rxr_merged --rescued-pl-threshold 0.93 --clean-pl-threshold 0.85`

后处理结果：
- `original_kept = 1571`
- `filtered_kept = 1112`
- `removed_kept = 459`
- `removed_image_dirs = 459`

### merged summary（修正后最终结果）

- `num_episodes = 6634`
- `num_kept = 1112`
- `keep_rate = 0.1676`
- `success_rate = 0.5965`
- `avg_num_rescue_events = 7.4174`
- `avg_kept_pl = 0.9772`
- `avg_kept_distance_to_goal = 0.1796`

### kept 轨迹分布

- `rescue_tier`:
  - `long_rescue = 647`
  - `short_rescue = 337`
  - `clean = 128`
- `model_success`:
  - `False = 984`
  - `True = 128`

### kept 轨迹长度与质量

- `avg_actions = 69.13`
- `median num_steps = 56`
- `p90 num_steps = 130`
- `p95 num_steps = 154`
- `max num_steps = 466`
- `avg path_length = 9.6804`
- `avg pl = 0.9772`
- `avg distance_to_goal = 0.1796`
- `avg rescue_events = 2.6286`

## Base 对比

### R2R

- `base`:
  - `10819` 条
  - `avg_actions = 58.35`
  - `p50 = 56`
  - `p90 = 83`
  - `p95 = 93`
- `daggerv4 kept`:
  - `4035` 条
  - `avg_actions = 59.50`
  - `p50 = 56`
  - `p90 = 85`
  - `p95 = 96`

判断：
- `R2R daggerv4` 的长度分布和 `base` 已经非常接近
- 说明这版 `R2R` 的 kept mixed-policy 轨迹形态是健康的

### RxR

- `base`:
  - `6634` 条
  - `avg_actions = 95.54`
  - `p50 = 87`
  - `p90 = 164`
  - `p95 = 193`
- `daggerv4 kept`:
  - `1112` 条
  - `avg_actions = 69.13`
  - `p50 = 56`
  - `p90 = 130`
  - `p95 = 154`

判断：
- `RxR daggerv4` 明显比 `base` 短
- 主要原因不是 `pl` 差，而是当前保留下来的样本是强过滤后的成功子集
- 分层看：
  - `clean avg_steps = 25.98`
  - `short_rescue avg_steps = 39.40`
  - `long_rescue avg_steps = 93.15`
- 也就是说，真正把均值拉短的主要是 `clean + short_rescue`
- `long_rescue` 的长度已经比较接近 `RxR base` 的长程分布

## HPAC / 训练数据

### episode-level prepare

- 新增：
  - `src/data/dagger_prepare_episode_v4.py`
- 思路：
  - 不再走 `v2/v3` 的 `correction-only` prepare
  - 直接对 `v4 kept` 的整条 mixed-policy trajectory 做 HPAC 切分
  - 输出 schema 与 base 训练集保持一致：
    - `system`
    - `conversations`
    - `action_history`
    - `episode_id`
    - `task type`

### prepared_daggerv4 结果

- `R2R`:
  - 目录：
    - `data/daggerv4/r2r_merged/prepared_daggerv4`
  - `num_vln_rows = 168663`
  - `num_idm_rows = 54832`
  - `num_full_rows = 223495`
- `RxR`:
  - 目录：
    - `data/daggerv4/rxr_merged/prepared_daggerv4`
  - `num_vln_rows = 52469`
  - `num_idm_rows = 17138`
  - `num_full_rows = 69607`

### 合并后的训练文件

- `dagger-only`:
  - `data/dagger_train_rxr_r2r_v4.jsonl`
  - `293102` 条
- `base + daggerv4`:
  - `data/train_r2r_rxr_qwen3vl4b_full_with_daggerv4.jsonl`
  - `1109999` 条
  - 其中：
    - `base = 816897`
    - `daggerv4 = 293102`

## 当前判断

- `R2R daggerv4` 已经具备训练价值：
  - keep 逻辑修正后，长度分布和 base 非常接近
  - `pl` 很高
  - 但仍然 `rescued-heavy`
- `RxR daggerv4` 也已经可用，但分布和 base 差异更大：
  - kept 样本量偏少
  - 平均轨迹明显更短
  - 更像一个高质量成功子集，而不是完整替代 `RxR base`
- 如果直接混训：
  - `R2R` 风险较小
  - `RxR` 更值得考虑降权或二次筛选


## 训练与评测

### 训练

- 训练脚本：
  - `scripts/train_qwen3_r2r_rxr_daggerv4.sh`
- 训练集：
  - `data/train_r2r_rxr_qwen3vl4b_full_with_daggerv4.jsonl`
- 初始化模型：
  - `models/qwen3vl4b_ckpt12765`
- 学习率：
  - `5e-6`

### checkpoint-1000 评测

- checkpoint:
  - `result/qwen3vl4b_r2r_rxr_daggerv4_zero2/checkpoint-1000`
- 模型目录：
  - `models/qwen3vl4b_daggerv4_ckpt1000`
- 评测方式：
  - 非 `vLLM`
  - 直接调用 `src/eval/eval.py`
  - `GPU 6/7`
  - `2 worker`
  - `split-num = 2`
  - `prompt-style = baseline`
- 结果目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt1000_r2r_valunseen_direct_gpu67_2w_baseline`

### checkpoint-1000 最终结果

- `Success rate = 1016 / 1839 = 0.5525`
- `Oracle success rate = 1160 / 1839 = 0.6308`
- `SPL = 0.4972`
- `Distance to goal = 5.4081`
- `Path length = 11.2954`

### 与基座 ckpt12765 对比

基线目录：
- `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_ckpt12765_r2r_valunseen_vllm_gpu4_baseline_16w`

基座最终结果：
- `Success rate = 0.5557`
- `Oracle success rate = 0.6063`
- `SPL = 0.5083`
- `Distance to goal = 5.3920`
- `Path length = 10.4757`

对比结论：
- `daggerv4 ckpt1000` 与强基座已经基本打平
- `SR` 略低于基座：
  - `0.5525 < 0.5557`
- `OSR` 明显高于基座：
  - `0.6308 > 0.6063`
- `hard fail` 更少：
  - `679 < 724`
- 但 `PL` 更长，`SPL` 仍略低：
  - `11.2954 > 10.4757`
  - `0.4972 < 0.5083`

当前判断：
- `v4` 没有重现 `v2/v3` 那种明显训坏基座的情况
- `checkpoint-1000` 这个很早的点位已经表现出：
  - 更强的覆盖能力
  - 更少的彻底失败
  - 但路径效率还没有超过基座


### checkpoint-7000 最终结果

- checkpoint:
  - `result/qwen3vl4b_r2r_rxr_daggerv4_zero2/checkpoint-7000`
- 模型目录：
  - `models/qwen3vl4b_daggerv4_ckpt7000`
- 评测方式：
  - 非 `vLLM`
  - 直接调用 `src/eval/eval.py`
  - `GPU 6`
  - `2 worker`
  - `split-num = 2`
  - `prompt-style = baseline`
- 结果目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt7000_r2r_valunseen_direct_gpu6_2w_baseline`

最终指标：
- `Success rate = 1034 / 1839 = 0.5623`
- `Oracle success rate = 1183 / 1839 = 0.6433`
- `SPL = 0.5028`
- `Distance to goal = 4.7436`
- `Path length = 10.6084`

与基座 `ckpt12765` 对比：
- 基座目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_ckpt12765_r2r_valunseen_vllm_gpu4_baseline_16w`
- 基座最终结果：
  - `Success rate = 0.5557`
  - `Oracle success rate = 0.6063`
  - `SPL = 0.5083`
  - `Distance to goal = 5.3920`
  - `Path length = 10.4757`

对比结论：
- `daggerv4 ckpt7000` 的 `SR` 已经超过基座：
  - `0.5623 > 0.5557`
- `OSR` 明显高于基座：
  - `0.6433 > 0.6063`
- `DTG` 更低：
  - `4.7436 < 5.3920`
- `hard fail` 更少：
  - `656 < 724`
- 但 `SPL` 仍略低，`PL` 略长：
  - `0.5028 < 0.5083`
  - `10.6084 > 10.4757`

当前判断：
- `daggerv4` 到 `checkpoint-7000` 为止，已经表现出稳定的正向收益
- 主要收益是：
  - 更高的 `SR`
  - 更高的 `OSR`
  - 更少的彻底失败
- 主要残留问题是：
  - `oracle-only` case 仍偏多
  - 路径效率还没有完全超过基座

## vLLM 同 ID 对齐对比（2026-05-26 中期）

### 对齐口径

- 使用目录：
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt1000_r2r_valunseen_vllm_gpu1w12`
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt7000_r2r_valunseen_vllm_gpu2w12`
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt11000_r2r_valunseen_vllm_gpu1w12`
  - `eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_daggerv4_ckpt12000_r2r_valunseen_vllm_gpu2w12`
  - 基线：`eval_log/qwen3_v2_r2r_rxr/qwen3vl4b_ckpt12765_r2r_valunseen_vllm_gpu4_baseline_16w`
- 比较方式：
  - 取 `baseline / 1000 / 7000 / 11000 / 12000` 五边共同存在的 `episode_id`
  - 当前公共交集为 `1581` 条
- 注意：
  - 这次 `7000` 仍在补测中
  - `11000 / 12000` 也还不是完整 `1839` 条
  - 所以下面是中期同 ID 结论，不是最终 full-set 结论

### 五边共同 ID 指标（1581 条）

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

### 当前同 ID 排序

- 当前五边共同 `1581` 条上：
  - `ckpt11000` 最强
  - `ckpt7000` 第二
  - `ckpt12000` 与基线接近，略强一点
  - `ckpt1000` 明显弱于基线

### 当前判断

- `ckpt1000`：
  - 只学到少量覆盖收益
  - `OSR` 略高于基线
  - 但 `SR / SPL / DTG / PL` 都明显更差
- `ckpt7000`：
  - 相比基线有明确正收益
  - `SR / OSR` 更高
  - `DTG` 更低
  - `PL` 基本持平
  - `SPL` 与基线接近
- `ckpt11000`：
  - 在当前共同 ID 子集上，比 `ckpt7000` 还更强
  - `SR / OSR / SPL` 都是当前最好
  - `DTG` 也是最低
- `ckpt12000`：
  - 相比 `11000` 已出现回落
  - 但还没有掉回基线以下

### 当前结论

- 若按当前 `1581` 条共同 ID 的中期结果看：
  - `daggerv4` 的峰值点更像在 `7000 ~ 11000` 区间
  - `11000` 当前比 `7000` 更优
  - `12000` 已开始回落
  - `1000` 仍然太早

