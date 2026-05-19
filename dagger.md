# DAgger V1

这次新增了一套 `episode-level` 的 DAgger 数据构造链路，目标是先尽量复刻 `StreamVLN` 默认采集逻辑，再把保留 episode 转成 `NAVIDA` 当前训练可直接使用的 HPAC 风格 JSONL。

## 新增代码

### `src/data/dagger_collect_episode.py`

作用：

- 在 `R2R` 或 `RxR` train split 上做 rollout
- 默认 `model` 主走
- 当出现明显错误时触发 `expert rescue`
- 按 `episode-level` 规则决定一整条轨迹是否保留
- 为保留 episode 保存图像、annotation 和统计信息

核心逻辑：

- `expert` 使用 `ShortestPathFollower`
- `expert` 目标沿 `reference_path` waypoint 前进，不直接 shortest-path 到终点
- rescue 触发条件对齐 `StreamVLN` 默认逻辑：
  - 提前 `stop` 且 `distance_to_goal >= 3.0`
  - `accumulated_error / max(1, int(ref_actions_len / (len(ref_path) - 1))) > 0.8`
  - `accumulated_error > 12`
- rescue 长度固定为 `4` 个原子动作
- keep 条件：
  - `distance_to_goal < 0.5`
  - 若发生过 rescue，则 `pl < 0.93`
  - 否则 `pl < 0.85`

输出目录默认：

- `data/dagger/r2r_dagger_v1/`
- `data/dagger/rxr_dagger_v1/`

输出文件：

- `config.json`
- `metrics_all.jsonl`
- `metrics_kept.jsonl`
- `result_summary.json`
- `kept_annotations.json`
- `images/<episode_id>/frame_*.jpg`
- `debug_videos/*.gif`

### `src/data/dagger_prepare_episode.py`

作用：

- 读取 `kept_annotations.json` 和保留帧
- 生成中间桥接文件 `prepared/sub_dataset.jsonl`
- 再严格复用当前项目 HPAC 风格，生成最终训练数据

输出文件：

- `prepared/sub_dataset.jsonl`
- `prepared/train_vln.jsonl`
- `prepared/train_idm.jsonl`
- `prepared/train_full.jsonl`
- `prepared/summary.json`

说明：

- `train_vln.jsonl` 和 `train_idm.jsonl` 的字段、文本动作风格、`conversations` 结构、`action_history` 结构都对齐当前训练集
- `RxR` 在 collector 阶段按轨迹保留一次，在 prepare 阶段对 `vln` 按 instruction list 展开
- `idm` 继续沿用当前项目逻辑，不额外按多 instruction 倍增

## 数据格式

### `kept_annotations.json`

每条保留 episode 包含：

- `episode_id`
- `trajectory_id`
- `dataset`
- `video_id`
- `image_dir`
- `instructions`
- `actions`
- `action_source`
- `num_rescue_events`
- `model_success`
- `kept_reason`

说明：

- `actions` 是最终实际执行过的原子动作序列
- `action_source` 与 `actions` 等长，值为 `model` 或 `expert`
- `instructions` 在 `RxR` 下是 list，在 `R2R` 下也统一存 list

## 使用方法

### 1. 采集 DAgger episode

当前建议使用长期保留在 `models/` 下的模型：

- `models/qwen3vl4b_ckpt12765`

`R2R` 示例：

```bash
/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python src/data/dagger_collect_episode.py \
  --dataset r2r \
  --model-path models/qwen3vl4b_ckpt12765 \
  --output-dir data/dagger/r2r_dagger_v1
```

`RxR` 示例：

```bash
/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python src/data/dagger_collect_episode.py \
  --dataset rxr \
  --model-path models/qwen3vl4b_ckpt12765 \
  --output-dir data/dagger/rxr_dagger_v1
```

如果只想先做小规模 smoke test，可以加：

```bash
--max-episodes 100
```

常用可选参数：

- `--lora-path`
- `--exp-config`
- `--max-episodes`
- `--max-debug-videos`
- `--forward-distance`
- `--turn-angle`
- `--max-action-history`
- `--resolution-ratio`
- `--prompt-style`
- `--num-generations`

### 2. 转成训练 JSONL

`R2R`：

```bash
python src/data/dagger_prepare_episode.py \
  --dataset r2r \
  --input-dir data/dagger/r2r_dagger_v1
```

`RxR`：

```bash
python src/data/dagger_prepare_episode.py \
  --dataset rxr \
  --input-dir data/dagger/rxr_dagger_v1
```

完成后可直接使用：

- `data/dagger/r2r_dagger_v1/prepared/train_full.jsonl`
- `data/dagger/rxr_dagger_v1/prepared/train_full.jsonl`

## 当前实现边界

当前是 `DAgger V1`：

- 采用 `episode-level` 保留
- 不做 `EfficientVLN` 式 `state relabel`
- 不修改现有训练器
- 只负责把 episode rollout 数据转成当前训练链路可吃的 HPAC JSONL

后续如果要做 `V2`，可以在此基础上再加：

- state-level relabel
- 更复杂的 keep/filter 策略
- 和 base train set 的拼接或混采逻辑
