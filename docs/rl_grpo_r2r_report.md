# R2R GRPO RL 训练记录

本文记录当前项目中这条 VLN 强化学习实验的实现方式、奖励函数、训练规模和评估结果。对应代码主要位于 `src/rl/`，训练入口为 `src/rl/train_grpo.py`。

## 1. 实验目标

这条 RL 是在已经做过 DAgger/SFT 的 VLN 模型基础上继续训练，目标是让模型在 Habitat R2R 环境中通过在线 rollout 获得导航反馈，并用组内相对优势更新 LoRA 参数。

当前实验不是从零训练策略，而是：

1. 加载 SFT/DAgger 后的 Qwen3-VL VLN 模型。
2. 加载或创建 LoRA adapter。
3. 在 R2R train 环境中采样导航 rollout。
4. 用环境奖励计算每条 rollout 的 return。
5. 对同一 episode 的多条采样做组内归一化 advantage。
6. 用 policy-gradient 风格的 token logprob loss 更新 LoRA。
7. 加 KL 约束，限制 LoRA policy 偏离 base/reference policy。

## 2. 代码结构

核心文件：

- `src/rl/train_grpo.py`: 训练入口、DDP、LoRA 加载、GRPO 更新、checkpoint 保存。
- `src/rl/rollout.py`: Habitat 环境 rollout，生成动作，记录每一步 prompt/response/reward。
- `src/rl/rewards.py`: step reward 设计。
- `src/rl/actions.py`: 将模型输出解析为离散动作。
- `src/rl/prompts.py`: RL rollout 使用的 VLN prompt。

日志和结果路径：

- 训练日志：`result/log/rl_grpo_qwen3_lora_ddp4_kl_g8_s80_resume_c300_u2000_minsteps.rank*.jsonl`
- 训练 checkpoint：`result/rl/rl_grpo_qwen3_lora_ddp4_kl_g8_s80_resume_c300_u2000_minsteps/`
- full eval 结果：`result/eval/qwen3_daggerv4_ckpt7000_rl_lora_ddp4_kl_s80_ckpt2015_merged_val_unseen_full_nomap_t02_split8/`

## 3. 模型和训练架构

基础模型：

- Base model: `models/qwen3vl4b_daggerv4_ckpt7000`
- 训练方式: LoRA 微调
- LoRA target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

这次主要训练的是 LoRA adapter，不更新完整模型参数。训练日志中可见：

```text
trainable params: 60,301,824
all params:       4,470,845,952
trainable%:       1.3488
```

DDP 训练：

- 使用 4 worker 并行。
- 每个 rank 独立负责一部分 episode。
- 每个 episode 采样 `group_size=2` 条 rollout。
- 每组 rollout 内根据总 reward 做 advantage 归一化。
- 为避免 DDP 不同步，当前实现使用各 rank 的 `min rollout steps` 对齐 backward 数量。

训练 loss 的核心形式：

$$
\mathcal{L}
= -A \cdot \overline{\log p_\theta(y)}
+ \beta_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}}
$$

其中 KL 使用 LoRA adapter disabled 后的 base policy 作为 reference policy。对每个 response token 的 KL 近似项为：

$$
\Delta = \log p_{\mathrm{ref}}(y) - \log p_\theta(y)
$$

$$
\mathrm{KL}_{\mathrm{token}} = \exp(\Delta) - \Delta - 1
$$

## 4. Rollout 和动作空间

环境来自 Habitat R2R：

```text
config/vln_r2r_train.yaml
```

每一步流程：

1. 从 Habitat 取当前 RGB observation 和 instruction。
2. 保留历史 RGB，最多 `max_action_history=200`。
3. prompt 中最多采样 `max_history_images=8` 张历史图。
4. 模型生成动作文本。
5. 解析成离散动作。
6. 在 Habitat 环境中执行动作。
7. 根据前后 metrics 计算 reward。
8. 记录该 step 的 prompt、response、动作、reward 和 metrics。

动作参数：

```text
forward_distance: 25
turn_angle:       15
max_steps:        400  # 与 eval 的 EARLY_STOP_STEPS 对齐，表示 primitive action 上限
```

动作语义和 eval 保持一致，主要包括：

- stop
- move forward
- turn left
- turn right

当前 RL rollout 已对齐 eval 的 HPAC 执行方式：模型每次输出一个动作块，解析前两个 high-level 子动作，每个 high-level 子动作最多展开 3 个 primitive action，然后连续执行这个 primitive queue。reward 在整个 HPAC block 执行前后计算一次。

## 5. 奖励函数

奖励函数在 `src/rl/rewards.py` 中实现。当前 RL 的 reward 粒度是 HPAC block：每次模型输出一个动作块，执行其中前两个 high-level 子动作展开后的 primitive queue，然后计算一次奖励。记第 `b` 个 block 执行前后的目标距离分别为：

$$
d_b = \texttt{prev\_metrics}[\texttt{"distance\_to\_goal"}], \qquad
d_{b+1} = \texttt{next\_metrics}[\texttt{"distance\_to\_goal"}]
$$

距离变化奖励为：

$$
r_{\mathrm{dist}}(b) = \alpha \left(d_b - d_{b+1}\right)
$$

其中 `alpha = distance_delta_scale`。靠近目标时 `d_b - d_{b+1} > 0`，奖励为正；远离目标时奖励为负。

完整 block 奖励为：

$$
r_b =
\alpha \left(d_b - d_{b+1}\right)
+ \beta_{\mathrm{step}}
+ \mathbb{I}_{\mathrm{invalid}}(b)\beta_{\mathrm{invalid}}
+ \mathbb{I}_{\mathrm{collision}}(b)\beta_{\mathrm{collision}}
+ \mathbb{I}_{\mathrm{stop}}(b)r_{\mathrm{stop}}(b)
$$

其中：

$$
r_{\mathrm{stop}}(b)=
\begin{cases}
\beta_{\mathrm{success}}, & \text{if } \mathrm{success}_{b+1}>0 \text{ or } d_{b+1}\le d_{\mathrm{goal}} \\
\beta_{\mathrm{wrong\_stop}}, & \text{otherwise}
\end{cases}
$$

当前默认参数：

```text
alpha / distance_delta_scale:      1.0
beta_success / success_bonus:      5.0
beta_wrong_stop:                  -3.0
beta_invalid:                     -0.5
beta_collision:                   -0.2
beta_step / step_penalty:         -0.01
goal_distance:                     3.0
```

具体含义：

- 靠近目标会获得正奖励，远离目标会得到负奖励。
- 每个 HPAC block 有轻微 step penalty，鼓励更短路径。
- 无效动作会扣分。
- block 内任一 primitive action 发生碰撞会扣分。
- 如果模型执行 stop：
  - 成功或距离目标小于等于 `3.0m`，加 `success_bonus=5.0`。
  - 否则加 `wrong_stop_penalty=-3.0`。

episode return 是所有 block reward 的和：

$$
R_i = \sum_b r_{i,b}
$$

## 6. GRPO / 组内优势

本实现不是 PPO 那种 actor-critic 结构，没有单独训练 value model。它更接近 GRPO / group relative policy optimization：对同一个 episode 采样多条 rollout，用组内 return 做相对优势。

对每个 episode，采样 `G = group_size` 条 rollout：

$$
\{\tau_1,\tau_2,\ldots,\tau_G\}
$$

每条 rollout 的 return 为：

$$
R_i = \sum_b r_{i,b}
$$

组内均值和标准差为：

$$
\mu_R = \frac{1}{G}\sum_{i=1}^{G}R_i,
\qquad
\sigma_R = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(R_i-\mu_R)^2}
$$

优势为：

$$
A_i = \frac{R_i-\mu_R}{\sigma_R+\epsilon}
$$

如果组内 `sigma_R` 太小，则 advantage 置 0，避免同质样本产生不稳定更新。

对第 `i` 条 rollout 中每一步模型生成的 response token，计算平均 log probability：

$$
\log p_i =
\frac{1}{|\mathcal{T}_i|}
\sum_{(t,k)\in\mathcal{T}_i}
\log \pi_\theta\left(y_{i,t,k}\mid x_{i,t}, y_{i,t,<k}\right)
$$

其中：

- `t` 是 rollout step。
- `k` 是该 step response 内的 token。
- prompt 部分 token 会被 mask 掉，只对 assistant response 计算 logprob。

策略梯度项为：

$$
\mathcal{L}_{\mathrm{pg}}(i) = -A_i \log p_i
$$

当 `A_i > 0` 时，最小化 loss 会提高该 rollout response 的概率；当 `A_i < 0` 时，会降低该 rollout response 的概率。

KL 约束使用关闭 LoRA adapter 后的 base policy 作为 reference policy。代码中对每个有效 response token 计算：

$$
\Delta_{i,t,k}
= \log \pi_{\mathrm{ref}}\left(y_{i,t,k}\mid x_{i,t}, y_{i,t,<k}\right)
- \log \pi_{\theta}\left(y_{i,t,k}\mid x_{i,t}, y_{i,t,<k}\right)
$$

$$
\mathrm{KL}_{i,t,k}
= \exp\left(\Delta_{i,t,k}\right) - \Delta_{i,t,k} - 1
$$

再对 response tokens 求平均：

$$
\mathcal{L}_{\mathrm{kl}}(i)
= \frac{1}{|\mathcal{T}_i|}
\sum_{(t,k)\in\mathcal{T}_i}\mathrm{KL}_{i,t,k}
$$

最终单步训练 loss 为：

$$
\mathcal{L}_i
= -A_i \log p_i
+ \beta_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}}(i)
$$

本次训练使用：

```text
G / group_size: 2
beta_kl:        0.04
```

这意味着本实现更接近“同题多采样的相对偏好优化”：

- 好于同组平均的 rollout，其 token logprob 被提高。
- 差于同组平均的 rollout，其 token logprob 被降低。
- KL 项约束当前 LoRA policy 不要偏离 base policy 太远。

## 7. 当前 HPAC RL 训练配置

当前代码建议使用的 run：

```text
run_name: rl_grpo_qwen3_lora_ddp4_kl_g8_s80_resume_c300_u2000_minsteps
```

关键配置分组如下。

模型和 LoRA：

```text
base model:              models/qwen3vl4b_daggerv4_ckpt7000
resume lora:             checkpoint-300
final checkpoint:         checkpoint-2015
merged checkpoint:        checkpoint-2015-merged
LoRA rank:               16
LoRA alpha:              32
LoRA dropout:            0.05
LoRA target modules:     q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
trainable params:         60,301,824
trainable ratio:          1.3488%
```

数据和 rollout：

```text
dataset split:            R2R train
exp config:               config/vln_r2r_train.yaml
forward distance:         25
turn angle:               15
max primitive steps:       400
max action history:       200
HPAC high-level actions:   2 per model output
max primitive per action:  3
max history images:       8
resolution ratio:         0.5
```

采样和生成：

```text
group size:               2
temperature:              0.7
top_p:                    1.0
max new tokens:           32
repetition penalty:       1.05
```

优化参数：

```text
learning rate:            5e-6
weight decay:             0.0
gradient accumulation:    8
max grad norm:            1.0
KL beta:                  0.04
torch dtype:              bfloat16
attention:                flash_attention_2
freeze vision:            true
freeze linear attention:  true
gradient checkpointing:   false
```

并行、步数和保存：

```text
world size:               4
num epochs:               1
target max updates:       2000
actual final global_step: 2015
save steps:               100
DDP timeout:              7200 seconds
```

由于每个 episode 的 rollout 长度不同，`global_step` 不一定刚好停在目标步数；后续续训 checkpoint 名称以实际 `global_step` 为准。

## 8. 训练覆盖量

以 R2R train 为分母：

```text
R2R train episode:     10819
R2R train trajectory:  3603
R2R train scene:       61
```

本次 RL 实际覆盖：

```text
唯一 episode:          696
实际 rollout:          1392
覆盖 trajectory:       657
覆盖 scene:            59
总环境步数:            85319
平均每条 rollout:      61.29 step
```

覆盖比例：

```text
episode 覆盖率:        6.43%
trajectory 覆盖率:     18.23%
scene 覆盖率:          96.72%
```

未覆盖到的 R2R train scene：

```text
YmJkqBEsHnH
gZ6f7yhEvPG
```

训练期间 rollout 自身的平均表现：

```text
rollout success rate:  0.6293
rollout SPL:           0.6087
```

注意：训练 rollout 指标不能直接等同于 val_unseen 泛化性能，因为它是在 train episode 上在线采样得到的。

## 9. Checkpoint

本次 run 保存了：

```text
checkpoint-1200
checkpoint-2015
checkpoint-2015-merged
```

`checkpoint-2015` 是 LoRA adapter。为了 eval，已将其合并为 full model：

```text
result/rl/rl_grpo_qwen3_lora_ddp4_kl_g8_s80_resume_c300_u2000_minsteps/checkpoint-2015-merged
```

早期只看到 `checkpoint-1200` 的原因是保存条件原来要求 `global_step % save_steps == 0`，而当前训练 step 会跳着增长，容易跳过整百点。现在代码已改为跨过保存区间即保存，后续不会再因为没精确命中整百而漏保存。

## 10. Full Val Unseen 评估结果

评估设置：

```text
split:        val_unseen full
episodes:     1839
temperature:  0.2
map saving:   false
model:        checkpoint-2015-merged
parallelism:  split8
```

`checkpoint-2015` full eval：

```text
SR:           0.5699  (1048 / 1839)
SPL:          0.5203
OracleSR:     0.6253  (1150 / 1839)
DTG:          4.7199
PathLen:      9.9631
```

已有对比：

```text
baseline daggerv4 ckpt7000 vllm:
SR 0.5650, SPL 0.5090, OracleSR 0.6297

RL checkpoint-300 full:
SR 0.5672, SPL 0.5092, OracleSR 0.6319

RL checkpoint-2015 full:
SR 0.5699, SPL 0.5203, OracleSR 0.6253
```

## 11. 当前结论

这条 RL 训练没有崩，full val_unseen 上相对 baseline 有小幅提升：

- SR 从约 `0.5650` 提到 `0.5699`。
- SPL 从约 `0.5090` 提到 `0.5203`。
- 成功数大约多 9 条左右，相比 ckpt300 多 5 条左右。
- OracleSR 下降，说明可到达潜力没有扩大，更多体现为路径更短或 stop/动作分布更干脆。

整体判断：

1. 当前 RL 方向是有效但提升幅度有限。
2. 场景覆盖已经很高，真正不足的是 train trajectory/episode 覆盖。
3. 继续训练应优先保证更多 trajectory 覆盖，而不是单纯追求更多 step。
4. 如果要系统扩大训练，建议做 trajectory-aware sampling，确保 3603 条 R2R train trajectory 至少覆盖一遍。

## 12. 后续建议

下一轮建议：

```text
目标: 覆盖全部 3603 条 R2R train trajectory 至少一次
采样: 每条 trajectory group_size=2 或 4
约束: 保留 KL beta，避免 policy 漂移
评估: 固定 temperature=0.2，full val_unseen
保存: 按跨过 save_steps 区间保存 checkpoint
```

这样比单纯继续随机跑更多 update 更可控，也更容易判断 RL 是否真的带来泛化提升。
