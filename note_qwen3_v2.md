# Qwen3-VL-4B V2

## 第二版重训背景

- 第一版 `Qwen3-VL-4B` 使用的是旧训练方案：
  - `DDP`
  - `sdpa`
  - `learning_rate=5e-6`
- 第一版虽然能训，但性能偏低，而且后续续训时在 `optimizer.step()` 阶段反复 `OOM`，说明旧方案不稳。
- 第二版的目标是：
  - 不改数据路线
  - 直接沿用已经在 `Qwen2.5-VL-3B` 上验证有效的训练范式
  - 重新测试 `Qwen3-VL-4B` 的真实上限

## 数据配置

  | 数据集               | 轨迹数  | VLN样本数  | IDM样本数 | 总样本数 |

  | R2R                 | 10,819  | 137,048   | 133,539   | 270,587 |
  | RxR (streamvln_rxr) | 6,634   | 411,373   | 134,937   | 546,310 |

## 训练脚本

- 脚本：
  - `scripts/train_qwen3_r2r_rxr.sh`
- 环境：
  - `navida_wzy`
- 输出目录：
  - `result/qwen3vl4b_r2r_streamvln_rxr_zero2`

## 第二版训练参数

- 模型：
  - `models/Qwen3-VL-4B-Instruct`
- 并行方式：
  - `DeepSpeed ZeRO-2`
- deepspeed 配置：
  - `scripts/zero2.json`
- 使用 GPU：
  - `3,4,5,6`
- `master_port=25435`

### 主要超参数

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
- `dataloader_pin_memory=False`
- `learning_rate=2e-5`
- `logging_steps=5`
- `eval_strategy=no`
- `save_strategy=steps`
- `save_steps=200`
- `save_total_limit=5`
- `report_to=tensorboard`

## 相比第一版的关键变化

- 从 `DDP` 改成 `DeepSpeed ZeRO-2`
- 从 `sdpa` 改成 `flash_attention_2`
- 学习率从 `5e-6` 提升到 `2e-5`
- 全局 batch 保持在 `64`
- 微批量改成更适合 4B 模型的：
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=8`

## 评测口径

- 主要评测方式：
  - `vLLM + Habitat worker`
- 常用配置：
  - `prompt_style=baseline`
  - `temperature=0.2`
  - `split=val_seen` 或 `val_unseen`
- 当前这轮 `val_unseen` 主要使用：
  - `GPU 2` 启动 vLLM
  - `GPU 0` 启动 Habitat worker
  - `12 workers`

### vLLM 启动注意事项

- Qwen3 checkpoint 目录默认只有模型权重和训练状态文件，不包含完整 tokenizer / processor 文件。
- 因此在评测前，需要把以下文件补到对应 checkpoint 目录里：
  - `chat_template.json`
  - `preprocessor_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `video_preprocessor_config.json`
  - `vocab.json`
  - `merges.txt`
- 另外因为根分区空间紧张，vLLM 缓存需要重定向到仓库目录：
  - `.vllm_cache_root`
  - `.vllm_inductor_cache`
  - `.vllm_triton_cache`
  - `.vllm_tmp`

## 当前已知关键结果

### `val_seen`

#### checkpoint-1400

```bash
Success rate: 317/778 (0.407)
Oracle success rate: 399/778 (0.513)
SPL: 294.115/778 (0.378)
Distance to goal: 7.014
Path length: 10.905
ndtw: 0.000
```

#### checkpoint-6800

```bash
Success rate: 388/778 (0.499)
Oracle success rate: 448/778 (0.576)
SPL: 366.947/778 (0.472)
Distance to goal: 5.733
Path length: 9.599
ndtw: 0.000
```

#### checkpoint-12765-final

```bash
Success rate: 492/778 (0.632)
Oracle success rate: 533/778 (0.685)
SPL: 463.181/778 (0.595)
Distance to goal: 4.697
Path length: 10.410
```

### `val_unseen`

#### checkpoint-6800

```bash
Success rate: 873/1839 (0.475)
Oracle success rate: 968/1839 (0.526)
SPL: 820.221/1839 (0.446)
Distance to goal: 6.218
Path length: 9.073
ndtw: 0.000
```

#### checkpoint-7400

```bash
Success rate: 900/1839 (0.489)
Oracle success rate: 1089/1839 (0.592)
SPL: 786.772/1839 (0.428)
Distance to goal: 6.280
Path length: 12.241
ndtw: 0.000
```

#### checkpoint-8600

```bash
Success rate: 923/1839 (0.502)
Oracle success rate: 994/1839 (0.541)
SPL: 863.613/1839 (0.470)
Distance to goal: 5.858
Path length: 9.292
```



#### checkpoint-9000

```bash
Success rate: 975/1839 (0.530)
Oracle success rate: 1078/1839 (0.586)
SPL: 902.022/1839 (0.490)
Distance to goal: 5.801
Path length: 10.072
```

#### checkpoint-11400

```bash
Success rate: 1006/1839 (0.547)
Oracle success rate: 1084/1839 (0.589)
SPL: 925.913/1839 (0.503)
Distance to goal: 5.342
Path length: 9.929
```
#### checkpoint-12765-final

```bash
Success rate: 1022/1839 (0.556)
Oracle success rate: 1115/1839 (0.606)
SPL: 934.782/1839 (0.508)
Distance to goal: 5.392
Path length: 10.476
```




