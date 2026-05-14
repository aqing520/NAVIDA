# Qwen 3 vl 4B  V1

## 第一版训练配置与失败原因

### 第一版使用的训练配置

- 模型：`models/Qwen3-VL-4B-Instruct`
- 数据：
  - `R2R + StreamVLN 处理后的 RxR`
  - 训练文件：`data/train_r2r_rxr_qwen3vl4b_full.jsonl`
- 启动脚本：
  - `scripts/train_qwen3_r2r_rxr.sh`
  - 旧版是 `DDP`，不是 `DeepSpeed`
- 主要超参：
  - `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`
  - `nproc_per_node=6`
  - `num_train_epochs=1`
  - `bf16`
  - `torch_dtype=bfloat16`
  - `attn_implementation=sdpa`
  - `lr_scheduler_type=cosine`
  - `warmup_ratio=0.01`
  - `gradient_checkpointing=True`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=12`
  - `dataloader_num_workers=0`
  - `dataloader_pin_memory=False`
  - `learning_rate=5e-6`
  - `logging_steps=10`
  - `save_steps=200`
  - `save_total_limit=5`

### 第一版存在的问题

- 这套配置和后来跑通的 `Qwen2.5-VL-3B` 方案差异较大：
  - 用的是 `DDP`
  - 注意力实现是 `sdpa`
  - 学习率只有 `5e-6`
- 训练虽然能跑完一版 `R2R + RxR`，但整体性能偏低，不能说明 `Qwen3-VL-4B` 本身一定弱，更可能是训练配置没有把模型充分训起来。

### 失败原因记录

- 在后续 `R2R + RxR + ScaleVLN + EnvDrop` 这条线上，旧版 `Qwen3-VL-4B` 训练中途停下后，多次续训失败。
- 最后稳定可恢复的 checkpoint 是 `checkpoint-16800`。
- 从这个 checkpoint 恢复后，会在 `16801` 附近反复报 `CUDA OOM`。
- 报错位置不是前向，而是 `optimizer.step()`，更具体地说是 `AdamW` 的 multi-tensor update 阶段。
- 说明根因不是数据路线错误，而是：
  - `DDP` 配置下显存余量太小
  - 续训恢复时 optimizer state / 显存峰值把 48G 卡顶满
  - 因此这套旧配置不适合继续作为 Qwen3 主方案
- 后续改进方向就是现在这版：
  - 切到 `DeepSpeed zero2`
  - 改用 `flash_attention_2`
  - 学习率提升到 `2e-5`
  - 保持同一套训练数据，再重新验证 Qwen3 的真实上限

## R2R RXR数据训练

### qwen3vl-4b rxr-r2r训练测试  val_seen

#### ckpt200
``` bash
Success rate: 15/163 (0.092)
Oracle success rate: 27/163 (0.166)
SPL: 12.262/163 (0.075)
Distance to goal: 8.947
Path length: 7.669
```

#### ckpt400
``` bash
Success rate: 21/106 (0.198)
Oracle success rate: 26/106 (0.245)
SPL: 19.552/106 (0.184)
Distance to goal: 8.848
Path length: 8.316
```

#### ckpt600
``` bash
Success rate: 30/169 (0.178)
Oracle success rate: 45/169 (0.266)
SPL: 27.012/169 (0.160)
Distance to goal: 8.377
Path length: 8.159
```


#### ckpt1000
``` bash
Success rate: 169/778 (0.217)
Oracle success rate: 252/778 (0.324)
SPL: 151.508/778 (0.195)
Distance to goal: 8.299
Path length: 9.613
```


#### ckpt2000
``` bash
Success rate: 188/778 (0.242)
Oracle success rate: 225/778 (0.289)
SPL: 177.017/778 (0.228)
Distance to goal: 7.910
Path length: 7.573
```

#### ckpt5200
``` bash
Success rate: 233/778 (0.299)
Oracle success rate: 289/778 (0.371)
SPL: 220.358/778 (0.283)
Distance to goal: 7.529
Path length: 8.474
```

#### ckpt7200
``` bash
Success rate: 245/778 (0.315)
Oracle success rate: 286/778 (0.368)
SPL: 232.049/778 (0.298)
Distance to goal: 7.405
Path length: 8.233
```

#### ckpt8200
``` bash
Success rate: 243/778 (0.312)
Oracle success rate: 285/778 (0.366)
SPL: 227.967/778 (0.293)
Distance to goal: 7.146
Path length: 8.198
```

#### ckpt11346
``` bash
Success rate: 240/778 (0.308)
Oracle success rate: 294/778 (0.378)
SPL: 228.662/778 (0.294)
Distance to goal: 7.373
Path length: 8.439
```


#### ckpt11346  val_unseen
``` bash
Success rate: 558/1839 (0.303)
Oracle success rate: 654/1839 (0.356)
SPL: 518.712/1839 (0.282)
Distance to goal: 7.991
Path length: 8.802
```

## R2R RXR ScaleVLN EnvDrop数据集训练

#### ckpt1000
``` bash
Success rate: 58/778 (0.075)
Oracle success rate: 107/778 (0.138)
SPL: 44.852/778 (0.058)
Distance to goal: 9.458
Path length: 8.479
```

#### ckpt7600
``` bash
Success rate: 164/778 (0.211)
Oracle success rate: 204/778 (0.262)
SPL: 154.234/778 (0.198)
Distance to goal: 8.616
Path length: 7.191
```

#### ckpt14400
``` bash
Success rate: 188/778 (0.242)
Oracle success rate: 228/778 (0.293)
SPL: 177.664/778 (0.228)
Distance to goal: 8.500
Path length: 7.435
```
