## 训练配置

- 模型：`Qwen2.5-VL-3B`
- 模型路径：`models/qwen2.5vl`
- 数据：
  - `R2R + StreamVLN 处理后的 RxR`
  - 训练文件：`data/train_r2r_rxr_qwen3vl4b_full.jsonl`
- 输出目录：
  - `result/qwen25vl3b_r2r_streamvln_rxr`
- 启动脚本：
  - `scripts/train_qwen25_r2r_streamvln.sh`
- 训练方式：
  - `DeepSpeed zero2`
  - 4 卡训练
- 主要超参：
  - `CUDA_VISIBLE_DEVICES=3,4,5,6`
  - `num_train_epochs=1`
  - `bf16`
  - `torch_dtype=bfloat16`
  - `attn_implementation=flash_attention_2`
  - `lr_scheduler_type=cosine`
  - `gradient_checkpointing=True`
  - `per_device_train_batch_size=4`
  - `gradient_accumulation_steps=4`
  - `global batch = 4 * 4 * 4 = 64`
  - `dataloader_num_workers=4`
  - `dataloader_pin_memory=True`
  - `learning_rate=2e-5`
  - `logging_steps=5`
  - `eval_strategy=no`
  - `save_strategy=steps`
  - `save_steps=200`
  - `save_total_limit=5`

### val_seen



#### ckpt 1200
``` bash
Success rate: 42/477 (0.088)
Oracle success rate: 197/477 (0.413)
SPL: 23.398/477 (0.049)
Distance to goal: 11.827
Path length: 22.864
```


#### ckpt 6600
``` bash
Success rate: 314/778 (0.404)
Oracle success rate: 424/778 (0.545)
SPL: 286.767/778 (0.369)
Distance to goal: 7.047
Path length: 12.244
```


### val_unseen
#### ckpt 6600
``` bash
Success rate: 680/1839 (0.370)
Oracle success rate: 914/1839 (0.497)
SPL: 605.729/1839 (0.329)
Distance to goal: 7.616
Path length: 12.220
```


#### ckpt 8400
``` bash
Success rate: 808/1839 (0.439)
Oracle success rate: 969/1839 (0.527)
SPL: 728.802/1839 (0.396)
Distance to goal: 6.296
Path length: 11.154
```



#### ckpt 10000
``` bash
Success rate: 840/1839 (0.457)
Oracle success rate: 978/1839 (0.532)
SPL: 762.796/1839 (0.415)
Distance to goal: 6.394
Path length: 10.719
```



#### ckpt 11000
``` bash
Success rate: 857/1839 (0.466)
Oracle success rate: 979/1839 (0.532)
SPL: 786.085/1839 (0.427)
Distance to goal: 6.375
Path length: 10.386
```

#### ckpt final
``` bash
Success rate: 855/1839 (0.465)
Oracle success rate: 999/1839 (0.543)
SPL: 783.363/1839 (0.426)
Distance to goal: 6.273
Path length: 10.570
```

#### ckpt final  tmp=0.1
``` bash
Success rate: 835/1829 (0.457)
Oracle success rate: 978/1829 (0.535)
SPL: 759.337/1829 (0.415)
Distance to goal: 6.328
Path length: 10.665
```
