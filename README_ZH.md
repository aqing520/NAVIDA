# NaVIDA 项目详细文档

> **NaVIDA**: Vision-Language Navigation with Inverse Dynamics Augmentation
> 
> 一个基于逆动力学监督的轻量级视觉-语言导航框架，使用 Qwen2.5-VL-3B 作为骨干模型。

---

## 📖 项目概述

NaVIDA 是一个端到端的视觉-语言导航（VLN）系统。它将导航任务转化为**视觉-语言大模型的对话任务**：输入历史观测图像序列 + 当前图像 + 自然语言指令，模型直接输出导航动作（如"forward 50 cm, turn left 15 degree"）。

### 核心创新点

1. **逆动力学监督（IDM）**：让模型学习"从当前视图到目标视图应该执行什么动作"
2. **分层动作分块（Action Chunking）**：一次预测多个连续动作，减少模型调用次数
3. **轻量级设计**：仅 3B 参数，在 R2R/RxR 基准上超越 7B-8B 模型

---

## 🗂️ 目录结构总览

```
NAVIDA/
├── README.md                          # 原始英文 README
├── README_DETAILED.md                 # 本文件（详细中文文档）
│
├── asset/                             # 文档图片资源
│   ├── architecture.png               # 模型架构图
│   └── data.png                       # 数据处理流程图
│
├── config/                            # Habitat 模拟器配置文件（YAML）
│   ├── vln_r2r.yaml                   # R2R 验证集配置
│   ├── vln_r2r_train.yaml             # R2R 训练集配置
│   ├── vln_rxr.yaml                   # RxR 验证集配置
│   ├── vln_rxr_train.yaml             # RxR 训练集配置
│   ├── vln_envdrop.yaml               # EnvDrop 数据集配置
│   └── vln_scalevln.yaml              # ScaleVLN 数据集配置
│
├── habitat_extensions/                # Habitat 框架扩展
│   ├── task.py                        # 自定义 VLN 任务与数据集定义
│   └── measures.py                    # 自定义评估指标（nDTW, OracleSuccess 等）
│
├── scripts/                           # Shell 脚本（一键执行）
│   ├── preprocess.sh                  # 数据预处理脚本
│   ├── extract_frame.sh               # RGB 帧提取脚本
│   ├── prepare_training_data.sh       # 训练数据构造脚本
│   ├── train.sh                       # 模型训练脚本
│   ├── eval.sh                        # 评估脚本（Transformers 版）
│   ├── eval_vllm.sh                   # 评估脚本（vLLM 加速版）
│   ├── start_vllm_server.sh           # 启动 vLLM 推理服务
│   ├── analyze_results.sh             # 结果汇总分析脚本
│   ├── kill_eval.sh                   # 终止评估进程脚本
│   ├── zero1.json                     # DeepSpeed ZeRO-1 配置
│   ├── zero2.json                     # DeepSpeed ZeRO-2 配置（默认）
│   ├── zero3.json                     # DeepSpeed ZeRO-3 配置
│   └── zero3_offload.json             # DeepSpeed ZeRO-3 + CPU Offload 配置
│
└── src/                               # 核心源代码
    ├── data/                          # 数据处理模块
    │   ├── preprocess.py              # 原始数据格式统一
    │   ├── extract_frame.py           # Habitat 模拟器渲染 RGB 帧
    │   └── prepare_training_data.py   # 训练样本构造（动作分块 + IDM）
    ├── train/
    │   └── train.py                   # 模型训练（DeepSpeed + LoRA）
    └── eval/
        ├── eval.py                    # 评估主程序（加载本地模型推理）
        ├── eval_vllm.py               # 评估主程序（调用 vLLM 服务推理）
        └── analyze_results.py         # 评估结果统计分析
```

---

## 🔧 配置文件详解 (`config/`)

### 文件列表

| 文件 | 用途 | 关键参数 |
|------|------|----------|
| `vln_r2r.yaml` | R2R 验证集评估配置 | `split: val_unseen`, `success_distance: 3.0` |
| `vln_r2r_train.yaml` | R2R 训练集帧提取配置 | `split: train`, `max_episode_steps: 500` |
| `vln_rxr.yaml` | RxR 验证集评估配置 | 含 nDTW 指标，多语言指令过滤 |
| `vln_rxr_train.yaml` | RxR 训练集帧提取配置 | 同 R2R，数据集路径不同 |
| `vln_envdrop.yaml` | EnvDrop 评估配置 | 用于域随机化测试 |
| `vln_scalevln.yaml` | ScaleVLN 评估配置 | HM3D 场景，长距离导航 |

### 配置示例（vln_r2r.yaml）

```yaml
habitat:
  environment:
    max_episode_steps: 500          # 每个 episode 最多 500 步
  simulator:
    agents:
      main_agent:
        sim_sensors:
          rgb_sensor:
            width: 640              # RGB 相机分辨率
            height: 480
            hfov: 90                # 水平视场角
    forward_step_size: 0.25         # 每步前进 0.25 米（25cm）
    turn_angle: 15                  # 每次转向 15 度
  task:
    measurements:
      success:
        type: Success
        success_distance: 3.0       # 距离目标 3 米内算成功
```

---

## 🐚 脚本详解 (`scripts/`)

### 1. `preprocess.sh` — 数据预处理

**功能**：将原始数据集的 `.json.gz` 格式转换为统一的 `.jsonl` 格式。

**调用命令**：
```bash
./scripts/preprocess.sh
```

**内部执行**：
```bash
python3 src/data/preprocess.py \
    --dataset_name r2r streamvln_rxr envdrop scalevln
```

**处理内容**：
- 解压 `.json.gz` 文件
- 提取 `episode_id`, `instruction`, `actions` 等关键字段
- 输出到 `data/sub_dataset/{dataset}.jsonl`
- RxR 数据集只保留英语指令（`en-US`, `en-IN`）

**输入输出**：
```
输入: data/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
输出: data/sub_dataset/r2r.jsonl
```

---

### 2. `extract_frame.sh` — RGB 帧提取

**功能**：在 Habitat 模拟器中重放专家轨迹，渲染并保存每一帧 RGB 图像。

**调用命令**：
```bash
./scripts/extract_frame.sh
```

**内部执行**：
```bash
python3 src/data/extract_frame.py \
    --dataset_name r2r rxr envdrop scalevln \
    --num_thread 16
```

**处理内容**：
- 启动 Habitat 环境，加载 3D 场景
- 按专家动作序列逐步执行
- 每步保存当前观测的 RGB 帧（resize 到 320×240）
- **16 进程并行**加速

**输入输出**：
```
输入: data/sub_dataset/r2r.jsonl + MP3D 场景文件
输出: data/images/r2r/{episode_id}/frame_0.jpg, frame_1.jpg, ...
```

---

### 3. `prepare_training_data.sh` — 训练数据构造

**功能**：将原始动作序列转换为适合大模型训练的对话格式，并实现**动作分块**和**IDM 任务**构造。

**调用命令**：
```bash
./scripts/prepare_training_data.sh
```

**内部执行**：
```bash
python3 src/data/prepare_training_data.py \
    --dataset_name r2r streamvln_rxr envdrop scalevln \
    --task_type vln idm \
    --output_path data/navida_train_data
```

**核心逻辑**：
- **VLN 任务**：`历史图像序列 + 当前图像 + 指令 → 下一步动作块`
- **IDM 任务**：`当前视图 + 目标视图 → 两帧之间的动作`
- **动作分块**：连续相同动作以 70% 概率合并（如 `forward 25 + forward 25 → forward 50`）
- 每个 chunk 最多包含 **3 个动作**

**输入输出**：
```
输入: data/sub_dataset/*.jsonl + data/images/*
输出: data/navida_train_data.jsonl
```

---

### 4. `train.sh` — 模型训练

**功能**：使用 DeepSpeed 分布式训练框架微调 Qwen2.5-VL-3B-Instruct。

**调用命令**：
```bash
./scripts/train.sh
```

**内部执行**：
```bash
deepspeed --master_port 25420 src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name data/only_r2r_idm_vln_mix_data.jsonl \
    --model_name_or_path .cache/Qwen2.5-VL-3B-Instruct \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2.0e-5 \
    --output_dir result/navida_qwen2_5_vl_3b_train_on_r2r_mix_data_wo_dagger
```

**关键参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| `--deepspeed` | `zero2.json` | ZeRO-2 分布式优化 |
| `--bf16` | - | BF16 混合精度训练 |
| `--attn_implementation` | `flash_attention_2` | FlashAttention 加速 |
| `--per_device_train_batch_size` | 4 | 每卡 batch size |
| `--gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--num_train_epochs` | 1 | 训练轮数 |

**训练特性**：
- 冻结视觉编码器（`model.visual`），只训练 merger 层和语言模型
- 使用 `transformers.Trainer` 简化训练循环
- 支持断点续训（自动检测 `last_checkpoint`）

---

### 5. `eval.sh` — 评估（Transformers 版）

**功能**：加载训练好的模型（含 LoRA），在 Habitat 环境中进行导航评估。

**调用命令**：
```bash
./scripts/eval.sh
```

**内部执行**：
```bash
# 启动 8 个并行评估进程（每个 GPU 跑 2 个）
for IDX in 0..7; do
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} python src/eval/eval.py \
        --exp-config config/vln_r2r.yaml \
        --split-num 8 \
        --split-id $IDX \
        --model-path $MODEL_PATH \
        --result-path eval_log/navida_r2r
done

# 汇总结果
python src/eval/analyze_results.py --path eval_log/navida_r2r
```

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--split-num` | 8 | 将数据集分成 8 份并行评估 |
| `--split-id` | 0-7 | 当前进程处理第几份数据 |
| `--forward-distance` | 25 | 每步前进距离（cm） |
| `--turn-angle` | 15 | 每次转向角度（度） |
| `--resolution-ratio` | 0.5 | 图像分辨率缩放比例 |
| `--max-action-history` | 200 | 保留的最大历史帧数 |
| `--num-generations` | 1 | 采样次数（>1 用于多数表决） |

**评估逻辑**：
- 每个 episode 独立运行
- Agent 观察环境 → 构造 prompt → 大模型推理 → 解析动作 → 执行
- 支持**动作缓存**（pending_action_list）：一次生成的多个动作分步执行
- 早停机制：连续 25 步距离目标无变化，或超过 400 步，强制 stop

---

### 6. `eval_vllm.sh` — 评估（vLLM 加速版）

**功能**：通过调用 vLLM 推理服务进行评估，速度比 Transformers 版快 3-5 倍。

**调用命令**：
```bash
# 先启动 vLLM 服务
./scripts/start_vllm_server.sh

# 再运行评估
./scripts/eval_vllm.sh
```

**内部执行**：
```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8201/v1"

python src/eval/eval_vllm.py \
    --exp-config config/vln_r2r.yaml \
    --split-num 16 \
    --result-path eval_log/navida_r2r
```

**与 eval.sh 的区别**：
| 特性 | `eval.sh` | `eval_vllm.sh` |
|------|-----------|----------------|
| 推理方式 | 本地加载模型 | 调用 vLLM HTTP 服务 |
| 并行度 | 8 进程 | 16 进程 |
| 速度 | 较慢 | 快 3-5 倍 |
| 适用场景 | 模型调试 | 大规模评估 |

---

### 7. `start_vllm_server.sh` — 启动 vLLM 服务

**功能**：启动 vLLM 推理服务器，提供 OpenAI-compatible API。

**调用命令**：
```bash
./scripts/start_vllm_server.sh
```

**内部执行**：
```bash
vllm serve $MODEL_PATH --task generate \
    --trust-remote-code \
    --limit-mm-per-prompt image=99999 \
    --mm_processor_kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 \
    --port 8201 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

**关键参数**：
| 参数 | 说明 |
|------|------|
| `--limit-mm-per-prompt image=99999` | 每个 prompt 最多 99999 张图 |
| `--max-model-len 32768` | 最大上下文长度 |
| `--port 8201` | 服务端口 |
| `--gpu-memory-utilization 0.9` | GPU 显存利用率上限 |

---

### 8. `analyze_results.sh` — 结果分析

**功能**：汇总所有 episode 的评估结果，计算 SR、SPL、NE 等指标。

**调用命令**：
```bash
./scripts/analyze_results.sh
```

**内部执行**：
```bash
python src/eval/analyze_results.py --path eval_log/navida_r2r
```

**输出示例**：
```
Success rate: 614/1000 (0.614)
Oracle success rate: 695/1000 (0.695)
SPL: 547.123/1000 (0.547)
Distance to goal: 4.321
Path length: 12.456
ndtw: 0.723
```

---

### 9. `kill_eval.sh` — 终止评估

**功能**：强制终止所有正在运行的评估进程。

**使用场景**：评估脚本卡住或需要重新运行时。

---

### 10. DeepSpeed 配置文件

| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `zero1.json` | ZeRO-1：只优化器状态分片 | 单卡/小模型 |
| `zero2.json` | ZeRO-2：优化器+梯度分片 | **默认配置**，4-8 卡 |
| `zero3.json` | ZeRO-3：参数+梯度+优化器全分片 | 显存极度受限 |
| `zero3_offload.json` | ZeRO-3 + CPU Offload | 显存不足，用内存换 |

**ZeRO-2 配置详解**：
```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": {
        "stage": 2,                    // ZeRO 阶段
        "overlap_comm": true,          // 通信与计算重叠
        "contiguous_gradients": true   // 梯度连续存储
    }
}
```

---

## 🐍 核心源码详解 (`src/`)

### 数据处理模块 (`src/data/`)

#### `preprocess.py`

**功能**：将各数据集的原始格式统一为标准的 JSONL 格式。

**支持的数据集**：

| 函数 | 输入 | 输出 | 特殊处理 |
|------|------|------|----------|
| `process_r2r()` | `R2R_VLNCE_v1-3_preprocessed/train/*.json.gz` | `data/sub_dataset/r2r.jsonl` | 无 |
| `process_rxr()` | `RxR_VLNCE_v0/train/*.json.gz` | `data/sub_dataset/rxr.jsonl` | 过滤非英语指令 |
| `process_streamvlnrxr()` | `data/streamvln/RxR/annotations.json` | `data/sub_dataset/streamvln_rxr.jsonl` | 多指令列表 |
| `process_envdrop()` | `R2R_VLNCE_v1-3_preprocessed/envdrop/*.json.gz` | `data/sub_dataset/envdrop.jsonl` | 无 |
| `process_scalevln()` | `data/ScaleVLN/annotations.json` | `data/sub_dataset/scalevln.jsonl` | 多指令展开 |

**统一后的数据格式**：
```json
{
    "episode_id": 12345,
    "video_id": 12345,
    "instruction": "Go straight and turn left at the door",
    "actions": [1, 1, 1, 2, 3, 1, 0]
}
```

---

#### `extract_frame.py`

**功能**：在 Habitat 模拟器中执行专家轨迹，保存每步的 RGB 观测帧。

**核心函数**：

```python
def extract_data(result_queue, env_config, annotations, dataset, save_image, image_path):
    env = habitat.Env(env_config.habitat, dataset)   # 创建环境
    
    for episode in env.episodes:
        env.current_episode = episode
        observation = env.reset()
        step_id = 0
        
        while not env.episode_over:
            rgb = observation["rgb"]                    # 获取 RGB (640×480)
            rgb_frame = Image.fromarray(rgb)
            rgb_frame = rgb_frame.resize((320, 240))    # 压缩到 320×240
            rgb_frame.save(f"frame_{step_id}.jpg")      # 保存
            
            action = reference_actions.pop(0)           # 取出专家动作
            observation = env.step(action)              # 执行动作
            step_id += 1
```

**多进程并行**：
```python
# 将数据集分成 split_num 份，每份一个进程
for i in range(split_num):
    p = mp.Process(target=extract_data, args=(...))
    p.start()
```

---

#### `prepare_training_data.py`

**功能**：构造最终的训练样本，实现**动作分块**和**IDM 任务**。

**核心数据结构**：

```python
config = {
    'r2r': {
        'image_path': 'data/images/r2r',
        'annotation_path': 'data/sub_dataset/r2r.jsonl',
        'split_method': lambda x: x.split("_")[1].split(".")[0],  # 从文件名提取帧序号
    },
    # ... 其他数据集类似
}
```

**动作分块逻辑**：

```python
# 原始动作：[1,1,1,2,2,3,1,0]
# 分块过程：
#   第1步：forward 25 cm
#   第2步：prob=0.6 < 0.7，合并 → forward 50 cm
#   第3步：prob=0.8 > 0.7，不合并，且 count=1 < 2，追加 → forward 50 cm, forward 25 cm
#   第4步：动作不同(2≠1)，开启新 chunk → turn left 15 degree
# ...
```

**生成的训练样本格式**：

```json
{
    "system": "You are a helpful assistant.",
    "conversations": [
        {
            "from": "user",
            "value": "Imagine you are a robot... Your assigned task is: 'Go to the kitchen'",
            "image": ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]
        },
        {
            "from": "assistant",
            "value": "forward 75 cm, turn left 15 degree"
        }
    ],
    "action_history": ["forward 25 cm"],
    "episode_id": "1234",
    "task type": "vln"
}
```

**IDM 任务样本**：

```json
{
    "conversations": [
        {
            "from": "user",
            "value": "Imagine you are a robot... current view and goal view",
            "image": ["frame_0.jpg", "frame_1.jpg"]   // 只有两张图
        },
        {
            "from": "assistant",
            "value": "forward 25 cm"
        }
    ],
    "task type": "idm"
}
```

---

### 训练模块 (`src/train/`)

#### `train.py`

**功能**：使用 HuggingFace Transformers + DeepSpeed 微调 Qwen2.5-VL-3B。

**核心流程**：

```
1. 加载数据集（JSONL → HuggingFace Dataset）
2. 加载 Processor（图像预处理 + Tokenizer）
3. 加载模型（Qwen2_5_VLForConditionalGeneration）
4. 冻结视觉编码器，训练 merger + LLM
5. 使用 Trainer 训练
6. 保存模型和 Processor
```

**数据转换函数 `convert_example()`**：

```python
def convert_example(example):
    # 将 JSON 格式转换为 Qwen2.5-VL 的 message 格式
    messages = []
    
    # System prompt
    messages.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
    
    # VLN 任务：历史帧（最多8张）+ 当前帧 + 指令
    if example['task type'] == 'vln':
        content = [
            {"type": "text", "text": "Imagine you are a robot..."},
            {"type": "image", "image": "frame_0.jpg"},   # 历史帧（均匀采样）
            {"type": "image", "image": "frame_2.jpg"},
            ...
            {"type": "text", "text": "and an image of the current observation"},
            {"type": "image", "image": "frame_n.jpg"},   # 当前帧
            {"type": "text", "text": "Your assigned task is: '...'"}
        ]
    
    # IDM 任务：当前视图 + 目标视图
    elif example['task type'] == 'idm':
        content = [
            {"type": "text", "text": "Imagine you are a robot..."},
            {"type": "image", "image": "frame_i.jpg"},     # 当前视图
            {"type": "text", "text": "and an image of the goal view"},
            {"type": "image", "image": "frame_j.jpg"},     # 目标视图
            {"type": "text", "text": "Analyze the two images..."}
        ]
    
    messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "forward 50 cm"}]})
    return messages
```

**图像采样策略 `uniform_sample_with_ends()`**：

```python
def uniform_sample_with_ends(data, n):
    # 从历史帧中均匀采样 n 张，保留首尾
    # 例如 20 帧 → 采样 8 帧：[0, 2, 5, 8, 11, 14, 17, 19]
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]
```

**Collate 函数 `collate_fn()`**：

```python
def collate_fn(examples):
    # 1. 应用 chat template
    texts = [processor.apply_chat_template(messages, tokenize=False) for example in examples]
    
    # 2. 处理图像（resize 到 308×252）
    imgs = [item.resize((308, 252)) for item in imgs]
    
    # 3. 构建 batch
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    # 4. 构造 labels
    labels = batch["input_ids"].clone()
    labels[labels == pad_token_id] = -100       # padding 不计算损失
    labels[labels == image_token_id] = -100     # image token 不计算损失
    
    # 5. 只保留 assistant 回答部分的损失
    # 屏蔽 system prompt 和 user instruction
    for input_id, label, text in zip(batch["input_ids"], labels, texts):
        rounds = text.split('<|im_end|>\n<|im_start|>')
        sys_prompt = rounds[0]
        sys_prompt_len = len(processor.tokenizer(sys_prompt)['input_ids']) + 2
        label[:sys_prompt_len] = -100   # 屏蔽 system
        
        for instruction, response in zip(rounds[0::2], rounds[1::2]):
            instruction_len = len(processor.tokenizer(instruction)['input_ids']) + 6
            label[cur_len:cur_len + instruction_len] = -100   # 屏蔽 instruction
            cur_len += instruction_len + response_len
    
    batch["labels"] = labels
    return batch
```

**模型初始化**：

```python
# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)

# 冻结视觉编码器
for p in model.visual.parameters():
    p.requires_grad = False          # 冻结 ViT  backbone
for p in model.visual.merger.parameters():
    p.requires_grad = True           # 训练 merger（视觉-语言投影层）
```

---

### 评估模块 (`src/eval/`)

#### `eval.py`

**功能**：加载本地模型，在 Habitat 环境中执行导航评估。

**核心类 `NaVIDA_Agent`**：

```python
class NaVIDA_Agent(Agent):
    def __init__(self, model_path, lora_path, ...):
        # 加载模型（支持 LoRA）
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, ...)
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()   # 合并 LoRA 权重
        
        # 生成配置
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.2,          # 低温度，更确定性
            max_new_tokens=512,
            repetition_penalty=1.05
        )
    
    def act(self, observations, info, episode_id):
        # 1. 获取当前帧并保存历史
        rgb = observations["rgb"]
        rgb_ = Image.fromarray(rgb).resize((308, 252))
        self.rgb_list.append(rgb_)
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]   # 滑动窗口
        
        # 2. 动作缓存：如果还有 pending 动作，直接执行
        if len(self.pending_action_list) != 0:
            return {"action": self.pending_action_list.pop(0)}
        
        # 3. 构造 prompt（历史帧 + 当前帧 + 指令）
        content = [
            {"type": "text", "text": "Imagine you are a robot..."},
            # 历史帧（最多8张，base64编码）
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}
            for item in self.uniform_sample_with_ends(self.rgb_list[:-1], 8)
        ]
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})  # 当前帧
        content.append({"type": "text", "text": f"Your assigned task is: '{instruction}'"})
        
        self.conversations.append({"role": "user", "content": content})
        
        # 4. 大模型推理
        navigation = self.predict_inference()
        
        # 5. 解析输出
        result = self.extract_multi_result(navigation)
        # 如："forward 50 cm, turn left 15 degree" → [[1, 50], [2, 15]]
        
        # 6. 转换为离散动作并缓存
        for action_index, numeric in result:
            if action_index == 1:   # forward
                for _ in range(min(3, round(numeric/25))):
                    self.pending_action_list.append(1)
            elif action_index == 2: # turn left
                for _ in range(min(3, round(numeric/15))):
                    self.pending_action_list.append(2)
            elif action_index == 3: # turn right
                for _ in range(min(3, round(numeric/15))):
                    self.pending_action_list.append(3)
        
        return {"action": self.pending_action_list.pop(0)}
```

**输出解析 `extract_result()`**：

```python
def extract_result(self, output):
    # 提取 <answer> 标签内容
    output_match = re.search(r'<answer>(.*?)</answer>', output)
    output = output_match.group(1).strip() if output_match else output.strip()
    
    output = output.lower()
    if "stop" in output:
        return 0, None
    elif "forward" in output:
        match = re.search(r'-?\d+', output)
        return 1, float(match.group())   # (动作ID, 距离/角度)
    elif "left" in output:
        match = re.search(r'-?\d+', output)
        return 2, float(match.group())
    elif "right" in output:
        match = re.search(r'-?\d+', output)
        return 3, float(match.group())
    return None, None
```

---

#### `eval_vllm.py`

**功能**：通过 OpenAI API 调用 vLLM 服务进行推理，逻辑与 `eval.py` 相同，只是推理方式不同。

**关键区别**：

```python
# eval.py：本地推理
outputs = self.model.generate(**prompt_inputs, generation_config=self.generation_config)

# eval_vllm.py：远程调用
outputs = self.client.chat.completions.create(
    messages=self.conversations,
    model=self.model,
    max_completion_tokens=512,
    temperature=0.2
)
```

---

#### `analyze_results.py`

**功能**：汇总所有 episode 的评估结果，计算各项指标。

```python
for j in jsons:
    data = json.load(open(j))
    succ += int(data['success'])
    spl += data['spl']
    distance_to_goal += data['distance_to_goal']
    oracle_succ += int(data['oracle_success'])
    path_length += data['path_length']
    if 'ndtw' in data:
        ndtw += data['ndtw']

print(f'Success rate: {succ}/{len(jsons)} ({succ/len(jsons):.3f})')
print(f'Oracle success rate: {oracle_succ}/{len(jsons)} ({oracle_succ/len(jsons):.3f})')
print(f'SPL: {spl/len(jsons):.3f}')
print(f'Distance to goal: {distance_to_goal/len(jsons):.3f}')
print(f'ndtw: {ndtw/len(jsons):.3f}')
```

---

## 🏠 Habitat 扩展 (`habitat_extensions/`)

### `task.py`

**功能**：定义自定义的 VLN 数据集和 Episode 结构。

**核心类**：

```python
@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = None
    reference_path: Optional[List[List[float]]] = None
    instruction: ExtendedInstructionData = attr.ib(default=None)
    trajectory_id: Optional[Union[int, str]] = None

@registry.register_dataset(name="R2RVLNCE-v1")
class VLNDatasetV1(Dataset):
    def from_json(self, json_str, scenes_dir=None):
        # 从 gzip 压缩的 JSON 加载 episode 数据
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)
```

---

### `measures.py`

**功能**：实现 VLN 领域特有的评估指标。

| 指标类 | UUID | 说明 | 公式 |
|--------|------|------|------|
| `PathLength` | `path_length` | 实际路径长度 | 累加每步欧氏距离 |
| `OracleNavigationError` | `oracle_navigation_error` | 最短导航误差 | min(距离目标) |
| `OracleSuccess` | `oracle_success` | Oracle 成功率 | I(ONE ≤ 3m) |
| `OracleSPL` | `oracle_spl` | Oracle SPL | max(SPL) over path |
| `StepsTaken` | `steps_taken` | 实际步数 | 动作执行次数 |
| `NDTW` | `ndtw` | 归一化动态时间规整 | exp(-DTW/(len×threshold)) |
| `SDTW` | `sdtw` | 成功加权 nDTW | Success × nDTW |

**nDTW 计算**：

```python
def update_metric(self, ...):
    current_position = self._sim.get_agent_state().position.tolist()
    self.locations.append(current_position)
    
    # 计算 DTW 距离（动态时间规整）
    dtw_distance = fastdtw(self.locations, self.gt_locations, dist=euclidean_distance)[0]
    
    # 归一化
    nDTW = np.exp(-dtw_distance / (len(self.gt_locations) * self._success_distance))
    self._metric = nDTW
```

---

## 🚀 快速开始指南

### 环境准备

```bash
# 1. 创建环境
conda create -n navida python=3.10
conda activate navida

# 2. 安装 Habitat
pip install -e habitat-lab
pip install -e habitat-baselines

# 3. 安装 NaVIDA 依赖
pip install peft trl==0.16.0 transformers==4.50.3 tensorboardx qwen_vl_utils deepspeed
pip install numpy==1.24.0 tqdm opencv-python vllm==0.9.1 torch torchvision
pip install flash-attn --no-build-isolation --no-cache-dir
```

### 数据准备

```bash
# 1. 下载场景数据（MP3D/HM3D）放到 data/scene_datasets/

# 2. 下载 VLN-CE episode 数据放到 data/

# 3. 运行数据预处理流水线
./scripts/preprocess.sh           # 格式统一
./scripts/extract_frame.sh        # 渲染帧（约需数小时）
./scripts/prepare_training_data.sh # 构造训练样本
```

### 训练

```bash
# 修改 train.sh 中的 model_name_or_path 和 dataset_name
./scripts/train.sh
```

### 评估

```bash
# 方式1：本地模型（适合调试）
./scripts/eval.sh

# 方式2：vLLM 加速（适合大规模评估）
./scripts/start_vllm_server.sh   # 终端1：启动服务
./scripts/eval_vllm.sh           # 终端2：运行评估
```

---

## 📊 指标说明

| 指标 | 全称 | 含义 | 方向 |
|------|------|------|------|
| **SR** | Success Rate | 成功率（最终距离目标 ≤ 3m） | ↑ 越高越好 |
| **SPL** | Success weighted by Path Length | 成功率 × 最短路径/实际路径 | ↑ 越高越好 |
| **NE** | Navigation Error | 最终位置到目标的距离 | ↓ 越低越好 |
| **OS** | Oracle Success | 路径上是否曾接近目标 | ↑ 越高越好 |
| **nDTW** | normalized Dynamic Time Warping | 路径与专家路径的相似度 | ↑ 越高越好 |
| **PL** | Path Length | 实际走过的路径长度 | - |

---

## 📝 常见问题

**Q: 为什么训练时冻结视觉编码器？**
A: Qwen2.5-VL 的视觉编码器已经在大规模数据上预训练，泛化能力足够。冻结它可以减少可训练参数，防止过拟合，同时加快训练速度。

**Q: 动作分块的最大动作数为什么是 3？**
A: 这是超参数，平衡了"减少模型调用次数"和"预测精度"两个目标。太多动作会导致预测不准确，太少则无法体现分块优势。

**Q: IDM 任务和 VLN 任务的数据比例是多少？**
A: 默认 1:1（`--task_type vln idm`），可以根据需要调整。

**Q: 评估时为什么需要 `--split-num` 和 `--split-id`？**
A: 为了并行加速评估。将数据集分成多份，每份由一个独立进程处理，充分利用多 GPU。

---

## 📚 相关资源

- 📄 论文：[arXiv:2601.18188](https://arxiv.org/abs/2601.18188)
- 🤗 模型：[HuggingFace](https://huggingface.co/waynechu/NaVIDA)
- 🏠 Habitat：[habitat-sim](https://github.com/facebookresearch/habitat-sim) / [habitat-lab](https://github.com/facebookresearch/habitat-lab)
- 🌐 Qwen2.5-VL：[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
