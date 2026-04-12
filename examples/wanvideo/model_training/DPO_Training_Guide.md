# Wan2.1-T2V-1.3B DPO LoRA 训练 — 代码修改记录

## 1. 概述

本文档记录了为 DiffSynth-Studio 项目增加 **DPO（Direct Preference Optimization）训练**所做的全部代码修改。目标模型为 **Wan2.1-T2V-1.3B**，使用 **LoRA** 微调。

DPO 的核心思想：给定同一个 prompt 下的偏好视频对（chosen/rejected），通过对比学习让模型生成更符合人类偏好的视频。相比 SFT 仅学习单个样本，DPO 利用 reference 模型（基座 DiT，LoRA 关闭）作为基线，构造隐式奖励来引导 LoRA 参数的优化方向。

---

## 2. 修改文件总览

| # | 文件路径 | 修改类型 | 说明 |
|---|---------|---------|------|
| 1 | `diffsynth/diffusion/loss.py` | **新增函数** | 新增 `FlowMatchDPOLoss` |
| 2 | `diffsynth/diffusion/parsers.py` | **新增函数 + 修改** | 新增 `add_dpo_config()`，修改 `add_general_config()` |
| 3 | `diffsynth/diffusion/training_module.py` | **修改** | `loss_required_params` 增加 DPO 字段 |
| 4 | `diffsynth/core/data/unified_dataset.py` | **新增类** | 新增 `DPOVideoDataset` |
| 5 | `diffsynth/diffusion/runner.py` | **新增函数** | 新增 `launch_dpo_training_task` |
| 6 | `diffsynth/diffusion/__init__.py` | **修改** | 导出 `launch_dpo_training_task` |
| 7 | `examples/wanvideo/model_training/train.py` | **多处修改** | DPO 数据处理、前向逻辑、任务注册 |

---

## 3. 各文件修改详情

### 3.1 `diffsynth/diffusion/loss.py` — 新增 `FlowMatchDPOLoss`

**位置**：在 `DirectDistillLoss` 函数之后、`TrajectoryImitationLoss` 类之前（第 73~137 行）

**新增内容**：`FlowMatchDPOLoss(pipe, dpo_beta=0.1, **inputs)` 函数

**核心逻辑**：

```
1. 随机采样一个 timestep（chosen 和 rejected 共享同一个 timestep）
2. 生成噪声，对 chosen/rejected latents 分别加噪
3. 当前模型（LoRA 开启）对 chosen 和 rejected 分别做前向预测
4. Reference 模型（LoRA 关闭）在 torch.no_grad() 下对 chosen 和 rejected 分别做前向预测
5. 计算 DPO loss：
   - model_loss_chosen  = MSE(pred_chosen, target_chosen)
   - model_loss_rejected = MSE(pred_rejected, target_rejected)
   - ref_loss_chosen    = MSE(ref_pred_chosen, target_chosen)
   - ref_loss_rejected  = MSE(ref_pred_rejected, target_rejected)
   - logit = beta * ((model_loss_rejected - ref_loss_rejected) - (model_loss_chosen - ref_loss_chosen))
   - loss = -logsigmoid(logit) * training_weight(timestep)
```

**关键设计点**：
- 通过 peft 的 `dit.disable_adapter_layers()` / `dit.enable_adapter_layers()` 切换 LoRA 开关，实现 ref 模型，无需额外加载模型权重
- chosen 和 rejected 使用相同噪声（如形状一致），减少方差
- 支持 `first_frame_latents`（I2V 场景），与现有 SFT loss 保持一致的处理方式
- 传入 `model_fn` 前过滤掉 DPO 特有的字段（`input_latents_chosen`/`input_latents_rejected`/`dpo_beta`）

---

### 3.2 `diffsynth/diffusion/parsers.py` — 新增 DPO 参数

**修改 1**：新增 `add_dpo_config` 函数（第 63~65 行）

```python
def add_dpo_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                        help="DPO temperature parameter. Controls preference learning strength.")
    return parser
```

**修改 2**：在 `add_general_config` 中调用 `add_dpo_config`（第 74 行）

```python
def add_general_config(parser):
    ...
    parser = add_gradient_config(parser)
    parser = add_dpo_config(parser)      # 新增此行
    return parser
```

**效果**：所有使用 `add_general_config` 的训练脚本自动获得 `--dpo_beta` 参数。

---

### 3.3 `diffsynth/diffusion/training_module.py` — 扩展参数白名单

**修改位置**：`split_pipeline_units` 方法的 `loss_required_params` 默认值（第 263 行）

**修改前**：
```python
loss_required_params=("input_latents", "max_timestep_boundary", "min_timestep_boundary",
                      "first_frame_latents", "video_latents", "audio_input_latents",
                      "num_inference_steps"),
```

**修改后**：
```python
loss_required_params=("input_latents", "max_timestep_boundary", "min_timestep_boundary",
                      "first_frame_latents", "video_latents", "audio_input_latents",
                      "num_inference_steps",
                      "input_latents_chosen", "input_latents_rejected"),
```

**原因**：当使用 `remove_unnecessary_params=True` 的数据缓存模式时，此白名单决定哪些参数会被保留。如果不加入 DPO 的两个 latent 字段，它们会在缓存阶段被清除。

---

### 3.4 `diffsynth/core/data/unified_dataset.py` — 新增 `DPOVideoDataset`

**位置**：文件末尾（第 121~165 行），在 `UnifiedDataset` 类之后

**新增内容**：

```python
class DPOVideoDataset(torch.utils.data.Dataset):
```

**与 `UnifiedDataset` 的区别**：

| 对比项 | `UnifiedDataset` | `DPOVideoDataset` |
|--------|-----------------|-------------------|
| 数据格式 | `{"video": "...", "prompt": "..."}` | `{"video_chosen": "...", "video_rejected": "...", "prompt": "..."}` |
| 视频加载 | 通过 `data_file_keys` + `main_data_operator` | 固定读取 `video_chosen` 和 `video_rejected`，使用 `video_operator` |
| 缓存支持 | 支持 `load_from_cache` | `load_from_cache = False`（不支持缓存模式） |
| 返回值 | `{"video": List[PIL.Image], "prompt": str}` | `{"video_chosen": List[PIL.Image], "video_rejected": List[PIL.Image], "prompt": str}` |

**metadata 格式要求**（JSON 示例）：
```json
[
  {
    "prompt": "一只猫在花园里追蝴蝶",
    "video_chosen": "data/good_001.mp4",
    "video_rejected": "data/bad_001.mp4"
  }
]
```

---

### 3.5 `diffsynth/diffusion/runner.py` — 新增 `launch_dpo_training_task`

**位置**：在 `launch_data_process_task` 之后、`initialize_deepspeed_gradient_checkpointing` 之前（第 75~114 行）

**与 `launch_training_task` 的区别**：

| 对比项 | `launch_training_task` | `launch_dpo_training_task` |
|--------|----------------------|---------------------------|
| 梯度裁剪 | 无 | `accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| 缓存支持 | `if dataset.load_from_cache: model({}, inputs=data)` | 不支持缓存模式，始终 `model(data)` |

**增加梯度裁剪的原因**：DPO 每步执行 4 次 DiT 前向传播，梯度规模可能较大，裁剪有助于训练稳定性。

---

### 3.6 `diffsynth/diffusion/__init__.py` — 导出新函数

**修改前**：
```python
from .runner import launch_training_task, launch_data_process_task
```

**修改后**：
```python
from .runner import launch_training_task, launch_data_process_task, launch_dpo_training_task
```

---

### 3.7 `examples/wanvideo/model_training/train.py` — 主训练脚本

这是修改量最大的文件，包含以下变更：

#### 3.7.1 新增导入（第 3 行）

```python
from diffsynth.core.data.unified_dataset import DPOVideoDataset
```

#### 3.7.2 `WanTrainingModule.__init__` 新增 `dpo_beta` 参数（第 27 行）

```python
def __init__(self, ..., dpo_beta=0.1):
```

#### 3.7.3 `task_to_loss` 字典注册 DPO 任务（第 59、64~65 行）

新增三条映射：
```python
"dpo:data_process": lambda pipe, *args: args,
"dpo":       lambda ...: FlowMatchDPOLoss(pipe, dpo_beta=self.dpo_beta, ...),
"dpo:train": lambda ...: FlowMatchDPOLoss(pipe, dpo_beta=self.dpo_beta, ...),
```

#### 3.7.4 存储 `dpo_beta`（第 69 行）

```python
self.dpo_beta = dpo_beta
```

#### 3.7.5 修改 `get_pipeline_inputs` 增加 DPO 分支（第 86~88 行）

```python
def get_pipeline_inputs(self, data):
    if self.task.startswith("dpo"):
        return self._get_dpo_pipeline_inputs(data)
    # 原有 SFT 逻辑不变...
```

#### 3.7.6 新增 `_get_dpo_pipeline_inputs` 方法（第 113~135 行）

从 `data["video_chosen"]` 和 `data["video_rejected"]` 构建管线输入。chosen 视频作为 `input_video`（走正常管线编码），rejected 视频暂存为 `input_video_rejected`（后续在 `_forward_dpo` 中单独编码）。

#### 3.7.7 修改 `forward` 增加 DPO 分支（第 137~145 行）

```python
def forward(self, data, inputs=None):
    ...
    if self.task.startswith("dpo") and not self.task.endswith(":data_process"):
        return self._forward_dpo(inputs)
    # 原有逻辑不变...
```

#### 3.7.8 新增 `_forward_dpo` 方法（第 147~170 行）

DPO 前向的核心流程：
1. 从 `inputs_shared` 中取出并移除 `input_video_rejected`
2. 正常运行管线单元，编码 chosen 视频 → `input_latents`
3. 在 `torch.no_grad()` 下，用 `pipe.preprocess_video` + `pipe.vae.encode` 单独编码 rejected 视频
4. 将 `input_latents_chosen` 和 `input_latents_rejected` 存入 `inputs_shared`
5. 调用 `FlowMatchDPOLoss` 计算 loss

#### 3.7.9 模型实例化传入 `dpo_beta`（第 245 行）

```python
model = WanTrainingModule(..., dpo_beta=args.dpo_beta)
```

#### 3.7.10 `launcher_map` 注册 DPO 任务（第 254、259~260 行）

```python
"dpo:data_process": launch_data_process_task,
"dpo":              launch_dpo_training_task,
"dpo:train":        launch_dpo_training_task,
```

#### 3.7.11 主函数数据集创建增加 DPO 分支（第 193~223 行）

```python
video_operator = UnifiedDataset.default_video_operator(...)
if args.task.startswith("dpo"):
    dataset = DPOVideoDataset(
        base_path=..., metadata_path=..., repeat=..., video_operator=video_operator,
    )
else:
    dataset = UnifiedDataset(...)  # 原有逻辑不变
```

---

## 4. DPO 训练数据流全图

```
┌─────────────────────────────────────────────────────────────┐
│                     DPOVideoDataset                         │
│  metadata.json:                                             │
│  {"prompt": "...", "video_chosen": "a.mp4",                 │
│   "video_rejected": "b.mp4"}                                │
│                                                             │
│  __getitem__ → {                                            │
│      "prompt": str,                                         │
│      "video_chosen": List[PIL.Image],                       │
│      "video_rejected": List[PIL.Image]                      │
│  }                                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              WanTrainingModule.forward(data)                 │
│                                                             │
│  1. _get_dpo_pipeline_inputs(data)                          │
│     → inputs_shared["input_video"] = video_chosen           │
│     → inputs_shared["input_video_rejected"] = video_rejected│
│     → inputs_posi["prompt"] = prompt                        │
│                                                             │
│  2. _forward_dpo(inputs)                                    │
│     ┌────────────────────────────────────────────┐          │
│     │ 取出 input_video_rejected                   │          │
│     │ 运行管线单元 → chosen 视频经过:               │          │
│     │   TextEncoder(prompt) → context             │          │
│     │   VAE.encode(chosen)  → input_latents       │          │
│     │ 单独 VAE.encode(rejected) → rejected_latents│          │
│     └────────────────────────────────────────────┘          │
│                                                             │
│  3. FlowMatchDPOLoss(pipe, dpo_beta, ...)                   │
│     ┌────────────────────────────────────────────┐          │
│     │ 采样 timestep, 生成噪声                      │          │
│     │ noisy_chosen  = add_noise(chosen_latents)   │          │
│     │ noisy_rejected = add_noise(rejected_latents)│          │
│     │                                             │          │
│     │ [LoRA ON]  pred_chosen,  pred_rejected      │  ×2 fwd │
│     │ [LoRA OFF] ref_chosen,   ref_rejected       │  ×2 fwd │
│     │                                             │          │
│     │ logit = β * ((loss_rej - ref_rej)           │          │
│     │            - (loss_cho - ref_cho))           │          │
│     │ loss = -logsigmoid(logit) * weight           │          │
│     └────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
          accelerator.backward(loss) → 梯度更新 LoRA 参数
```

---

## 5. 训练启动命令

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --task "dpo" \
  --dataset_base_path "./data/dpo_dataset" \
  --dataset_metadata_path "./data/dpo_dataset/metadata.json" \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --learning_rate 1e-5 \
  --dpo_beta 0.1 \
  --num_epochs 3 \
  --save_steps 20 \
  --output_path "./output/dpo_lora" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --num_frames 81 \
  --height 480 \
  --width 832
  新增加的训练过程中预览
  accelerate launch examples/wanvideo/model_training/train.py \
  ... \
  --preview_steps 200 \
  --preview_prompt "a cat sitting on a boat" \
  --preview_num_inference_steps 8 \
  --preview_num_frames 81 \
  --preview_fps 15

```

---

## 6. 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--dpo_beta` | 0.05 ~ 0.5 | 从 0.1 开始；不稳定则减小 |
| `--learning_rate` | 1e-6 ~ 5e-5 | DPO 建议比 SFT 更小的学习率 |
| `--lora_rank` | 16 ~ 64 | 秩越大适应能力越强，但显存也越大 |
| `--gradient_accumulation_steps` | 4 ~ 16 | DPO 每步 4 次前向，建议加大累积步数 |
| `--num_epochs` | 1 ~ 5 | DPO 容易过拟合，不宜太多 epoch |

---

## 7. 显存注意事项

DPO 每步需要 **4 次 DiT 前向**（chosen×当前模型 + chosen×ref模型 + rejected×当前模型 + rejected×ref模型），其中只有 2 次（当前模型）需要计算梯度。显存约为 SFT 的 2~3 倍。

优化手段：
- `--use_gradient_checkpointing`（已强制开启）
- `--gradient_accumulation_steps 4` 或更大

  
