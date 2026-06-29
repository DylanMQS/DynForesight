# Wan2.2-TI2V-5B VAE Encoder 特征提取配置说明

本文档说明如何**仅使用 Wan2.2-TI2V-5B 的 VAE encoder** 提取视频/图像特征，无需加载完整生成模型。

## 1. 环境与依赖

已配置 `requirements.txt` 后，建议：

```bash
cd /path/to/Wan2.2
pip install -r requirements.txt
```

- **Python**：建议 3.10+
- **PyTorch**：>= 2.4.0
- **CUDA**：需与 PyTorch 匹配（VAE 在 GPU 上运行更合适）
- 若 `flash_attn` 安装失败，可先装其余依赖，最后单独安装 `flash_attn`。

**只做 VAE 特征提取时**：按 **README** 的 `pip install -r requirements.txt` 即可，无需再按 INSTALL.md 做 `pip install .`。INSTALL.md 是给「把 Wan2.2 当作包安装」或用 Poetry 时用的；README 是主安装说明，按 README 装就够用。  
运行 `extract_vae_features.py` 时在 **Wan2.2 仓库根目录** 下执行即可（脚本会直接加载 VAE 模块，不依赖 `import wan`，因此不会触发 S2V 等可选依赖如 librosa）。

## 2. 模型权重：只下 VAE

TI2V-5B 的 VAE 权重名为 **`Wan2.2_VAE.pth`**，通常与 TI2V-5B 一起发布。

**方式 A：只下载 VAE（推荐，体积小）**

- Hugging Face：
  ```bash
  pip install "huggingface_hub[cli]"
  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir ./Wan2.2-TI2V-5B
  ```
- ModelScope：
  ```bash
  pip install modelscope
  modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B
  ```
  若 CLI 不支持单文件，可先整包下载，再只使用其中的 `Wan2.2_VAE.pth`。

**方式 B：下载完整 TI2V-5B**

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

VAE 路径为：`./Wan2.2-TI2V-5B/Wan2.2_VAE.pth`。

## 3. VAE 配置参数（与 TI2V-5B 一致）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 权重文件 | `Wan2.2_VAE.pth` | 文件名固定 |
| `vae_stride` | `(4, 16, 16)` | 时间 4×、高 16×、宽 16× 下采样 |
| 潜在维度 `z_dim` | 48 | 每个时空位置的 latent 通道数 |
| 输入数值范围 | **[-1, 1]** | 与 PyTorch `ToTensor` + `(x-0.5)/0.5` 一致 |

编码后 latent 形状（单段视频）：

- 输入：`(C, T, H, W)`，例如 C=3, T=帧数, H,W=高宽  
- 输出：`(z_dim, T', H', W')`，其中  
  - `T' = (T - 1) // 4 + 1`（时间 stride 4）  
  - `H' = H // 16`, `W' = W // 16`（空间 stride 16）  
  - 内部还有 patch_size=2 的 patchify，**H、W 建议为 32 的倍数**（如 704、1280 等），避免尺寸不对齐。

## 4. 代码用法：只加载 VAE 并编码

```python
import os
import torch
from wan.modules.vae2_2 import Wan2_2_VAE

# 1) 路径：TI2V-5B 目录下必须有 Wan2.2_VAE.pth
ckpt_dir = "./Wan2.2-TI2V-5B"
vae_pth = os.path.join(ckpt_dir, "Wan2.2_VAE.pth")
assert os.path.isfile(vae_pth), f"VAE weight not found: {vae_pth}"

# 2) 只初始化 VAE（不加载 DiT/T5）
device = "cuda"  # 或 "cpu"
vae = Wan2_2_VAE(
    vae_pth=vae_pth,
    device=device,
    dtype=torch.bfloat16,  # 可选，与 TI2V-5B 推理一致
)
vae.model.eval()

# 3) 准备输入：list of tensors，每个 (C, T, H, W)，范围 [-1, 1]
#    单帧图像可视为 (C, 1, H, W)
video_tensor = ...  # shape: (3, T, H, W), dtype float32, range [-1, 1]

latents = vae.encode([video_tensor])  # list of tensors
z = latents[0]  # shape: (48, T', H', W')
```

- **图像**：先转为 `(3, 1, H, W)`，数值缩放到 [-1, 1]，再 `vae.encode([img_tensor])`。  
- **视频**：维度 `(3, T, H, W)`，H/W 建议 32 的倍数，再 `vae.encode([video_tensor])`。

仓库内已提供脚本 **`extract_vae_features.py`**，可从视频/图像文件读入、做简单预处理并调用上述 VAE 编码，详见脚本内参数说明。

## 5. 与 generate.py 的对应关系

- 完整 TI2V-5B 推理（`generate.py --task ti2v-5B`）会加载：
  - T5、DiT、**以及** `Wan2_2_VAE`（见 `wan/textimage2video.py`）。
- VAE 配置来自 **`wan/configs/wan_ti2v_5B.py`**：
  - `vae_checkpoint = 'Wan2.2_VAE.pth'`
  - `vae_stride = (4, 16, 16)`
- 仅做特征提取时，只需加载 **`Wan2_2_VAE`** 与 `Wan2.2_VAE.pth`，无需 T5/DiT。

## 6. 常见问题

- **找不到 `wan` 模块**：在 Wan2.2 仓库根目录执行，或 `export PYTHONPATH=/path/to/Wan2.2:$PYTHONPATH`。
- **CUDA OOM**：减小 batch（一次只 encode 一条）、或降低分辨率/帧数，或使用 `dtype=torch.float16`。
- **尺寸错误**：保证 H、W 为 32 的倍数；时间维 T 任意，编码时按 `(T-1)//4+1` 得到 T'。

按上述配置即可单独使用 Wan2.2-TI2V-5B 的 VAE encoder 做特征提取。
