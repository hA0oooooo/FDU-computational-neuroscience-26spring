# Cross-Modal Brain MRI Translation: T1 <-> T2-FLAIR

本项目研究脑 MRI 中 T1 与 T2-FLAIR 之间的跨模态生成问题：在只有一种模态可用时，利用深度生成模型预测同一受试者的另一种模态。实验以 3D residual U-Net 为主要模型，在 bottleneck 中加入 Transformer block 建模较长程的三维空间依赖，并使用 L1、SSIM 和 gradient loss 的组合同时约束强度、结构相似性和局部边界细节。为了比较三维建模和切片建模的差异，项目还实现了一个 2.5D U-Net 作为对比：输入中心切片附近 7 张连续切片，先通过浅层 3D encoder 融合局部 z 轴上下文，再用 2D decoder 预测中心切片。根目录保留训练和评估入口，模型、数据、损失、推理和指标计算拆分在 src 目录下。

### 数据

数据位于 dataset/cn_project_t1t2，共 600 个配对病例，固定划分训练集 480 例，验证集 60 例，测试集 60 例。每个病例目录下包含 `T1.nii.gz` 和 `T2_FLAIR.nii.gz` 两个模态，`paired_600_manifest.csv` 记录了病例 ID 和原始文件来源。实验不做离线预处理，训练时直接读取 NIfTI，按训练集统计的模态级 mean/std 做 z-score normalization，评估时再反归一化回原图像空间计算指标。

599/600 个病例的 voxel spacing 为 1.0 x 1.0 x 1.0 mm，1 个病例为 1.0 x 0.746 x 0.746 mm，T1 和 T2-FLAIR 在每个病例内已经具有相同空间维度，600 个病例的体数据维度统计如下：

| 维度 | min | median | mean | max |
|---|---:|---:|---:|---:|
| D | 156 | 173 | 173.5 | 208 |
| H | 203 | 229 | 229.8 | 256 |
| W | 166 | 203 | 204.5 | 256 |

评估指标在反归一化后的完整 3D volume 上计算，`MAE = mean(|pred - target|)` 表示平均绝对误差，数值越小代表预测强度越接近真实图像；`MSE = mean((pred - target)^2)` 对大误差更敏感；`PSNR = 10 * log10(data_range^2 / MSE)` 用 target 的强度范围衡量重建信噪比；SSIM 衡量结构相似性，越接近 1 表示局部对比度、亮度和结构越接近真实图像。值得注意的是，MAE 和 MSE 依赖 target 模态的原始强度尺度，因此 T1 -> T2-FLAIR 和 T2-FLAIR -> T1 的 MAE 不适合直接横向比较，PSNR 和 SSIM 更适合判断跨方向的重建质量。

### 方法

##### 3D U-Net

3D 方法直接使用三维 patch 学习跨模态映射，可以同时建模轴向、冠状面和矢状面的空间上下文，损失函数设置为 `1.0 * L1 + 0.4 * SSIM loss + 0.2 * gradient loss`。

训练输入采样与标准化：

```text
source volume [D, H, W], target volume [D, H, W]
-> sample the same random 3D crop from source and target
-> source patch [64, 128, 128], target patch [64, 128, 128]
-> normalize by modality-level train-set z-score
-> source tensor [B, 1, 64, 128, 128], target tensor [B, 1, 64, 128, 128]
```

Residual block 指两层 3x3x3 Conv + InstanceNorm + LeakyReLU，并把输入通过 identity 或 1x1x1 Conv 加回输出。3D U-Net 维度流：

```text
input [B, 1, 64, 128, 128]
-> stem 3x3x3 conv: [B, 24, 64, 128, 128]
-> enc1 residual block: x1 [B, 24, 64, 128, 128]

-> stride-2 3x3x3 conv: [B, 48, 32, 64, 64]
-> enc2 residual block: x2 [B, 48, 32, 64, 64]

-> stride-2 3x3x3 conv: [B, 96, 16, 32, 32]
-> enc3 residual block: x3 [B, 96, 16, 32, 32]

-> stride-2 3x3x3 conv: [B, 192, 8, 16, 16]
-> enc4 residual block: x4 [B, 192, 8, 16, 16]

-> stride-2 3x3x3 conv: [B, 384, 4, 8, 8]
-> bottleneck residual block: [B, 384, 4, 8, 8]
-> flatten to 256 tokens
-> 2-layer Transformer encoder
-> reshape: x5 [B, 384, 4, 8, 8]

-> transposed conv up: [B, 192, 8, 16, 16]
-> concat with x4: [B, 384, 8, 16, 16]
-> decoder residual block: [B, 192, 8, 16, 16]

-> transposed conv up: [B, 96, 16, 32, 32]
-> concat with x3: [B, 192, 16, 32, 32]
-> decoder residual block: [B, 96, 16, 32, 32]

-> transposed conv up: [B, 48, 32, 64, 64]
-> concat with x2: [B, 96, 32, 64, 64]
-> decoder residual block: [B, 48, 32, 64, 64]

-> transposed conv up: [B, 24, 64, 128, 128]
-> concat with x1: [B, 48, 64, 128, 128]
-> decoder residual block: [B, 24, 64, 128, 128]

-> 1x1x1 conv
-> prediction [B, 1, 64, 128, 128]
```

由于每个病例的完整体数据尺寸不同，评估时用滑窗覆盖完整体数据，滑窗窗口大小和训练 patch 一致，为 [64, 128, 128]，stride 为 [32, 64, 64]，因此相邻窗口约有 50% overlap，如果最后一个窗口不能正好覆盖到边界，就额外加入 `size - patch` 作为最后起点，保证整个 volume 都被覆盖。例如一个 test case 的体数据尺寸为 [181, 218, 223]，对应起点数量为 D 方向 5 个、H 方向 3 个、W 方向 3 个，总窗口数为 5 x 3 x 3 = 45，3D 推理每个病例约使用 24 到 54 个窗口，重叠区域对所有覆盖该 voxel 的窗口预测取平均，因此边界和窗口交界处会更平滑稳定。

##### 2.5D U-Net

2.5D 方法输入中心切片附近的 7 张 T1 切片，先用浅层 3D encoder 融合局部厚度信息，再用 2D U-Net decoder 输出中心切片，这样保留了更大的训练样本量，同时比纯 2D 多利用一点 z 轴上下文。训练输入流：

```text
source volume [D, H, W], target volume [D, H, W]
-> choose center slice z, source stack [z-3, z-2, z-1, z, z+1, z+2, z+3]
-> target center slice z, same H/W crop from source stack and target slice
-> source crop [7, 160, 160], target crop [1, 160, 160]
-> source tensor [B, 7, 160, 160], target tensor [B, 1, 160, 160]
```

2.5D U-Net 前两层 3D Conv 只负责在 7 张相邻切片之间融合局部 z 轴信息，随后用 depth-collapse Conv 把 depth 维度压到 1，再进入标准 2D encoder-decoder。2.5D U-Net 维度流：

```text
input [B, 7, 160, 160]
-> unsqueeze depth channel: [B, 1, 7, 160, 160]
-> shallow 3D conv stem: [B, 48, 7, 160, 160]
-> depth-collapse conv over 7 slices: [B, 48, 1, 160, 160]
-> squeeze depth: [B, 48, 160, 160]
-> enc1 2D residual block: x1 [B, 48, 160, 160]

-> stride-2 2D conv: [B, 96, 80, 80]
-> enc2 2D residual block: x2 [B, 96, 80, 80]

-> stride-2 2D conv: [B, 192, 40, 40]
-> enc3 2D residual block: x3 [B, 192, 40, 40]

-> stride-2 2D conv: [B, 384, 20, 20]
-> enc4 2D residual block: x4 [B, 384, 20, 20]

-> stride-2 2D conv: [B, 768, 10, 10]
-> bottleneck 2D residual block: [B, 768, 10, 10]
-> flatten to 100 tokens
-> 1-layer Transformer encoder
-> reshape: x5 [B, 768, 10, 10]

-> transposed conv up: [B, 384, 20, 20]
-> concat with x4: [B, 768, 20, 20]
-> decoder 2D residual block: [B, 384, 20, 20]

-> transposed conv up: [B, 192, 40, 40]
-> concat with x3: [B, 384, 40, 40]
-> decoder 2D residual block: [B, 192, 40, 40]

-> transposed conv up: [B, 96, 80, 80]
-> concat with x2: [B, 192, 80, 80]
-> decoder 2D residual block: [B, 96, 80, 80]

-> transposed conv up: [B, 48, 160, 160]
-> concat with x1: [B, 96, 160, 160]
-> decoder 2D residual block: [B, 48, 160, 160]

-> 1x1 conv
-> prediction [B, 1, 160, 160]
```

当 z 轴边界不足 7 张时，代码使用边界切片重复补齐，训练时每个病例每个 epoch 采样 128 个中心切片，H/W crop 使用均衡采样，既覆盖脑组织区域，也保留边缘和背景位置。评估时从 z=0 到 z=D-1 逐 slice 推理，每张输出 [1, H, W]，最后按 z 轴拼回完整 3D prediction [D, H, W]。

### 训练与测评

3D T1 -> T2-FLAIR：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py --config configs/t1t2_smooth_seed42.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 eval.py --config configs/t1t2_smooth_seed42.yaml --checkpoint output/3dunet/smooth/t1tot2/weights/best_seed42.pt
```

2D / 2.5D T1 -> T2-FLAIR：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py --config configs/t1t2_2dunet_seed42.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 eval.py --config configs/t1t2_2dunet_seed42.yaml --checkpoint output/2dunet/t1tot2/weights/best_seed42.pt
```

3D T2-FLAIR -> T1：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py --config configs/t2t1_smooth_seed42.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 eval.py --config configs/t2t1_smooth_seed42.yaml --checkpoint output/3dunet/smooth/t2tot1/weights/best_seed42.pt
```

### 实验

测试集结果如下，指标均在 raw-space full-volume prediction 上计算：

| 实验 | Direction | MAE | MSE | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| 3D U-Net | T1 -> T2-FLAIR | **23.667** | **2363.125** | **29.732** | **0.8708** |
| 2D / 2.5D U-Net | T1 -> T2-FLAIR | 24.971 | 2475.824 | 29.505 | 0.8472 |
| 3D U-Net | T2-FLAIR -> T1 | 71.841 | 29119.437 | 28.886 | 0.8933 |

对于 T1 -> T2-FLAIR，3D U-Net 在 MAE、MSE、PSNR 和 SSIM 上均优于 2D/2.5D U-Net，说明完整三维上下文对跨模态体数据生成更有帮助。2D/2.5D 的优势是训练样本量更大、实现较轻，但它逐 slice 建模，z 轴一致性和全局结构恢复不如 3D 主线。T2-FLAIR -> T1 的 SSIM 达到 0.8933，结构相似性较好，MAE 为 71.841，明显大于 T1 -> T2-FLAIR 的 23.667，主要原因是 T1 图像的原始强度范围和整体强度尺度比 T2-FLAIR 不同，同样比例的误差会对应更大的绝对强度差。因此跨方向比较时应更多参考 PSNR 和 SSIM。综合结果看，最终以 3D U-Net 的预测作为结果。
