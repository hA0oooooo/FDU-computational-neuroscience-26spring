# 任务要求

当前工程包含两个数据集。UKB 数据集来自 `UKB_T1_100cases.tar.gz`，共 100 例 T1 影像和一个 age/sex CSV，病例目录名与 CSV 中 ID 匹配，任务是预测年龄和性别，提交格式为 `ID,Age,Sex`。ADNI 数据集来自 `ADNI_data_105cases.tar.gz`，共 105 例 T1 影像和一个 label CSV，任务是 CN/MCI/AD 三分类，提交格式为 `ID,Pre`。最终的主线实验是：UKB Age 使用 `sfcn_ukb_age_finetune_data_aug`，UKB Sex 使用 `sfcn_ukb_sex_finetune`，ADNI 使用 `rootstrap_adni_finetune_data_aug`。BrainIAC、3D-CNN、BrainMVP、SFCN-ADNI 作为已尝试路线保留代码和结果，但不是当前最终选择。

# 数据预处理

### 处理命令

原始数据保留在 `dataset/` 下，UKB 解压后目录为 `dataset/UKB_T1_100cases/`，ADNI 解压后目录为 `dataset/ADNI_data/`。

```bash
cd dataset
tar -tzf UKB_T1_100cases.tar.gz | head
tar -xzf UKB_T1_100cases.tar.gz

tar -tzf ADNI_data_105cases.tar.gz | head
tar -xzf ADNI_data_105cases.tar.gz
```

UKB 预处理命令：

```bash
conda activate cn
python data.py --dataset ukb --model brainiac
python data.py --dataset ukb --model sfcn
```

ADNI 预处理命令：

```bash
conda activate cn
python data.py --dataset adni --model brainiac
python data.py --dataset adni --model sfcn
python data.py --dataset adni --model 3dcnn
python data.py --dataset adni --model brainmvp
python data.py --dataset adni --model rootstrap
```

### 处理细节

BrainIAC 流程：

1. 匹配病例目录名和 CSV 中的 ID/label 字段，UKB 写出 `ID,image_path,Age,Sex`，ADNI 写出 `ID,image_path,label`
2. 读取 NIfTI；如为 DICOM-like 文件夹，则先转 NIfTI
3. SimpleITK N4 bias field correction：校正 MRI 中低频强度不均匀，减少磁场和扫描位置导致的 intensity bias
4. 线性重采样到 `1 x 1 x 1 mm`
5. 如果 `BrainIAC/src/preprocessing/atlases/temp_head.nii.gz` 存在，刚性配准到该模板
6. 使用 BrainIAC 仓库内 `HD_BET.hd_bet` 做 skull stripping，默认 GPU 0
7. 中心 crop/pad 到 `96 x 96 x 96`
8. 非零体素 z-normalization，背景为 0

SFCN 流程：

1. 启动时强制检查 `flirt`、`bet`、`FSLDIR` 和 `$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz`
2. 若病例目录已有 `T1_brain_linearto_MNI.nii.gz`、`T1_brain_to_MNI.nii.gz` 或 `T1_unbiased_brain_linearto_MNI.nii.gz`，直接使用该文件进入官方 scale 和最终 crop/pad
3. 否则读取 NIfTI；如为 DICOM-like 文件夹，则先转 NIfTI
4. SimpleITK N4 bias field correction
5. 线性重采样到 `1 x 1 x 1 mm`
6. FSL BET brain extraction：`bet <in> <out> -R -f 0.5 -g 0`
7. FSL FLIRT 12-dof 线性配准到 MNI152 1mm brain template
8. 按 `UKBiobank_deep_pretrain/examples.ipynb` 官方示例做 `data / data.mean()`，匹配官方 SFCN 权重的输入强度分布
9. 中心 crop/pad 到 `160 x 192 x 160`，不 resize

3D-CNN 流程：

1. 只用于 ADNI，输出到 `dataset/processed_3dcnn/ADNI/`
2. 读取原始 ADNI NIfTI/DICOM，不复用 BrainIAC/SFCN 成品
3. 按官方 3D-CNN 逻辑处理为 `96 x 96 x 73`
4. 使用 non-zero voxel mean/std normalization
5. 进入 PyTorch 时转为 `1 x 73 x 96 x 96`

BrainMVP 流程：

1. 只用于 ADNI，输出到 `dataset/processed_brainmvp/ADNI/`
2. 读取原始 ADNI T1，按单模态 T1 处理
3. reorient 到 RAS，重采样到 1mm spacing
4. foreground crop 后按 5th-95th percentile clip 并 rescale 到 `[0,1]`
5. resize 到 `128 x 128 x 64`
6. deterministic center crop 到 `96 x 96 x 64`

Rootstrap 流程：

1. 只用于 ADNI，输出到 `dataset/processed_rootstrap/ADNI/`
2. 读取原始 ADNI NIfTI/DICOM，不复用其它模型成品
3. FSL `fslreorient2std` 对齐方向
4. FSL FLIRT affine registration 到 MNI152 T1 1mm template
5. FSL BET skull stripping：`bet <in> <out> -R -f 0.4 -g 0`
6. 在 skull-stripped image 上做 SimpleITK N4 bias correction
7. 训练/推理 transform 使用 MONAI `ScaleIntensity`、channel-first、`Resize((96,96,96))`

train-val 划分：

训练统一使用 5-fold cross validation。UKB 的 Age/Sex 按 seed 固定划分，ADNI 是 CN/MCI/AD 三分类，使用 StratifiedKFold 保持每个 fold 中三类比例一致。每个 fold 只用 train split 训练，val split 不做数据增强。产生模型权重的实验每个 fold 保存一个验证集最优权重，从 `fold_0.pt` 到 `fold_4.pt`。若多个 epoch 的主指标相同，选择更早出现的 epoch，避免偏向后期过拟合权重。最终 `metrics.json` 记录每个 fold 和 mean 结果，`pred.csv` 保存 out-of-fold 预测或直接评估预测。

# 模型

### BrainIAC

BrainIAC 是面向 3D brain MRI 的 foundation encoder。其核心思想是使用大规模脑 MRI 数据预训练一个通用表征模型，使模型能够从结构 MRI 中学习脑区形态、组织结构和全脑空间模式。架构上，BrainIAC 使用 ViT-style backbone，将 3D MRI volume 划分为 patch/token 后送入 Transformer encoder，通过 self-attention 建模不同脑区之间的长程依赖关系。与传统 3D CNN 相比，ViT encoder 更强调全局空间关系和跨区域结构模式，因此适合提取全脑级 embedding。

在下游任务中，BrainIAC encoder 通常作为共享 backbone 使用。对于回归任务，可以在 encoder 输出后接 regression head；对于分类任务，可以接 classification head；也可以冻结 encoder，仅提取 embedding，再交给传统机器学习分类器。模型的核心不是某一个固定任务头，而是一个可迁移的 brain MRI representation encoder。

权重：https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=b9cnvwh8&dl=0

训练命令：

```bash
python train.py --config configs/brainiac_ukb_frozen.yaml
python train.py --config configs/brainiac_ukb_finetune.yaml
python train.py --config configs/brainiac_ukb_sklearn_frozen.yaml
python train.py --config configs/brainiac_ukb_sklearn_finetune.yaml
python train.py --config configs/brainiac_ukb_dl_age_finetune.yaml
python train.py --config configs/brainiac_ukb_dl_sex_finetune.yaml
python train.py --config configs/brainiac_adni_sklearn_frozen.yaml
python train.py --config configs/brainiac_adni_dl_finetune.yaml
```

训练模式：

- `mode: frozen`：冻结 backbone，只训练启用的 head
- `mode: finetune`：训练 backbone 和启用的 head，backbone/head 使用不同 learning rate
- `task: joint`：Age/Sex 联合训练
- `task: age`：只训练 Age head
- `task: sex`：只训练 Sex head
- `task: label`：ADNI CN/MCI/AD 三分类

### SFCN

SFCN 是用于 UK Biobank 脑 MRI 年龄预测和性别预测的 3D convolutional network。SFCN 全称通常理解为 Simple Fully Convolutional Network，其特点是尽量避免大型全连接层，而是主要通过 3D convolution、normalization、activation 和 pooling 逐层压缩空间特征，最后得到全脑结构表征并输出预测结果。由于它直接处理 3D T1 MRI，因此能够利用脑结构的空间信息，而不是把 MRI 切成 2D slice 后分别处理。

SFCN 的结构偏向高效 3D CNN：前面多层 3D convolutional blocks 提取局部到全局的空间特征，后面通过全局 pooling 或接近全卷积的输出头完成预测。它在 UK Biobank 上预训练，因此对年龄相关脑萎缩、脑室扩大、皮层和白质结构变化等模式较敏感。其优势在于脑年龄和性别这类与全脑结构强相关的任务；但由于原始权重不是针对神经退行性疾病分类训练的，将其迁移到疾病分型任务时，通常需要替换或微调最后的任务头。

权重：

- 仓库：`UKBiobank_deep_pretrain/`
- Age 权重：`UKBiobank_deep_pretrain/brain_age/run_20190719_00_epoch_best_mae.p`
- Sex 权重：`UKBiobank_deep_pretrain/sex_prediction/run_20191008_00_epoch_last.p`

训练命令：

```bash
python train.py --config configs/sfcn_ukb_age_finetune.yaml
python train.py --config configs/sfcn_ukb_sex_finetune.yaml
python train.py --config configs/sfcn_ukb_age_finetune_data_aug.yaml
python train.py --config configs/sfcn_ukb_sex_finetune_data_aug.yaml
```

```bash
python train.py --config configs/sfcn_ukb_age_baseline.yaml
python train.py --config configs/sfcn_ukb_age_frozen.yaml
python train.py --config configs/sfcn_ukb_sex_baseline.yaml
python train.py --config configs/sfcn_ukb_sex_frozen.yaml
python train.py --config configs/sfcn_adni_sklearn_frozen.yaml
python train.py --config configs/sfcn_adni_dl_finetune.yaml
python train.py --config configs/sfcn_adni_dl_finetune_data_aug.yaml
```

训练模式：

- `mode: pretrained_eval`：全部冻结，直接用官方权重推理
- `mode: frozen`：冻结 `feature_extractor`，只训练最后 classifier/head
- `mode: finetune`：全部不冻结，全模型微调
- `data_augment.enabled: true`：只对 train fold 做在线 3D 增强，val/test 不增强，不生成离线增强图像


### 3D-CNN

该 3D-CNN 来自 ADNI 上的 Alzheimer’s disease classification 预训练模型，原始任务是 CN 与 AD 的二分类。架构上，它是一个直接读取 3D MRI volume 的卷积神经网络，通过多层 3D convolution 和 pooling 从 T1 MRI 中提取空间特征，最后接二分类 head 输出 CN/AD 判断。与 2D slice 模型不同，3D-CNN 可以在三维空间中同时建模脑区形态、脑室扩张、海马萎缩等结构线索。

该模型的原始权重包含完整二分类头，因此它不是一个纯粹的 backbone-only checkpoint。迁移到新的分类任务时，通常有两种方式：一种是去掉原始二分类 head，把中间层或倒数第二层作为 embedding extractor；另一种是保留 backbone，替换最后分类层，再进行微调。由于原始任务是 CN/AD 二分类，模型学到的特征更偏向区分典型正常对照和典型 Alzheimer’s disease，对于 MCI 这种中间状态未必天然适配。 

权重：`3D_CNN_pretrained_model/AD_pretrained_weights.pt`

训练命令：

```bash
python train.py --config configs/3dcnn_adni_sklearn_frozen.yaml
python train.py --config configs/3dcnn_adni_dl_finetune.yaml
```

### BrainMVP

BrainMVP 是面向脑 MRI 的多模态自监督预训练框架。它不是一个单纯的分类网络，而是通过多模态 MRI 预训练学习可迁移的医学影像表征。模型主要包含一个视觉 encoder 和重建/对比学习相关模块，推荐版本使用 UniFormer backbone，另有 UNet 版本。UniFormer 结合 convolution 与 Transformer-style token mixing，既能捕获局部影像结构，也能建模更长程的空间依赖。

BrainMVP 的预训练任务围绕多模态 MRI 设计，包括 cross-modal reconstruction、modality-wise data distillation 和 modality-aware contrastive learning。简单说，模型会利用一个 modality 的图像去辅助重建或约束另一个 modality 的表示，使 encoder 学到不同 MRI modality 之间共享的脑结构信息。下游使用时，通常取预训练 encoder 作为特征提取器，再接具体任务头进行分类或分割。对于单模态 T1 MRI，实际使用的是其 encoder 的迁移能力，而不是完整的多模态重建框架。

权重：`models/brainmvp_uniformer.pth`

训练命令：

```bash
python train.py --config configs/brainmvp_adni_sklearn_frozen.yaml
python train.py --config configs/brainmvp_adni_dl_finetune.yaml
```

### Rootstrap

Rootstrap Alzheimer Classifier 是一个基于 MONAI 3D DenseNet121 的 ADNI MRI 三分类模型。模型输入是单通道 3D NIfTI MRI，经过强度缩放、通道维度整理和 resize 后形成 [B, 1, 96, 96, 96] 的 3D tensor。主干网络为 DenseNet121(spatial_dims=3, in_channels=1, out_channels=3)，即 3D 版本 DenseNet121，最后直接输出三个类别的 logits。

DenseNet 的核心特点是 dense connection：每一层不仅接收前一层输出，也接收前面多个层的特征，从而增强特征复用和梯度传播。3D DenseNet121 将这一结构扩展到三维医学影像中，通过 3D convolutional dense blocks 提取不同层级的脑结构特征。该模型提供的是完整三分类模型权重，而不是单独的 embedding 权重；其输出类别顺序为 Alzheimer’s、Mild Cognitive Impairment、Control，对应 AD、MCI、CN。若需要使用 embedding，可以在最后分类层之前截断模型，自行提取 penultimate feature。

权重：`https://huggingface.co/rootstrap-org/Alzheimer-Classifier-Demo`

训练命令：

```bash
python train.py --config configs/rootstrap_adni_baseline.yaml
python train.py --config configs/rootstrap_adni_finetune.yaml
python train.py --config configs/rootstrap_adni_finetune_data_aug.yaml
```


# 实验

### Age & Sex

| 实验 | Age MAE | Sex Acc | Sex Balanced Acc |
|---|---:|---:|---:|
| `naive` | 6.260 | 0.520 | 0.500 |
| `brainiac_ukb_frozen` | 5.604 | 0.640 | 0.626 |
| `brainiac_ukb_finetune` | 5.550 | 0.650 | 0.646 |
| `brainiac_ukb_sklearn_frozen` | 7.858 | 0.670 | 0.688 |
| `brainiac_ukb_sklearn_finetune` | 6.752 | 0.680 | 0.701 |
| `brainiac_ukb_dl_age_finetune` | 5.595 | - | - |
| `brainiac_ukb_dl_sex_finetune` | - | 0.650 | 0.632 |
| `sfcn_ukb_age_baseline` | 2.906 | - | - |
| `sfcn_ukb_age_finetune` | 2.412 | - | - |
| `sfcn_ukb_age_finetune_data_aug` | **2.200** | - | - |
| `sfcn_ukb_age_finetune_data_aug_3` | 2.217 | - | - |
| `sfcn_ukb_sex_baseline` | - | 0.770 | 0.779 |
| `sfcn_ukb_sex_finetune` | - | **0.990** | **0.989** |
| `sfcn_ukb_sex_finetune_data_aug` | - | **0.990** | **0.989** |


### ADNI

| 实验 | Acc | Balanced Acc | Macro F1 |
|---|---:|---:|---:|
| `naive` | 0.333 | 0.333 | 0.167 |
| `rootstrap_adni_baseline` | 0.533 | 0.533 | 0.522 |
| `rootstrap_adni_finetune` | 0.695 | 0.695 | 0.688 |
| `rootstrap_adni_finetune_data_aug` | **0.762** | **0.762** | **0.756** |
| `brainiac_adni_sklearn_frozen` | 0.457 | 0.457 | 0.444 |
| `brainiac_adni_dl_finetune` | 0.610 | 0.610 | 0.605 |
| `sfcn_adni_sklearn_frozen` | 0.448 | 0.448 | 0.439 |
| `sfcn_adni_dl_finetune` | 0.448 | 0.448 | 0.396 |
| `sfcn_adni_dl_finetune_data_aug` | 0.448 | 0.448 | 0.408 |
| `3dcnn_adni_sklearn_frozen` | 0.457 | 0.457 | 0.459 |
| `3dcnn_adni_dl_finetune` | 0.514 | 0.514 | 0.500 |
| `brainmvp_adni_sklearn_frozen` | 0.400 | 0.400 | 0.395 |
| `brainmvp_adni_dl_finetune` | 0.581 | 0.581 | 0.539 |


# 评测

### Age & Sex

1. 如果测试集是压缩包，直接传给 `eval1.py`，代码会解压到 `dataset/<dataset_name>/`
2. 如果测试集已经手动解压，直接传解压后的目录
3. `<dataset_name>` 自动由路径名推导，例如 `dataset/UKB_TEST.tar.gz` -> `UKB_TEST`
4. `eval1.py` 调用 SFCN 预处理，输出到 `dataset/processed_sfcn/<dataset_name>/`
5. Age 使用 `outputs/sfcn_ukb_age_finetune_data_aug_seed3/seed_<seed>_fold_<fold>.pt`，3 个 seed × 5 个 fold 的年龄分布平均后求预测年龄
6. Sex 使用 `outputs/sfcn_ukb_sex_finetune_seed3/seed_<seed>_fold_<fold>.pt`，3 个 seed × 5 个 fold 的 class probability 平均后 argmax
7. Age 和 Sex 合并成唯一提交文件 `outputs/<dataset_name>/pred.csv`

命令：

```bash
conda activate cn
# 测试集为 tar.gz，假设内部结构与 UKB_T1_100cases.tar.gz 接近
python eval1.py --dataset dataset/UKB_TEST.tar.gz
# 测试集已解压，假设目录下是 CASEID 子文件夹和可选 csv
python eval1.py --dataset dataset/UKB_TEST
```

- Age：`configs/sfcn_ukb_age_finetune_data_aug_seed3.yaml`
- Sex：`configs/sfcn_ukb_sex_finetune_seed3.yaml`


### ADNI

1. 如果测试集是压缩包，直接传给 `eval2.py`，代码会解压到 `dataset/<dataset_name>/`
2. 如果测试集已经手动解压，直接传解压后的目录
3. `eval2.py` 调用 Rootstrap 预处理，输出到 `dataset/processed_rootstrap/<dataset_name>/`
4. 当前使用 `rootstrap_adni_finetune_data_aug_seed3`，3 个 seed × 5 个 fold 的 logits 平均后 argmax
5. 输出唯一提交文件 `outputs/<dataset_name>/pred.csv`，字段为 `ID,Pre`，`Pre` 为 `CN/MCI/AD`

命令：

```bash
conda activate cn
# 测试集为 tar.gz，假设内部结构与 ADNI_data_105cases.tar.gz 接近
python eval2.py --dataset dataset/ADNI_TEST.tar.gz
# 测试集已解压，假设目录下是 CASEID 子文件夹和可选 csv
python eval2.py --dataset dataset/ADNI_TEST
```

- ADNI：`configs/rootstrap_adni_finetune_data_aug_seed3.yaml`
