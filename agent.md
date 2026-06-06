# 任务要求

该任务涉及到两个数据集，分别是UKB_T1_100cases.tar.gz和ADNI_data_105cases.tar.gz，对于UKB，其压缩包内包含了100例影像数据和一个csv表格，影像数据存储在对应的CASEID的子文件夹下面，与csv表格以此来match，任务为预测影像个体的年龄和性别，最终的测试提交格式为一个csv文件，包括三列，分别为ID,Age,Sex；对于ADNI,其压缩包中包括105例影像数据和一个csv表格，影像数据同样存储在ID命名的子文件夹中，csv中包括了ID和对应的label，任务为预测影像个体的label，包括CN，MCI和AD，最终的测试提交格式为一个csv文件，包括两列，分别为ID,Pre

# 数据预处理

### UKB

预处理命令：不同模型使用不同输入。

```bash
conda activate cn

python data.py --dataset ukb --model brainiac
python data.py --dataset ukb --model sfcn
```

固定路径：

- UKB 原始压缩包：`dataset/UKB_T1_100cases.tar.gz`
- UKB 原始目录：`dataset/UKB_T1_100cases/`
- UKB 表格：`dataset/UKB_T1_100cases/selected_100_age_sex.csv`
- 表格字段：`eid, age, sex`
- BrainIAC 输出：`dataset/processed_brainiac/UKB/`
- SFCN 输出：`dataset/processed_sfcn/UKB/`

BrainIAC 预处理输出：

- `dataset/processed_brainiac/UKB/images/<ID>.nii.gz`
- `dataset/processed_brainiac/UKB/metadata.csv`

BrainIAC 流程：

1. 匹配病例目录名和 CSV 中的 `eid/age/sex`
2. 读取 NIfTI；如为 DICOM-like 文件夹，则先转 NIfTI
3. SimpleITK N4 bias field correction：校正 MRI 中低频强度不均匀，减少磁场和扫描位置导致的 intensity bias
4. 线性重采样到 `1 x 1 x 1 mm`
5. 如果 `BrainIAC/src/preprocessing/atlases/temp_head.nii.gz` 存在，刚性配准到该模板
6. 使用 BrainIAC 仓库内 `HD_BET.hd_bet` 做 skull stripping，默认 GPU 0
7. 中心 crop/pad 到 `96 x 96 x 96`
8. 非零体素 z-normalization，背景为 0

SFCN 预处理输出：

- `dataset/processed_sfcn/UKB/images/<ID>.nii.gz`
- `dataset/processed_sfcn/UKB/metadata.csv`
- `dataset/processed_sfcn/UKB/details.csv`

SFCN 流程：

1. 启动时强制检查 `flirt`、`bet`、`FSLDIR` 和 `$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz`
2. 若病例目录已有 `T1_brain_linearto_MNI.nii.gz`、`T1_brain_to_MNI.nii.gz` 或 `T1_unbiased_brain_linearto_MNI.nii.gz`，直接使用该文件进入官方 scale 和最终 crop/pad
3. 否则读取 NIfTI；如为 DICOM-like 文件夹，则先转 NIfTI。DICOM 是扫描仪原始序列格式，训练代码需要单个 3D NIfTI volume，因此先转换成统一文件
4. SimpleITK N4 bias field correction：校正结构像中缓慢变化的亮度偏差，让同一组织的强度更一致
5. 线性重采样到 `1 x 1 x 1 mm`：统一物理分辨率，使不同病例在空间尺度上可比
6. FSL BET brain extraction：`bet <in> <out> -R -f 0.5 -g 0`。该步骤去除头皮、头骨、眼眶和颅外组织，只保留脑组织输入
7. FSL FLIRT 12-dof 线性配准到 MNI152 1mm brain template，模板路径为 `$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz`
8. 按 `UKBiobank_deep_pretrain/examples.ipynb` 官方示例做 `data / data.mean()`，匹配官方 SFCN 权重的输入强度分布
9. 中心 crop/pad 到 `160 x 192 x 160`，不 resize。crop 去掉标准空间中多余边界，pad 补齐不足边界，保持 1mm 空间比例

SFCN 不复用 BrainIAC 的 `96 x 96 x 96` 图像，不使用 HD-BET，不写 skull stripping fallback，不跳过 MNI 配准。失败样本写入 `metadata.csv`：`image_path` 为空，`preprocessing_status` 为 `fail: <reason>`。


### train-val

所有当前训练默认使用 5-fold。BrainIAC 使用固定 seed 随机切分；SFCN 使用 `Sex + Age 等频分箱` 做 stratified 5-fold，使每个 fold 的性别比例和年龄分布尽量接近。每个 fold 约 20 例做 validation，其余约 80 例做 train。每个样本只在一个 fold 中作为 validation 出现一次。

DL 训练每个 fold 都会在自己的 train split 上训练，并用自己的 val split 选 checkpoint。BrainIAC DL 用最小 `val_loss` 选该 fold 的 best；SFCN Age 用最小 `val_age_mae` 选 best；SFCN Sex 用最大 `val_sex_balanced_acc` 选 best。因为 5-fold 本质上有 5 个不同 train/val 划分，所以不存在一个天然唯一的全局 best model；代码保存 `fold_0.pt` 到 `fold_4.pt`。

BrainIAC DL 的最终 `pred.csv` 使用 5 个 fold checkpoint 对全体 metadata 样本预测后集成：Age 取 5 个预测的平均值，Sex 对 logits 求平均后再 argmax。SFCN supervised 的 `pred.csv` 是每个样本所在 validation fold 的 out-of-fold 预测拼接。

BrainIAC sklearn frozen 先用原始 BrainIAC encoder 提取 frozen embedding；因为 backbone 没有用 UKB label 训练，这一步不产生 label 泄漏。每个 fold 内只用 train embedding fit `StandardScaler`、Ridge 和 LogisticRegression，再在 val embedding 上评估。

BrainIAC sklearn finetuned 每个 fold 独立运行：先只用该 fold 的 train split 微调 backbone，用该 fold 的 val split 选 `fold_i.pt`，再用这个 fold backbone 提取 train/val embedding。Ridge 和 LogisticRegression 只在 train embedding 上 fit，再在 val embedding 上评估，避免先用全数据 finetune 造成泄漏。

新输出规则：实验目录下不再创建 `fold_0/` 到 `fold_4/` 子文件夹。训练日志统一写 `train_log.csv`，预测统一写 `pred.csv`，总指标统一写 `metrics.json`。需要保存模型时，按 fold 平铺为 `fold_0.pt` 到 `fold_4.pt`；sklearn 模型平铺为 `age_ridge_fold_i.joblib`、`sex_logreg_fold_i.joblib` 和 `scaler_fold_i.joblib`。


# BrainIAC

### 代码

权重：

- BrainIAC 官方权重手动放到 `models/BrainIAC.ckpt`
- 下载入口见 BrainIAC README：https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=b9cnvwh8&dl=0

训练命令：

```bash
python train.py --config configs/brainiac_ukb_frozen.yaml
python train.py --config configs/brainiac_ukb_finetune.yaml
python train.py --config configs/brainiac_ukb_sklearn_frozen.yaml
python train.py --config configs/brainiac_ukb_sklearn_finetune.yaml
python train.py --config configs/brainiac_ukb_dl_age_finetune.yaml
python train.py --config configs/brainiac_ukb_dl_sex_finetune.yaml
```

训练模式：

- `mode: frozen`：冻结 backbone，只训练启用的 head。
- `mode: finetune`：训练 backbone 和启用的 head，backbone/head 使用不同 learning rate。
- `task: joint`：Age/Sex 联合训练。
- `task: age`：只训练 Age head。
- `task: sex`：只训练 Sex head。

### 模型

实现细节

- `src/models_brainiac.py` 动态加载 `BrainIAC/src/model.py` 中的 `ViTBackboneNet`
- 输入为单通道 `96 x 96 x 96` 3D volume
- DL 路线使用共享 BrainIAC backbone，加 Age head 和 Sex head
- Age loss：`MSELoss`，Sex loss：`CrossEntropyLoss`
- Age 训练默认按 fold 内 train split 标准化 target，预测和日志指标反标准化回真实年龄
- 所有实验使用 5-fold
- sklearn 头相比 fully-connection 头作为基线，Age：Ridge regression，Sex：LogisticRegression
- finetuned embedding 必须在每个 fold 内先只用 train split finetune backbone，再提取 train/val embedding，避免数据泄漏


# SFCN

### 代码

仓库和官方权重：

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
```

训练模式：

- `mode: pretrained_eval`：全部冻结，直接用官方权重推理
- `mode: frozen`：冻结 `feature_extractor`，只训练最后 classifier/head
- `mode: finetune`：全部不冻结，全模型微调
- `data_augment.enabled: true`：只对 train fold 做在线 3D 增强，val/test 不增强，不生成离线增强图像

### 模型

实现细节

- 架构：SFCN 是 3D fully convolutional network。官方 `SFCN` 由 6 个 3D convolution block 组成，前 5 个 block 使用 `Conv3d -> BatchNorm3d -> MaxPool3d -> ReLU`，最后一个 block 使用 `1 x 1 x 1 Conv3d -> BatchNorm3d -> ReLU`
- 输入：模型输入 shape 为 `[batch, 1, 160, 192, 160]`。预处理后的 NIfTI 已经是 `160 x 192 x 160`，训练 transform 只负责读取、补 channel 维度和保险性尺寸对齐
- 分类器：feature extractor 后接 `AvgPool3d([5, 6, 5])`、可选 `Dropout(0.5)` 和 `1 x 1 x 1 Conv3d`，最后对输出做 `log_softmax`
- Age head：官方 Age 权重输出 40 个 age bin 的 log probability。当前用 bin center `42.5 ... 81.5` 计算期望年龄，finetune 时用 soft age label 和 `KLDivLoss`
- Sex head：官方 Sex 权重输出 2 类 log probability。当前用 `argmax` 得到类别，finetune 时用 `NLLLoss`
- 权重加载：官方 checkpoint 是 `torch.nn.DataParallel` 保存格式，key 带 `module.` 前缀；wrapper 加载时去掉该前缀后严格加载
- 单任务设置：Age 和 Sex 使用不同官方权重，Sex 还使用不同 channel number `[28, 58, 128, 256, 256, 64]`，因此 SFCN 当前分开做 age/sex 单任务，不做 joint


# 实验结论

### BrainIAC

| 实验 | Age MAE | Age baseline MAE | Sex Acc | Sex baseline Acc | Sex balanced Acc |
|---|---:|---:|---:|---:|---:|
| `brainiac_ukb_frozen` | 5.604 | 6.315 | 0.640 | 0.480 | 0.626 |
| `brainiac_ukb_finetune` | 5.550 | 6.315 | 0.650 | 0.480 | 0.646 |
| `brainiac_ukb_sklearn_frozen` | 7.858 | 6.315 | 0.670 | 0.480 | 0.688 |
| `brainiac_ukb_sklearn_finetune` | 6.752 | 6.315 | 0.680 | 0.480 | 0.701 |
| `brainiac_ukb_dl_age_finetune` | 5.595 | 6.315 | - | - | - |
| `brainiac_ukb_dl_sex_finetune` | - | - | 0.650 | 0.480 | 0.632 |

总结：

1. BrainIAC DL head 路线的 Age 指标优于平均年龄 baseline，sklearn Ridge 读取 frozen embedding 的 Age 信息较弱
2. BrainIAC Sex 任务整体超过多数类 baseline，finetuned embedding + LogisticRegression 的 balanced accuracy 当前最好
3. Age 当前最好的是 `brainiac_ukb_finetune`，但与 `brainiac_ukb_dl_age_finetune` 很接近，100 例下不能过度解释小差异


### SFCN

| 实验 | Age MAE | Age baseline MAE | Sex Acc | Sex baseline Acc | Sex balanced Acc |
|---|---:|---:|---:|---:|---:|
| `sfcn_ukb_age_baseline` | 2.906 | 6.260 | - | - | - |
| `sfcn_ukb_age_finetune` | 2.412 | 6.260 | - | - | - |
| `sfcn_ukb_age_finetune_data_aug` | 2.200 | 6.260 | - | - | - |
| `sfcn_ukb_sex_baseline` | - | - | 0.770 | 0.520 | 0.779 |
| `sfcn_ukb_sex_finetune` | - | - | 0.990 | 0.520 | 0.989 |
| `sfcn_ukb_sex_finetune_data_aug` | - | - | 0.990 | 0.520 | 0.989 |

总结：

1. SFCN 官方 Age 权重直接推理已经明显优于平均年龄 baseline，全模型微调进一步降低 MAE
2. Age 当前最好的是 `sfcn_ukb_age_finetune_data_aug`，5-fold mean MAE 为 2.200；在线 3D 增强带来约 0.21 MAE 改善
3. Sex 当前最好且最简单的是 `sfcn_ukb_sex_finetune`，5-fold mean accuracy 为 0.990；数据增强没有提高 Sex 指标，因此最终 Sex 采用不增强模型
4. 最终 UKB Age/Sex 路线选用 SFCN：Age 使用 data augmentation finetune 的 5 个 fold，Sex 使用 no augmentation finetune 的 5 个 fold



# 测试

1. 如果测试集是压缩包，直接传给 `eval.py`，代码会解压到 `dataset/<dataset_name>/`
2. 如果测试集已经手动解压，直接传解压后的目录
3. `<dataset_name>` 自动由路径名推导，例如 `dataset/UKB_TEST.tar.gz` -> `UKB_TEST`
4. `eval.py` 调用 SFCN 预处理，输出到 `dataset/processed_sfcn/<dataset_name>/`
5. Age 使用 `outputs/sfcn_ukb_age_finetune_data_aug/fold_0.pt` 到 `fold_4.pt`，五个预测取平均
6. Sex 使用 `outputs/sfcn_ukb_sex_finetune/fold_0.pt` 到 `fold_4.pt`，五个模型的 class probability 平均后再 argmax
7. Age 和 Sex 合并成唯一提交文件 `outputs/<dataset_name>/pred.csv`

命令：

```bash
conda activate cn
# 测试集为 tar.gz，假设内部结构与 UKB_T1_100cases.tar.gz 接近
python eval.py --dataset dataset/UKB_TEST.tar.gz
# 测试集已解压，假设目录下是 CASEID 子文件夹和可选 csv
python eval.py --dataset dataset/UKB_TEST
```

- Age：`configs/sfcn_ukb_age_finetune_data_aug.yaml`
- Sex：`configs/sfcn_ukb_sex_finetune.yaml`