<p align="center">
  <h1 align="center">MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network</h1>
</p>

<p align="center">
  <em>基于形态感知的肺血管分割网络 — Pattern Recognition 2025</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pattern_Recognition-2025-blue" alt="Pattern Recognition 2025"/>
  <img src="https://img.shields.io/badge/Python-3.8+-green" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch 2.0+"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## 📌 概述

**MorVess** 是一种面向高分辨率胸部 CT 的自动化肺血管分割方法。该方法将**可微的几何先验**（血管厚度图和血管距离图）与大规模视觉基础模型（SAM）的 3D 自适应相结合，有效解决了肺血管分割中远端微血管断裂和拓扑完整性难以保持的关键问题。

### ✨ 核心特点

- 📐 **双重几何先验引导**：首次在血管分割中引入血管距离图（VDM）和血管厚度图（VTM），提供显式的边界和连通性监督。
- 🧊 **轻量级 2.5D 自适应**：通过参数高效的 2.5D Adapter 模块，在冻结的 SAM ViT 编码器中引入跨切片空间上下文。
- 🧩 **全局-局部融合模块 (GLFB)**：自适应整合多尺度语义特征与几何先验线索，实现高保真的拓扑重建。
- 🚀 **两阶段高效训练策略**：从宏观 3D 结构适配到微观拓扑精调，在降低显存消耗的同时稳健提升性能。

---

## 🏗️ 方法框架

### 整体架构

<p align="center">
  <img src="https://raw.githubusercontent.com/MaoFuyou/MorVess/main/Fig1.png" alt="MorVess Overall Framework" width="100%"/>
</p>
<p align="center"><b>图 1.</b> MorVess 整体框架。一个轻量级的 2.5D Adapter 增强了冻结的 SAM ViT 编码器的层间上下文，而多头解码器则联合预测语义掩码和可微几何先验。全局-局部融合模块（GLFB）整合了多尺度语义线索和几何场来细化血管拓扑。</p>

### 几何先验生成 (VDM & VTM)

<p align="center">
  <img src="https://raw.githubusercontent.com/MaoFuyou/MorVess/main/Fig2.png" alt="Geometric Priors Generation" width="100%"/>
</p>
<p align="center"><b>图 2.</b> 几何先验生成流程。左侧（VDM）：通过形态学腐蚀和指数距离衰减，将离散的二值掩码转化为连续可微的势场。右侧（VTM）：从内部距离场提取拓扑骨架，并将中心线半径传播到体素掩码中。</p>

---

## 📁 项目结构

```text
MorVess/
├── 📄 README.md                    # 本文档
├── 📄 MorVess_Development_Guide.md # 详细开发与使用技术文档
│
├── 📄 train_hq_parse_stage1.py     # 🏋️ Stage 1 训练脚本 (宏观适配)
├── 📄 train_hq_parse_stage2.py     # 🏋️ Stage 2 训练脚本 (拓扑精调)
├── 📄 test_parse_stage1.py         # 🔮 Stage 1 推理脚本
├── 📄 test_parse_stage2.py         # 🔮 Stage 2 推理脚本
│
├── 📄 generate_distance_map.py     # 📐 VDM 距离图生成核心算法
├── 📄 generate_thickness.py        # 📐 VTM 厚度图生成核心算法
├── 📄 generate_distance_process.py # 🔄 VDM 批量预处理
├── 📄 generate_thickness_process.py# 🔄 VTM 批量预处理
│
├── 📂 segment_anything/            # 🧠 MorVess 核心网络库 (基于 SAM)
│   ├── 📄 build_sam.py             # 模型构建与注册工厂
│   └── 📂 modeling/
│       ├── 📄 image_encoder_hq.py  # 包含 2.5D Adapter 的图像编码器
│       ├── 📄 mask_decoder_hq.py   # 多头几何预测解码器
│       ├── 📄 hq_refiner.py        # 全局-局部融合模块 (GLFB)
│       └── 📄 sam_distance_hq.py   # MorVess 完整模型组装
│
├── 📂 datasets/                    # 📦 数据集加载模块
│   └── 📄 dataset_distance.py      # 多任务数据读取与几何增强
│
└── 📂 preprocessing/               # 🧹 数据预处理模块
    └── 📄 util_script_parse2022_ok.py # Parse2022 标准预处理管线
```

---

## 🔧 环境安装

### 前置要求

- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Wu-beining/MorVess.git
cd MorVess

# 2. 创建并激活 Conda 环境
conda create -n morvess python=3.9 -y
conda activate morvess

# 3. 安装 PyTorch 及其他依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas SimpleITK nibabel opencv-python Pillow einops icecream tqdm h5py
```

### 预训练权重

下载官方 SAM ViT-Base 权重以初始化编码器：

```bash
mkdir pretrained_weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P pretrained_weights/
```

---

## 📦 数据集

本项目在两大高难度胸部 CT 数据集上进行了全面验证：

| 数据集 | 来源 | 特点 | 模态 / 分辨率 |
|--------|------|------|---------------|
| **Parse2022** | [Parse Challenge 2022](https://parse2022.grand-challenge.org/) | 包含 100 例高分辨率胸部 CT，标注了精细的多级正常肺动脉分支 | CT / 各向同性高分辨率 |
| **AIIB2023** | [AIIB 2023 Challenge](https://aiib23.grand-challenge.org/) | 肺纤维化患者 CT，包含支气管扩张、管壁增厚等严重的病理结构变形 | CT / 病理表现 |

> 详细的数据划分与预处理流程（包括 HU 裁剪、VDM/VTM 计算、2.5D Slice 构建等），请参考完整的 [MorVess 发行文档](./MorVess_Development_Guide.md)。

---

## 🚀 快速开始

### 1. 数据预处理构建几何先验

使用提供的脚本为您的二值 Mask 生成必需的几何先验（VDM / VTM），并打包为 2.5D 格式：

```bash
# 生成 VTM 厚度图和 5-slice pkl 缓存，并生成 data split CSV
python preprocessing/util_script_parse2022_ok.py
```

### 2. Stage I: 宏观特征与 2.5D 适配训练

此阶段冻结 SAM 主体，仅训练 2.5D Adapter 和多头解码器，使大模型快速适应 3D 医疗数据空间：

```bash
python train_hq_parse_stage1.py \
  --root_path ./data/parse2022/train/2D_all_5slice \
  --output ./checkpoints/res_hq-par-512-stage1 \
  --batch_size 1 \
  --img_size 512 \
  --base_lr 0.000010 \
  --max_epochs 400 \
  --vit_name vit_b \
  --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
  --rank 32
```

### 3. Stage II: 几何与拓扑精调

此阶段冻结 Adapter，专精于 GLFB 和 VDM/VTM 解码头的训练，强化远端细支气管的连通性：

```bash
python train_hq_parse_stage2.py \
  --root_path ./data/parse2022/train/2D_all_5slice \
  --output ./checkpoints/res_hq-par-256-stage2 \
  --batch_size 8 \
  --img_size 256 \
  --base_lr 0.000010 \
  --max_epochs 400 \
  --vit_name vit_b \
  --ckpt ./checkpoints/res_hq-par-512-stage1/epoch_16.pth \
  --rank 32
```

### 4. 推理与测试

运行预测脚本生成完整的 3D NIfTI 预测结果：

```bash
python test_parse_stage2.py \
  --task parse \
  --root_path ./data/parse2022/train/2D_all_5slice \
  --output_dir ./out_predict \
  --img_size 256 \
  --vit_name vit_b \
  --rank 32 \
  --is_savenii
```

---

## 📊 实验结果

### 核心分割性能对比

在两大数据集上与最先进的基线方法进行对比：

| 数据集 | 方法 | Dice ↑ | clDice ↑ | HD95(mm) ↓ | AMR ↓ |
|--------|------|--------|----------|------------|-------|
| **Parse2022** | nn-UNET-V2 | 77.28±5.83 | 75.31±5.83 | 9.53±3.86 | 0.22±0.22 |
| | SegMamba | 79.24±5.19 | 73.18±4.69 | 9.91±3.79 | 0.25±0.14 |
| | COMMA | 83.27±4.29 | 80.10±3.74 | 5.11±3.42 | 0.14±0.17 |
| | **MorVess (Ours)** | **86.84±4.18** | **83.22±3.17** | **4.53±3.06** | **0.12±0.09** |
| **AIIB2023** | nn-UNET-V2 | 92.83±6.55 | 84.31±5.09 | 5.92±6.01 | 0.10±0.14 |
| | SegMamba | 91.29±7.24 | 85.51±5.29 | 4.59±6.11 | 0.12±0.17 |
| | COMMA | 92.88±5.25 | 86.23±3.94 | 4.25±4.94 | 0.09±0.02 |
| | **MorVess (Ours)** | **94.31±3.52** | **89.34±3.46** | **3.24±4.81** | **0.07±0.04** |

### 几何一致性分析（血管树解剖学评估）

使用 VMTK 对拓扑一致性进行评估，数值越低代表误差越小：

| 数据集 | 方法 | TVV Diff ↓ | Pearson Diameter ↑ | KL Divergence ↓ | BV Diff ↓ |
|--------|------|------------|--------------------|-----------------|-----------|
| **Parse2022** | COMMA | 16.73 | 0.93 | 0.08 | 0.86 |
| | **MorVess (Ours)** | **13.16** | **0.96** | **0.03** | **0.29** |

> **关键发现**: MorVess 不仅在体素级重叠度（Dice）上领先，更由于几何先验的存在，在**拓扑连通性 (clDice)**、**总血管体积 (TVV)** 和**末端微血管保留度 (BV)** 等反映临床真实解剖结构的指标上展现出压倒性优势。

---

## 🛠️ 计算效率

采用参数高效微调（PEFT）和两阶段设计的 MorVess 在硬件极其友好的范围内实现了 SOTA 性能：

| Method | Trainable Params | Total Params | GMACs / stack | Peak VRAM |
|--------|------------------|--------------|---------------|-----------|
| nnU-Net | 32.0 M | 32.0 M | 180 | 18 GB |
| Diff-UNet | 64.0 M | 64.0 M | 340 | 32 GB |
| **MorVess** | **1.0 M** | 93.6 M | **42** | **4.2 GB** |

*(测试条件：Parse2022，输入尺寸为 512×512×5 时的理论计算量与实际显存峰值)*

---

## 📜 引用

如果您在研究中使用了 MorVess，请引用我们的文章：

```bibtex
@article{mao2025morvess,
  title={MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network},
  author={Mao, Fuyou and Chen, Yifei and Wu, Beining and Lin, Lixin and
          Dai, Jinnan and Li, Zhiling and Chen, Yilei and Wang, Yaqi and
          Zhang, Hao and Tang, Yan and Zhou, Huiyu and Qin, Feiwei},
  journal={Pattern Recognition},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
