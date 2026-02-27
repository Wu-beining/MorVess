# MorVess 开发文档

> **MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network**
>
> 形态感知肺血管分割网络 — 完整开发与使用指南

---

## 1. 项目概述

### 1.1 研究背景

肺血管分割是医学影像分析中的基础任务，在肺部疾病诊断、灌注评估和术前规划中具有关键作用。肺血管呈现高度复杂的稀疏树状分布结构，远端微血管纤细且弯曲，传统方法难以同时维持分割精度和血管连通性。

现有深度学习方法将血管分割视为逐体素分类任务，缺乏对全局拓扑结构的显式建模。交叉熵和 Dice 损失仅优化局部像素级对应关系，无法约束血管连通性和层级分支关系。

### 1.2 核心创新

MorVess 提出了一种**形态感知**的分割框架，核心创新包括：

1. **几何先验监督**：联合优化血管厚度图（VTM）和血管距离图（VDM），为模型提供显式的拓扑和几何引导
2. **轻量级 2.5D Adapter**：在冻结的 SAM ViT 编码器中引入 3D 卷积结构，桥接 2D 平面特征与 3D 拓扑结构
3. **全局-局部融合模块（GLFB）**：自适应融合多尺度上下文特征和几何先验，增强细粒度血管特征感知
4. **两阶段训练策略**：从宏观结构适配到微观拓扑精调的渐进式优化

### 1.3 性能表现

| 数据集 | Dice | clDice | HD95(mm) |
|--------|------|--------|----------|
| Parse2022 | 86.84±4.18 | 83.22±3.17 | 4.53±3.06 |
| AIIB2023 | 94.31±3.52 | 89.34±3.46 | 3.24±4.81 |

相比第二名方法 COMMA，MorVess 在 Parse2022 上 Dice 提升 +3.57%，clDice 提升 +3.12%。

### 1.4 计算效率

| 指标 | MorVess | nnU-Net | Diff-UNet |
|------|---------|---------|-----------|
| 可训练参数 | **1.0M** | 32M | 64M |
| GMACs/slice stack | **42** | 180 | 340 |
| 显存占用 (batch=4) | **4.2GB** | 18GB | 32GB |

---

## 2. 项目结构

```
MorVess/
├── train_hq_parse_stage1.py          # Stage I 训练入口
├── train_hq_parse_stage2.py          # Stage II 训练入口
├── test_parse_stage1.py              # Stage I 测试/推理脚本
├── test_parse_stage2.py              # Stage II 测试/推理脚本
├── generate_distance_map.py          # VDM 距离图生成（单文件模式）
├── generate_distance_process.py      # VDM 距离图批量预处理
├── generate_batch_distance_map.py    # VDM 批量生成工具
├── generate_thickness.py             # VTM 厚度图生成
├── generate_thickness_process.py     # VTM 厚度图批量预处理
│
├── datasets/                         # 数据集加载模块
│   ├── __init__.py
│   ├── dataset.py                    # 基础数据集加载器
│   ├── dataset_distance.py           # 多任务数据集加载器（含 VDM/VTM）
│   ├── dataset_bbox.py               # BBox 数据集加载器
│   ├── dataset_v1.py                 # v1 版本数据集
│   ├── dispersion_analysis.py        # 特征空间分布分析
│   ├── dispersion_enhanced.py        # 增强分布分析
│   ├── dispersion_postviz.py         # 分布后处理可视化
│   └── scatter_with_ellipses.py      # 椭圆散点图可视化
│
├── preprocessing/                    # 数据预处理脚本
│   ├── dataset_split.md              # 数据集划分说明
│   ├── split_pancreas.pkl            # 胰腺数据集划分文件
│   ├── util_script_parse2022_ok.py   # Parse2022 预处理（推荐）
│   ├── util_sript_parse2022.py       # Parse2022 预处理（旧版）
│   ├── util_sript_parse2022_distance.py  # Parse2022 距离图预处理
│   ├── util_sript_aiib23.py          # AIIB2023 预处理
│   ├── util_script_btcv.py           # BTCV 预处理
│   ├── util_script_endovis18.py      # EndoVis18 预处理
│   └── util_script_prostateMRI.py    # Prostate MRI 预处理
│
└── segment_anything/                 # MorVess 核心模型库（基于 SAM 修改）
    ├── __init__.py                   # 包入口，导出模型注册表
    ├── build_sam.py                  # 模型工厂 & 注册表
    ├── predictor.py                  # SAM 预测器
    ├── automatic_mask_generator.py   # 自动掩码生成器
    │
    ├── modeling/                     # 模型核心组件
    │   ├── __init__.py               # 模型导出
    │   ├── image_encoder.py          # 原始 SAM ViT 图像编码器
    │   ├── image_encoder_hq.py       # ★ HQ 图像编码器（含 2.5D Adapter）
    │   ├── mask_decoder.py           # 原始 SAM 掩码解码器
    │   ├── mask_decoder_distance.py  # 距离图解码器
    │   ├── mask_decoder_hq.py        # ★ 多头几何预测解码器（含 GLFB）
    │   ├── mask_decoder_bbox.py      # BBox 解码器
    │   ├── hq_refiner.py             # ★ 全局-局部融合模块（GLFB）
    │   ├── prompt_encoder.py         # 提示编码器
    │   ├── transformer.py            # 双向 Transformer
    │   ├── sam.py                    # 原始 SAM 模型
    │   ├── sam_bbox.py               # BBox SAM 模型
    │   ├── sam_distance.py           # 距离图 SAM 模型
    │   ├── sam_distance_hq.py        # ★ MorVess 完整模型
    │   └── common.py                 # 公共组件（LayerNorm2d, MLPBlock）
    │
    └── utils/                        # 工具函数
        ├── __init__.py
        ├── amg.py                    # 自动掩码生成工具
        ├── onnx.py                   # ONNX 导出工具
        └── transforms.py            # 图像变换工具
```

> **★** 标注的文件为 MorVess 的核心实现文件。

---

## 3. 方法详解

### 3.1 整体架构

MorVess 由三个核心模块组成：

```
┌─────────────────────────────────────────────────────────────┐
│                     MorVess Framework                       │
│                                                             │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ 几何先验生成  │  │ 2.5D 几何适配网络 │  │ 两阶段训练策略  │  │
│  │  VDM + VTM   │  │ Encoder+Decoder │  │ Stage I + II   │  │
│  └──────────────┘  └────────────────┘  └────────────────┘  │
│                                                             │
│  输入: 3D CT → 2.5D 切片(5-slice)                           │
│  输出: 血管掩码 + 距离图(VDM) + 厚度图(VTM)                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 几何先验生成

#### 3.2.1 血管距离图（Vessel Distance Map, VDM）

**目的**：将离散的二值掩码边界转化为连续可微的势场，使模型能够感知血管边界梯度。

**算法流程**：

1. 对二值掩码 Ω 进行形态学腐蚀得到 Ω⊖ε
2. 提取血管边界层：∂Ω = Ω \ (Ω⊖ε)
3. 计算每个体素到边界层的加权欧几里得距离：D(x) = min‖(x−y)⊙Sp‖₂
4. 指数衰减转换为势场：VDM(x) = exp(−λ · D(x))，其中 λ 控制衰减速率

**实现文件**：[generate_distance_map.py](file:///D:/desktop/Morvess/generate_distance_map.py)

```python
# 核心实现
internal_dist_array = distance_transform_edt(mask_array, sampling=spacing)
boundary_dist_array = distance_transform_edt(boundary_array == 0, sampling=spacing)
potential_map_array = np.exp(-lambda_param * boundary_dist_array)
```

**关键参数**：
- `lambda_param`：衰减系数，默认 0.05。值越小势场范围越广，值越大越集中在边界
- `spacing`：体素间距向量，从 NIfTI 元数据获取，确保计算的是物理距离（mm）

#### 3.2.2 血管厚度图（Vessel Thickness Map, VTM）

**目的**：为每个血管体素分配与其拓扑位置一致的厚度值，提供全局尺度的一致性约束。

**算法流程**：

1. 计算内部距离场：D_internal(x) = min‖(x−y)⊙Sp‖₂，∀y ∈ 背景
2. 拓扑保持细化算法提取血管中心线骨架 S = τ(Ω)
3. 定义骨架点半径：r(s) = D_internal(s)
4. 最近骨架投影：πs(x) = argmin_{s∈S} d(x,s)
5. 厚度值：VTM(x) = 2 · r(πs(x))

**实现文件**：[generate_thickness.py](file:///D:/desktop/Morvess/generate_thickness.py)

```python
# 核心实现
internal_dist = distance_transform_edt(mask_arr, sampling=spacing[::-1])
skel_img = thinner.Execute(sitk.Cast(mask_img, sitk.sitkUInt8))  # 中轴提取
r_skel = internal_dist * skel_arr  # 骨架半径
_, inds = distance_transform_edt(arr, sampling=spacing[::-1], return_indices=True)
nearest_r = r_skel[z_idx, y_idx, x_idx]  # 最近骨架半径
thickness = 2.0 * nearest_r  # 直径 = 2×半径
```

**关键参数**：
- `smooth_sigma_mm`：可选的高斯平滑 sigma（mm），默认 0.05

### 3.3 模型架构

#### 3.3.1 图像编码器：ImageEncoderViT_hq

**文件**：[image_encoder_hq.py](file:///D:/desktop/Morvess/segment_anything/modeling/image_encoder_hq.py)

基于 SAM 的 ViT 编码器，新增以下关键组件：

**2.5D Adapter**（集成在 `Block` 中）：
```python
class Block(nn.Module):
    def __init__(self, ...):
        # 2.5D Adapter 组件
        self.adapter_channels = 384
        self.adapter_linear_down = nn.Linear(dim, 384, bias=False)    # 降维
        self.adapter_linear_up = nn.Linear(384, dim, bias=False)      # 升维
        self.adapter_conv = nn.Conv3d(384, 384, kernel_size=(3,1,1),  # 3D 卷积
                                       padding='same')
        self.adapter_act = nn.GELU()
        self.adapter_norm = norm_layer(dim)
        # 双路 Adapter（Attention 后 + MLP 后各一个）
        self.adapter_linear_down_2 = nn.Linear(dim, 384, bias=False)
        self.adapter_linear_up_2 = nn.Linear(384, dim, bias=False)
        self.adapter_conv_2 = nn.Conv3d(384, 384, kernel_size=(3,1,1), padding='same')
```

**多层特征输出**（`forward_features`）：
```python
def forward_features(self, x):
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i == 0:
            feat_early = x         # 浅层特征（局部纹理+边缘细节）
    feat_last = x                  # 深层特征（高级语义抽象）
    F_early = self.ln_early(self.proj_early(feat_early.permute(0,3,1,2)))
    F_last = self.neck(feat_last.permute(0,3,1,2))
    return F_last, F_early, F_last  # 三路输出
```

**设计意图**：
- `F_early`（第一个 Block 输出）：保留局部纹理和边缘细节，用于细粒度血管重建
- `F_last`（最后一层输出经 neck 处理）：提供全局语义上下文
- 两者结合 GLFB 实现多尺度信息融合

#### 3.3.2 多头几何预测解码器：MaskDecoder_multi_hq

**文件**：[mask_decoder_hq.py](file:///D:/desktop/Morvess/segment_anything/modeling/mask_decoder_hq.py)

这是 MorVess 的核心解码器，扩展了 SAM 原始解码器以支持多任务预测。

**Token 设计**：
```python
# 原始 SAM tokens
self.iou_token = nn.Embedding(1, transformer_dim)      # IoU 预测 token
self.mask_tokens = nn.Embedding(num_mask_tokens, transformer_dim)  # 掩码 tokens
# 新增 HQ token
self.hq_token = nn.Embedding(1, transformer_dim)       # 高质量细化 token
```

**多任务预测头**：

| 预测头 | 输入 | 输出 | 激活函数 |
|--------|------|------|----------|
| 掩码预测 | 各 mask token | (B, Q, H, W) | — |
| IoU 预测 | iou_token_out | (B, Q) | — |
| 距离图预测 | avg(mask_tokens) | (B, 1, H, W) | Sigmoid |
| 厚度图预测 | avg(mask_tokens) | (B, 1, H, W) | Softplus |

```python
# 距离图/厚度图使用平均 mask token 激活
avg_mask_token_out = torch.mean(mask_tokens_out, dim=1)

# 距离图预测
if self.predict_distance:
    dist_vec = self.distance_prediction_head(avg_mask_token_out)
    dist_logit = (dist_vec.unsqueeze(1) @ upscaled_embedding.view(b, c, h*w)).view(b,1,h,w)

# 厚度图预测
if self.predict_thickness:
    thick_vec = self.thickness_prediction_head(avg_mask_token_out)
    thick_logit = (thick_vec.unsqueeze(1) @ upscaled_embedding.view(b, c, h*w)).view(b,1,h,w)
```

#### 3.3.3 全局-局部融合模块（GLFB）：HQRefiner

**文件**：[hq_refiner.py](file:///D:/desktop/Morvess/segment_anything/modeling/hq_refiner.py)

融合三路信息 + 几何先验进行高保真拓扑重建：

```python
class HQRefiner(nn.Module):
    def __init__(self, c=256, use_geom=True, film=False, geom_in=3):
        # 融合网络：3×3 卷积促进空间交互
        self.fuse = nn.Sequential(
            nn.Conv2d(c*3 + geom_in, c, 3, padding=1),  # 通道拼接后融合
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 超网络：从 hq_token 生成动态 1×1 卷积权重
        self.hyper = nn.Sequential(
            nn.Linear(c, c), nn.ReLU(True),
            nn.Linear(c, c), nn.ReLU(True),
            nn.Linear(c, c)
        )

    def forward(self, hq_token_out, F_dec_256, F_early_256, F_last_256,
                D=None, T=None, gradD=None):
        # 拼接三路特征
        feats = torch.cat([F_dec_256, F_early_256, F_last_256], dim=1)
        # 拼接几何先验（距离图 D、厚度图 T、距离梯度 |∇D|）
        feats = torch.cat([feats, D, T, gradD], dim=1)
        fused = self.fuse(feats)
        # 动态 1×1 卷积
        dyn_w = self.hyper(hq_token_out).view(B, C, 1, 1)
        logits_hq = (fused * dyn_w).sum(dim=1, keepdim=True)
        return logits_hq
```

**在解码器中的集成**（`MaskDecoder_multi_hq.predict_masks`）：

```python
# Sobel 梯度计算
gradD_256 = _sobel_mag(dist_logit)

# GLFB 细化
logits_hq = self.hq_refiner(
    hq_token_out, F_dec_256, F_early_256, F_last_256,
    D=dist_logit, T=thick_logit, gradD=gradD_256
)

# 门控残差融合
alpha = torch.sigmoid(self.hq_gate)  # 可学习门控参数
masks = masks + alpha * logits_hq
```

#### 3.3.4 完整模型：Sam_multi_hq

**文件**：[sam_distance_hq.py](file:///D:/desktop/Morvess/segment_anything/modeling/sam_distance_hq.py)

```python
class Sam_multi_hq(nn.Module):
    def __init__(self,
        image_encoder: ImageEncoderViT_hq,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder_multi_hq,
        predict_distance: bool = False,
        predict_thickness: bool = False,
    ):
        ...

    def forward_train(self, batched_input, multimask_output, image_size):
        # 1. 输入重组: [B, N, 3, H, W] → [B*N, 3, H, W]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)

        # 2. 预处理 + 编码器（三路输出）
        input_images = self.preprocess(batched_input)
        image_embeddings, enc_early_64, enc_last_64 = self.image_encoder.forward_features(
            input_images, d_size
        )

        # 3. 提示编码（无提示模式）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        # 4. 多头解码
        low_res_masks, iou_predictions, low_res_dist, low_res_thick = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            enc_early_64=enc_early_64,
            enc_last_64=enc_last_64,
        )

        # 5. 后处理上采样 + 组装输出
        outputs = {"masks": masks, "iou_predictions": iou_predictions, ...}
        if self.predict_distance: outputs["distance_maps"] = ...
        if self.predict_thickness: outputs["thickness_maps"] = ...
        return outputs
```

### 3.4 模型注册与构建

**文件**：[build_sam.py](file:///D:/desktop/Morvess/segment_anything/build_sam.py)

```python
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_b_distance": build_sam_vit_b_distance,
    "vit_l_distance": build_sam_vit_l_distance,
    "vit_h_distance": build_sam_vit_h_distance,
    "vit_b_distance_thickness": build_sam_vit_b_distance_thickness,
    "vit_b_distance_thickness_hq": build_sam_vit_b_distance_thickness_hq  # ★ MorVess
}
```

MorVess 使用 `vit_b_distance_thickness_hq` 变体，内部调用 `_build_sam_hq()` 构建完整模型。

### 3.5 损失函数

MorVess 使用五项加权复合损失函数：

```
L_total = λ₁·L_CE + λ₂·L_Dice + λ₃·L_clDice + λ₄·L_dist + λ₅·L_thick
```

| 损失项 | 作用 | 说明 |
|--------|------|------|
| L_CE | 逐体素分类 | 交叉熵损失，惩罚分类错误 |
| L_Dice | 区域重叠 | Dice 损失，缓解类别不平衡 |
| L_clDice | 拓扑连通性 | 中心线 Dice，约束血管骨架重叠 |
| L_dist | 边界距离场回归 | Sigmoid + L1，VDM 预测与 GT 的 L1 差 |
| L_thick | 厚度场回归 | Softplus + 归一化 L1，尺度不变 |

**厚度损失的特殊处理**：
```
L_thick = |P_thick / max(P_thick) - G_thick / max(G_thick)|₁
```
预测值先经 Softplus 保证非负，然后预测和 GT 各自归一化后再算 L1，实现尺度不变性。

### 3.6 两阶段训练策略

#### Stage I：宏观特征与 2.5D 适配

| 配置项 | 值 |
|--------|------|
| 冻结组件 | SAM ViT 编码器主体 |
| 训练组件 | 2.5D Adapter + FacT 模块 + 多头解码器 + GLFB |
| 初始学习率 | 1×10⁻⁵ |
| 训练轮次 | 400 epochs |
| Warmup | 前 100 epochs |
| 学习率调度 | 多项式衰减 (power=7) |
| 图像尺寸 | 512×512 |
| Batch Size | 1（有效4 via 梯度累积） |

#### Stage II：拓扑精调

| 配置项 | 值 |
|--------|------|
| 冻结组件 | 2.5D Adapter（已收敛） |
| 训练组件 | GLFB + VDM/VTM 解码头 |
| 初始学习率 | 5×10⁻⁵ |
| 训练轮次 | 200 epochs（max_epochs=400, 从上阶段模型继续） |
| Warmup | 前 500 iterations |
| 学习率调度 | 余弦退火 |
| 图像尺寸 | 256×256 |
| Batch Size | 8 |
| 加载权重 | Stage I epoch_16.pth |

---

## 4. 数据处理流水线

### 4.1 支持的数据集

| 数据集 | 来源 | 特点 | 用途 |
|--------|------|------|------|
| Parse2022 | MICCAI 2022 肺动脉分割挑战赛 | 100例高分辨率胸部 CT，正常解剖 | 主要训练和评估 |
| AIIB2023 | 气道知情影像生物标志物 2023 | 肺纤维化 CT，含病理变形 | 跨域泛化评估 |

### 4.2 CT 数据预处理

#### 步骤 1：HU 值裁剪与归一化

```python
HU_min, HU_max = -500, 500
data = np.clip(data, HU_min, HU_max)
data = (data - HU_min) / (HU_max - HU_min) * 255.0
# Parse2022 统计量
data_mean_parse = 53.4296
data_std_parse = 68.6344
data = (data - data_mean_parse) / data_std_parse
```

#### 步骤 2：2.5D 切片构建

将 3D 体积沿深度轴切片为 5 通道输入：

```
对每个目标切片 z：取 [z-2, z-1, z, z+1, z+2] 共 5 层
边界处使用镜像填充
```

#### 步骤 3：几何先验生成

对每个 GT 掩码生成：
- **边界势场图**（VDM）：`generate_distance_map.py` / `generate_distance_process.py`
- **厚度图**（VTM）：`generate_thickness.py` / `generate_thickness_process.py`

#### 步骤 4：统一 CSV 生成

**文件**：[util_script_parse2022_ok.py](file:///D:/desktop/Morvess/preprocessing/util_script_parse2022_ok.py)

```python
# 生成包含五列路径的 training.csv
df_train = pd.DataFrame(path_list_all_train, columns=['image_pth'])
df_train['mask_pth'] = ...           # 掩码路径
df_train['boundary_pth'] = ...       # 边界势场路径
df_train['distance_pth'] = ...       # 内部距离图路径
df_train['thickness_map_pth'] = ...  # 厚度图路径
```

### 4.3 数据目录结构

预处理完成后的目录结构：

```
data/parse2022/train/2D_all_5slice/
├── training.csv                    # 训练集索引
├── test.csv                        # 测试集索引
├── PA000001/
│   ├── images/
│   │   ├── 2Dimage_0000.pkl       # 5-slice CT 图像块
│   │   ├── 2Dimage_0001.pkl
│   │   └── ...
│   ├── masks/
│   │   ├── 2Dmask_0000.pkl        # 对应掩码
│   │   └── ...
│   ├── boundary_potential/
│   │   ├── 2Dboundary_0000.pkl    # VDM 边界势场
│   │   └── ...
│   ├── internal_distance/
│   │   ├── 2Dinternal_0000.pkl    # 内部距离图
│   │   └── ...
│   └── thickness_map/
│       ├── 2Dthickness_0000.pkl   # VTM 厚度图
│       └── ...
├── PA000002/
│   └── ...
└── ...
```

### 4.4 数据集划分

**Parse2022 测试集**：
```python
test_fd_list = ['PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036']
```
其余样本作为训练集。论文中进行 5 折交叉验证。

### 4.5 多任务数据加载器

**文件**：[dataset_distance.py](file:///D:/desktop/Morvess/datasets/dataset_distance.py)

`dataset_reader_parse` 类同时加载 5 种数据，`RandomGenerator` 对所有数据施加一致的几何增强：

| 增强操作 | 概率 | 适用目标 |
|----------|------|----------|
| 随机旋转翻转 | 50% | 全部（几何一致） |
| 随机旋转 | 50% | 全部（图像三次插值，掩码最近邻） |
| 光照调整 | 50% | 仅图像 |

---

## 5. 环境配置

### 5.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| GPU | 8GB VRAM | NVIDIA L40 40GB |
| 内存 | 16GB | 32GB+ |
| 存储 | 50GB | 100GB+ |

### 5.2 软件依赖

```
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8

# 核心依赖
torch
torchvision
numpy
scipy
SimpleITK
nibabel
pandas
einops
icecream
opencv-python (cv2)
Pillow
tqdm
h5py
pickle
```

### 5.3 预训练权重

需下载 SAM ViT-Base 预训练权重：

```
sam_vit_b_01ec64.pth
```

放置路径示例：`pretrained_weights/sam_vit_b_01ec64.pth`

---

## 6. 使用指南

### 6.1 数据预处理

#### 6.1.1 生成血管距离图（VDM）

```bash
# 单文件
python generate_distance_map.py -i /path/to/mask.nii.gz -o /path/to/output -l 0.05

# 批量
python generate_distance_map.py -i /path/to/mask_folder -o /path/to/output --batch
```

#### 6.1.2 生成血管厚度图（VTM）

```bash
# 单文件
python generate_thickness.py -i /path/to/mask.nii.gz -o /path/to/output

# 批量（推荐）
python generate_thickness.py \
  -i /path/to/parse2022/train \
  -o /path/to/parse2022_thickness_map \
  --batch --out_subdir thickness_map
```

#### 6.1.3 2.5D 切片 + CSV 生成

```bash
# 1. 厚度图转 5-slice pkl
python preprocessing/util_script_parse2022_ok.py
# 脚本内函数调用顺序：
#   process_thickness_to_5slice()  # 生成 thickness pkl
#   get_unified_csv()              # 生成统一 CSV
```

### 6.2 模型训练

#### 6.2.1 Stage I 训练

```bash
python train_hq_parse_stage1.py \
  --root_path /path/to/parse2022/train/2D_all_5slice \
  --output /path/to/res_hq-par-512-stage1 \
  --num_classes 1 \
  --batch_size 1 \
  --img_size 512 \
  --base_lr 0.000010 \
  --max_epochs 400 \
  --vit_name vit_b \
  --ckpt /path/to/sam_vit_b_01ec64.pth \
  --rank 32 \
  --dice_param 0.8
```

**关键参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root_path` | — | 预处理后的 2.5D 数据根目录 |
| `--output` | — | 模型输出目录 |
| `--img_size` | 512 | Stage I 使用 512×512 |
| `--vit_name` | `vit_b` | ViT 变体（内部拼接 `_distance_thickness_hq`） |
| `--ckpt` | — | SAM 预训练权重路径 |
| `--rank` | 32 | FacT 低秩分解的秩 |
| `--dice_param` | 0.8 | Dice 损失权重 |
| `--skip_hard` | True | 跳过困难样本加速训练 |
| `--use_amp` | True | 混合精度训练 |
| `--tf32` | True | 启用 TF32 加速 |

#### 6.2.2 Stage II 训练

```bash
python train_hq_parse_stage2.py \
  --root_path /path/to/parse2022/train/2D_all_5slice \
  --output /path/to/res_hq-par-256-stage2 \
  --num_classes 1 \
  --batch_size 8 \
  --img_size 256 \
  --base_lr 0.000010 \
  --max_epochs 400 \
  --vit_name vit_b \
  --ckpt /path/to/res_hq-par-256/epoch_16.pth \
  --rank 32
```

> **注意**：Stage II 的 `--ckpt` 指向 Stage I 的原始 SAM 权重，但在代码内部会通过 `net.load_parameters()` 加载 Stage I 训练得到的模型权重（如 `epoch_16.pth`）。

### 6.3 模型测试

```bash
# Stage I 测试
python test_parse_stage1.py \
  --task parse \
  --root_path /path/to/parse2022/train/2D_all_5slice \
  --output_dir /path/to/test_output \
  --num_classes 1 \
  --img_size 512 \
  --vit_name vit_b \
  --rank 32 \
  --is_savenii

# Stage II 测试
python test_parse_stage2.py \
  --task parse \
  --root_path /path/to/parse2022/train/2D_all_5slice \
  --output_dir /path/to/test_output \
  --num_classes 1 \
  --img_size 256 \
  --vit_name vit_b \
  --rank 32 \
  --is_savenii
```

**推理流程**：

1. 从测试 CSV 逐例加载 3D 数据
2. 逐切片进行 2.5D 推理
3. 拼接为完整 3D 体积
4. 使用 Dice、clDice、HD95 等指标评估
5. 可选保存 NIfTI 格式预测结果（`--is_savenii`）

---

## 7. 评估指标

| 指标 | 缩写 | 方向 | 说明 |
|------|------|------|------|
| Dice 系数 | DSC | ↑ | 体素级空间重叠 |
| 中心线 Dice | clDice | ↑ | 血管骨架重叠，评估拓扑连通性 |
| 95% 豪斯多夫距离 | HD95 | ↓ | 边界几何一致性（mm） |
| 表观缺失率 | AMR | ↓ | 假阴性比例 |
| 检出分支比 | DBR | ↑ | 分支节点检出完整性 |
| 检出长度比 | DLR | ↑ | 血管整体长度恢复度 |

**几何一致性评估**（使用 VMTK）：
- TVV（总血管体积）一致性
- 直径分布一致性（Pearson 相关系数 + KL 散度）
- 小血管体积分数一致性

---

## 8. 消融实验结果

### 8.1 几何先验有效性

| 配置 | Dice | clDice | HD95(mm) |
|------|------|--------|----------|
| Baseline | 82.40 | 74.24 | 6.85 |
| +VDM | 83.90 | 79.32 | 5.72 |
| +VTM | 84.10 | 78.75 | 5.59 |
| **+VDM+VTM** | **86.84** | **83.22** | **4.53** |

→ VDM 降低 HD95（改善边界），VTM 提升 clDice（改善拓扑连通），两者协同效果最佳。

### 8.2 模块贡献

| 预训练 | 2.5D Adapter | GLFB | Dice |
|--------|-------------|------|------|
| ✗ | ✗ | ✗ | 0.6844 |
| ✗ | ✔ | ✗ | 0.7233 (+0.0389) |
| ✗ | ✗ | ✔ | 0.7481 (+0.0637) |
| ✗ | ✔ | ✔ | 0.7626 (+0.0782) |
| ✔ | ✗ | ✔ | 0.8033 (+0.1189) |
| ✔ | ✔ | ✗ | 0.8392 (+0.1548) |
| **✔** | **✔** | **✔** | **0.8544 (+0.1700)** |

### 8.3 FacT Rank 选择

Rank 从 2→64 逐步提升性能，在 **Rank=32** 处性能饱和（与 r=16 的差异 <1% Dice），选择 r=32 作为效率-性能最优平衡点。

---

## 9. 关键设计决策备忘

### 9.1 为何使用平均 mask token 驱动几何预测头？

距离图/厚度图是全局性质的几何场，不像分割掩码需要逐 token 的局部权重。使用所有 mask token 的均值嵌入 `mean(h_mask)` 能够捕获泛化的图像全局信息，同时保持参数高效。

### 9.2 为何采用门控残差而非直接替换？

```python
masks = masks + α · logits_hq   # α = sigmoid(learnable_gate)
```

门控残差机制允许模型在训练初期（α 接近 0）依赖原始 SAM 解码能力，随训练推进逐步激活 GLFB 的细化能力，实现渐进式提升而非破坏性替换。

### 9.3 Sobel 梯度作为几何先验的意义

```python
gradD = _sobel_mag(dist_logit)  # |∇D|
```

距离图的 Sobel 梯度 |∇D| 编码了血管边缘的位置和方向信息。与 D 和 T 一同输入 GLFB，为融合模块提供三种互补的几何线索：边界位置（D）、管径大小（T）、边缘方向（|∇D|）。

### 9.4 两阶段训练的必要性

Stage I 在大分辨率（512×512）下学习宏观空间适配，解决 2D→3D 的域迁移问题。Stage II 在小分辨率（256×256）下精调拓扑细节，冻结已收敛的 Adapter 防止遗忘，仅优化融合模块和预测头。这种渐进式策略避免了同时优化所有组件导致的训练不稳定。

---

## 10. 引用

如果您在研究中使用了 MorVess，请引用：

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

**源代码**：[https://github.com/MaoFuyou/MorVess](https://github.com/MaoFuyou/MorVess)

---

*本文档基于 MorVess 项目源代码与论文 "MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network" 生成。*
