# ANDT Gen2 与 speech_tftse 各 Case 的算子兼容性分析与优化方案

**分析日期：** 2026-11-12  
**分析对象：** `ANDT_3_0_8`（ANDT Model Deployment Tool V3.0.8，目标 Gen2 第二代神经网络加速芯片）  
**模型对象：** 本仓库 `speech_tftse` 的 `models/model_tfgridnet_cases.py`、`models/model_tfgridnet_mono.py` 及 `config/cases/*.yaml`  
**结论摘要：** 当前仓库的所有时频 Case 均无法直接部署到 ANDT Gen2；主要硬阻塞是 STFT/iSTFT、复数张量、动态 data×data 矩阵乘法、运行时广播和动态波形长度。硬阻塞集中在 STFT/ISTFT、频率 MHSA + RoPE、时间 Gated Focused FLA、声纹 FiLM 的运行时广播四个方面。Case A 新增了复数 mask 与输入谱的 data×data 乘法风险；Case B/C 新增了 DepthwiseConv1D 的低加速风险和大量 Transpose/Reshape，但算子类型本身比原 FFN 更贴近官方清单。后续应以“外部 STFT/ISTFT + 显式条件融合 + 卷积/GRU 主干”建立可部署基线，再逐步吸收已验证支持的结构。

---

## 1. 分析范围与资料来源

本分析只覆盖软件工具链可见的能力，不是芯片数据手册、RTL 设计说明或板级引脚表。主要依据：

- `../ANDT_3_0_8/supported_operator_description_v2.0.html`：官方算子支持说明 V2.0；
- `../ANDT_3_0_8/help.html`：工具帮助页，含输入 shape、流式 padding、多输入、量化配置等约束；
- `../ANDT_3_0_8/docs/ANDT_Gen2_TSE_PVAD_Platform_Guide.md`：已有的平台能力整理；
- `../ANDT_3_0_8/kws_stream_tcn`、`_exp_*` 等已导出样例：实际解析产物和 `model_parser.json`；
- 本仓库 `models/model_tfgridnet_cases.py`、`models/model_tfgridnet_mono.py`、`models/model_TFGridNet_k1_stage2_multi_v5.py`、`models/vendor/fla/linear_attn.py` 和 `config/cases/*.yaml`。

需要特别强调：文档里“支持某个算子”不等于“任意参数、任意布局、任意量化方式下都能高效执行”。ANDT 对每个算子还有参数、维度、流式和量化层面的附加限制，下文会逐项指出。

---

## 2. ANDT Gen2 官方支持算子总览

### 2.1 卷积层

| 算子 | 支持情况 | 关键限制 |
|---|---|---|
| Conv1D | 支持 | kernel / stride / dilation / bias / activation 支持；group ≤ 5 或等于输入通道数；流式 W 轴左右 padding 支持，H 轴只支持顶部 padding |
| ConvTranspose1D | 支持 | 同上；stride 在流式 H 轴默认为 1 |
| DepthwiseConv1D | 名义支持 | 官方明确写“不适合利用 NPU 加速性能”，大概率走低效路径 |
| Conv2D | 支持 | 输入 [N,H,W,C]；group ≤ 5 或等于输入通道数 |
| ConvTranspose2D | 支持 | 同上 |
| DepthwiseConv2D | 名义支持 | 同样明确“不适合利用 NPU 加速性能” |

**要点：** group ≤ 5 或等于输入通道数这个限制对本项目影响不大（当前 FFN 的 depthwise 卷积都是 groups=in_channels），但 depthwise 系列“能解析”和“能加速”是两件事，必须用 ANDT 实测逐层耗时确认。

### 2.2 全连接与循环层

- **DNN / FullyConnected / Dense / MatMul / Gemm**：只支持 **data(Tensor) × parameters(weights)**，不支持任意 data×data 矩阵乘。
- **LSTM / GRU**：支持 forward / reverse / bidirectional；初始 hidden state 只确认支持 0；channel 维作为 hidden size。

**要点：** 当前 FLA 中 `q @ kv`、`q @ k` 等 data×data 矩阵乘是硬性不兼容点，不只是性能问题。GRU 是少数官方明确支持的序列建模结构，应作为可部署研究线的重要候选。

### 2.3 池化层

- AveragePooling、GlobalAveragePooling、MaxPooling、GlobalMaxPooling。
- 流式 global pooling 需要配置固定时间窗口，不能依赖无限长缓存。

### 2.4 归一化层

- BatchNormalization：支持。
- LayerNormalization：带 `*`，部分参数条件下走 Soft-SIMD（由 DSP 或 CPU 执行）；仅支持按 channel 归一化，不支持按时间轴归一化。

**要点：** 本模型的 `nn.LayerNorm(c.channels)` 是按最后一维 channel 归一化，方向上是兼容的，但大概率是 Soft-SIMD 路径。`RMSGroupNorm` 不是官方算子，需要手动拆解或替换。

### 2.5 数学算子

- Add、Sum、Sub、Multiply。
- ReciprocalSquareRoot、Sqrt、Reciprocal、Log。
- sin、cos、atan、atanh、asin、asinh。
- 广播只确认支持 initializer 向 data 广播；运行时 data 向 data 的广播不应默认可用。

**要点：** FiLM 的 gamma/beta 是运行时数据，`[B,D] → [B,T,F,D]` 的广播方向正是文档警告的类型；复数 mask 乘法是 data×data Multiply，非对称量化明确不支持。

### 2.6 激活函数

- Sigmoid、HardSigmoid、Tanh、HardTanh、ReLU、ReLU6、LeakyReLU、PReLU。
- Softplus、Softsign、SiLU、HardSwish、Mish、GELU、ELU、SELU。
- Softmax、LogSoftmax：带 `*`，部分条件下 Soft-SIMD。

**要点：** 本模型用到的 SiLU、Softplus、Sigmoid、Tanh、ReLU 都在列表内，激活本身不是主要障碍；真正的障碍是注意力结构里的 data×data matmul 和动态归一化。

### 2.7 张量操作

- Concat、Slice`*`、Split、ReduceMax、ReduceMean、Unfold`*`、Resize`*`、Reshape、Transpose`*`、ReverseSequence、Flatten。
- Slice 单轴可走硬件，多轴 Slice 走 Soft-SIMD；流式模式下 Slice/Split 不能涉及 H 轴；Flatten 仅支持 H=1。

### 2.8 工具链与图约束

1. **图输入 shape 必须固定**，动态时间轴不能以符号尺寸留给推理时决定。
2. **隐藏 tensor 最多 4 维**。
3. **推理时 batch 维度不在芯片上**，内部布局固定 HWC。
4. **没有 FFT / RFFT / IRFFT / STFT / ISTFT 算子**。
5. **没有标准多头注意力官方支持**。工具包中有 `MH_SPDAttention` 的 Soft-SIMD 注册项，只能说明存在扩展接口，不能证明标准 ONNX 注意力可在 CIM 上高效执行。
6. 流式模式每次唤醒处理一帧（W×C），W 轴支持左右 padding，H 轴只支持顶部 padding；时间轴因果左 padding 是否符合映射语义需要逐帧对比验证。
7. 部分算子由 DSP / CPU 的 Soft-SIMD 执行，实际耗时应以逐层资源报告和实板测量为准。

### 2.9 量化支持

- KWS 样例采用 `full8bit`、对称量化、`stream mode=1`，可作为第一版复现配置。
- 非对称量化只覆盖部分组合：
  - Conv/GEMM/FC 主要支持无激活或 ReLU；
  - Multiply 仅支持 data×parameter，不支持 data×data；
  - 其他算子可能回到对称量化。
- 混合路径的张量量化需遵守前后算子的兼容规则。

---

## 3. 三个 Case 的共同结构

`fla_tse_experiments/model.py` 中 `MaskFLATSE` 继承 `TargetSpeakerFLA`，共同主干为：

1. 声纹归一化与投影：`F.normalize(embedding)` → Linear → SiLU。
2. 输入尺度归一化：`mixture.std(...).clamp_min(1e-6)`。
3. STFT：`torch.stft` 得到复数谱，再拆 real/imag 成 [B,2,T,F]。
4. Stem：Conv2d(2, C, 3×3)。
5. Stem LayerNorm + FiLM 条件注入。
6. 多组 `ConditionedAxisBlock`：
   - 频率轴：LocoformerMHSA + RoPE；
   - 时间轴：Gated Focused FLA；
   - 每个块两侧各有 FFN 分支。
7. Head：Conv2d(C, 2, 3×3)。
8. Case A/B/C 均输出复数 residual mask：`(1+tanh(real)) + j·tanh(imag)`。
9. mask × 输入谱 → ISTFT → 波形。

Case 差异：

| Case | 参数量 | 结构改动 | 训练监督 |
|---|---:|---|---|
| A | 375,534 | 直接谱输出改为复数 mask | 单目标 |
| B | 678,274 | 复数 mask + C64/D128/4组 + 轻量逐点 SwiGLU + 轴向 depthwise FFN | 单目标 |
| C | 678,274 | 与 B 完全相同结构和初始权重 | 同一混音分别给两人声纹，监督两人传播语音 |

---

## 4. 共同问题与替代方案

以下问题在 A / B / C 中都存在，是当前结构无法直接部署的主要原因。

### 4.1 STFT / ISTFT / 复数张量

**问题**

- ANDT 没有 FFT、RFFT、IRFFT、STFT、ISTFT 算子。
- `torch.complex`、`spec.real`、`spec.imag`、复数乘法没有直接映射。
- 即使把复数运算全部拆成实数，FFT 本身也没有硬件路径。
- 当前输入是波形且长度动态，违反固定 shape 约束。

**替代方案**

**方案 1：STFT/ISTFT 外置，芯片只处理实数时频特征**

- 做法：主控或 DSP 先做 STFT，把 [T,F,2] 实数拼接谱作为模型输入；模型输出 mask/增强特征；主控再做 ISTFT。
- 优点：
  - 唯一确定的可行路径；
  - 芯片上模型输入输出都是实数固定 shape；
  - 与 ANDT 平台指南推荐一致；
  - 复数运算边界清晰，便于量化和调试。
- 缺点：
  - 需要外部 DSP/主控算力和带宽；
  - 工程链路变长，需要定义帧同步和缓存接口；
  - 芯片上模型无法直接端到端优化波形域目标。
- 适用：所有后续部署实验的默认前提。

**方案 2：波形域 Conv1D 端到端模型，完全绕开频域**

- 做法：用因果 Conv1D / 扩张卷积 / GRU 直接从波形到波形。
- 优点：
  - 不需要 STFT/ISTFT；
  - 卷积和 GRU 都是官方支持算子；
  - 容易设计成因果流式。
- 缺点：
  - 波形域 TSE 的表示能力和精度需要重新验证；
  - 长感受野需要很深的卷积或大 dilation，量化后可能不稳定；
  - 与现有 FLA 频域结果不可直接公平比较。
- 适用：作为独立部署基线，不影响主线研究。

**方案 3：注册自定义 STFT/ISTFT 算子**

- 做法：使用 ANDT 的 custom operator 流程注册外部实现。
- 优点：
  - 理论上可以保留端到端图；
  - 工具链明确提供自定义算子注册接口。
- 缺点：
  - 实际执行回到 CPU/DSP，未必比外置更快；
  - 需要额外实现和验证，工程成本高；
  - 需要厂商配合确认接口和性能。
- 适用：只在厂商明确支持并有参考实现时考虑。

### 4.2 频率轴 MHSA + RoPE

**问题**

- `F.scaled_dot_product_attention` 展开后是 QK^T、Softmax、data×data matmul。
- ANDT 的 MatMul/Gemm 只支持 data×weight，不支持 data×data。
- Softmax 带 `*`，部分条件 Soft-SIMD。
- RoPE 需要 sin/cos 和按位置的元素乘加；三角函数本身在数学算子列表里，但完整 RoPE 组合没有官方保证。
- 多头 reshape/permute 会引入大量 Transpose，部分走 Soft-SIMD。

**替代方案**

**方案 1：频率轴用固定窗口因果卷积替代 MHSA**

- 做法：用 Conv1D/Conv2D（沿频率轴）或 grouped 1×1 卷积替代注意力，保留残差。
- 优点：
  - 完全落在官方支持算子内；
  - 容易流式化；
  - 参数和计算可控。
- 缺点：
  - 频率轴全局依赖能力下降；
  - 需要实验验证对分离精度的影响；
  - 与当前研究结果不可直接对比。
- 适用：可部署基线的首选。

**方案 2：保留 MHSA，但拆成最小单元图单独实测**

- 做法：导出单头、固定长度、无 RoPE 的最小注意力图，单独跑 ANDT 解析，看是否插入大量 Soft-SIMD。
- 优点：
  - 能准确量化注意力路径的可行性和代价；
  - 不需要立即放弃现有结构。
- 缺点：
  - 大概率发现 data×data matmul 不可映射或极慢；
  - 结果只对该最小配置有效，外推要谨慎。
- 适用：作为诊断，不是主部署路线。

**方案 3：频率轴降级为固定窗口 pooling / Unfold + 线性混合**

- 做法：用固定窗口 Unfold/Conv 提取局部频率上下文，再用 Linear 做 channel 混合。
- 优点：
  - Unfold 名义支持；
  - 参数量低。
- 缺点：
  - Unfold 带 `*`，可能 Soft-SIMD；
  - 固定窗口的依赖范围有限。
- 适用：参数预算极紧时的候选。

### 4.3 时间轴 Gated Focused FLA

**问题**

`FocusedLinearAttention` 的实现包含：

- RMSGroupNorm：不是官方算子；
- Softplus scale：官方支持，但 scale 是运行时数据；
- q/k 的 `**focusing_factor` 幂运算和逐元素范数归一化：没有直接对应；
- `q @ k.mean(...)`、`(k^T * scale) @ (v * scale)`、`q @ kv`：都是 data×data 矩阵乘，超出 DNN 的 data×weight 约束；
- 门控 Sigmoid × attention 输出：data×data Multiply；
- head 维折叠引入大量 reshape/permute。

**替代方案**

**方案 1：GRU 替换时间轴 FLA**

- 做法：每个频点（或频带分组后）用 GRU 建模时间依赖，输出回 TF 网格。
- 优点：
  - GRU 官方支持，且是少数明确适配序列的结构；
  - 流式状态天然存在；
  - `unet_profile_512_gru` 已经开始验证这条路线。
- 缺点：
  - 每频点独立 GRU 的计算量随频点数线性增长，需要频带分组降低开销；
  - 参数量和长程依赖能力可能不如 FLA；
  - 初始 hidden state 只确认支持 0，需确认流式状态更新方式。
- 适用：可部署研究线的首选时间建模。

**方案 2：因果扩张卷积 TCN 替换 FLA**

- 做法：用 Conv1D + dilation + 残差构建时间感受野。
- 优点：
  - 完全官方支持；
  - ANDT 自带 KWS TCN 样例，可直接参考配置和流式映射；
  - 参数量容易控制在预算内。
- 缺点：
  - 感受野增长需要深度或大 dilation；
  - 长序列全局建模能力弱于 FLA；
  - 需要验证因果 padding 在流式 H 轴的语义。
- 适用：最小部署基线的默认选择。

**方案 3：固定窗口 linear attention 的近似展开**

- 做法：限制注意力窗口长度，把 `k^T v` 拆成固定窗口内的逐元素乘加，或用厂商自定义算子。
- 优点：
  - 保留 FLA 的部分设计思想；
  - 固定窗口后 shape 固定。
- 缺点：
  - data×data matmul 的根本限制仍在；
  - 展开后算子数量可能爆炸，Soft-SIMD 占比高；
  - 工程和验证成本高。
- 适用：只有在厂商确认相关路径可用时才值得投入。

### 4.4 声纹 FiLM 与运行时广播

**问题**

`FiLM.forward` 中：

```python
gamma, beta = self.affine(z).chunk(2, dim=-1)
return x * (1 + gamma[:, None, None, :]) + beta[:, None, None, :]
```

- gamma/beta 是运行时数据，不是 initializer；
- `[B,D] → [B,T,F,D]` 的广播方向没有官方保证；
- 即便能解析，也可能插入大量 Tile/Transpose 或落 Soft-SIMD；
- `F.normalize(embedding)` 也是运行时 data 归一化。

**替代方案**

**方案 1：显式 Tile/Expand embedding 后 Concat + 1×1 Conv 融合**

- 做法：把 embedding 显式扩展到 [B,T,F,D]，与时频特征 Concat，再用 1×1 Conv 或 Linear 融合。
- 优点：
  - 不依赖运行时广播；
  - Concat/Conv/Linear 均为官方算子；
  - 导出后 `model_parser.json` 容易检查。
- 缺点：
  - embedding 展开会增加中间 tensor 大小；
  - Concat 通道数增加，Conv 参数量上升；
  - 显式 Tile 本身可能走软件路径，需要实测。
- 适用：可部署基线的默认条件融合方式。

**方案 2：embedding 预先在芯片外展开为 [B,T,D] 输入**

- 做法：主控在每一帧输入时同时提供当前帧的 embedding（或预展开后的 embedding 序列），作为第二个模型输入。
- 优点：
  - 图内无需 Tile/广播；
  - 多输入由工具明确支持；
  - 输入语义清晰。
- 缺点：
  - 外部数据量增加；
  - 帧与 embedding 的对齐责任移到主控；
  - 仍需在图内显式处理 [B,T,D] 与特征的融合。
- 适用：流式部署，尤其是注册声纹固定的场景。

**方案 3：保留 FiLM，但预融合成每帧 affine 参数**

- 做法：把 speaker projection 的输出在外部或模型入口处展开成与帧对齐的 gamma/beta 序列，图内只做同形状 Multiply/Add。
- 优点：
  - 保留 FiLM 数学形式；
  - Multiply/Add 是官方算子；
  - 条件信息仍可逐层注入。
- 缺点：
  - 每层 FiLM 的参数都要展开，输入/中间 tensor 变大；
  - 多层注入时外部预处理复杂；
  - 运行时 Multiply 的量化路径需单独验证。
- 适用：希望尽量保留原模型结构时。

**方案 4：把条件从 FiLM 改为逐帧 Concat 特征**

- 做法：不做 affine，直接把 embedding 拼接到特征通道，交给后续 Conv/Linear 学习融合。
- 优点：
  - 最简单、最少算子；
  - 不涉及运行时广播；
  - 参数量低。
- 缺点：
  - 条件作用方式变弱，可能需要更多层才能达到 FiLM 的调制效果；
  - 精度需要重新训练验证。
- 适用：最小部署基线首选，后续再对比 FiLM 预展开版本。

### 4.5 动态输入尺度归一化

**问题**

`mixture.std(dim=-1).clamp_min(1e-6)` 是对整段波形做统计，流式场景无法直接实现，也引入动态 shape 和动态数值范围。

**替代方案**

1. **外部固定尺度归一化**：主控按固定 gain 或短时 RMS 做归一化后再输入。优点是图内无统计算子；缺点是需要定义一致的缩放协议，训练/部署要匹配。
2. **固定窗口 RMS 归一化**：在芯片外按固定窗口计算尺度。优点是流式可行；缺点是引入额外延迟和边界效应。
3. **去掉归一化，靠量化校准覆盖动态范围**。优点是最简单；缺点是量化误差可能放大，校准数据必须覆盖真实电平分布。
4. **用 BatchNorm 或固定参数 affine 替代**。优点是官方支持；缺点是统计意义不同，需要重新训练或微调。

### 4.6 复数 mask 乘法

**问题**

Case A/B/C 的输出都是：

`M = (1+tanh(real)) + j·tanh(imag)`，`S_hat = M × X`。

- `torch.complex` 没有直接映射；
- 复数乘法展开为实数后是 4 次 data×data Multiply + 2 次 Add；
- 非对称量化明确不支持 data×data Multiply；
- mask 与谱的乘法如果留在芯片上，必须用对称量化并单独验证误差。

**替代方案**

**方案 1：复数乘法全部展开成实数算子，留在芯片上**

`real_out = mr·x_r − mi·x_i`  
`imag_out = mr·x_i + mi·x_r`

- 优点：
  - 只用 Multiply/Add，算子官方支持；
  - 与原数学完全等价；
  - 不需要自定义算子。
- 缺点：
  - data×data Multiply 只能用对称量化；
  - 4 次乘法可能引入明显量化误差，需逐层定位；
  - tensor 数量增加。
- 适用：若希望在芯片上完成 mask 应用。

**方案 2：芯片只输出 mask，复数乘法和 ISTFT 外置**

- 优点：
  - 芯片图最简单；
  - 复数运算和 ISTFT 都在高精度域完成；
  - 量化误差只影响 mask 本身。
- 缺点：
  - 外部算力和带宽增加；
  - 需要传输实部/虚部两个 mask 输出。
- 适用：第一版部署的推荐路径。

**方案 3：输出实数幅值 mask（或带相位的实数特征）**

- 优点：
  - 避免复数乘法；
  - 输出通道减半；
  - 量化更简单。
- 缺点：
  - 相位信息缺失或近似，重建质量可能下降；
  - 与当前复数 residual mask 的研究结论不直接可比。
- 适用：精度损失可接受的轻量版本。

**方案 4：输出增强谱而非 mask**

- 优点：
  - 没有 mask × 输入的 data×data 乘法；
  - 输出直接交给外部 ISTFT。
- 缺点：
  - 失去 identity-centered 初始化的数值稳定性；
  - 高 SIR 场景的保真度需要重新验证（Case A 正是为了改善这一点）。
- 适用：作为对照，不建议直接替代 A/B/C 的核心设计。

### 4.7 固定 shape 与流式语义

**问题**

- 当前模型输入长度动态；
- ANDT 要求图输入固定数值 shape；
- 时间轴放 H 时，流式只支持顶部 padding；因果左 padding 需要逐帧验证；
- 流式 Slice/Split 不能涉及 H 轴；
- Flatten 仅支持 H=1；
- 当前结构是离线非因果的，与流式部署目标冲突。

**替代方案**

1. **固定窗口帧栈输入**：外部维护帧缓存，每触发提供固定 W×C 输入。优点是符合工具模式；缺点是上下文管理在外部。
2. **改为因果结构重训**：卷积 left padding、GRU 状态、固定窗口 pooling。优点是流式语义清晰；缺点是需要重新训练且精度可能下降。
3. **保留离线研究模型 + 另建流式部署模型**：优点是不互相干扰；缺点是维护两套结构。
4. **固定长度分块 + 块间无状态**：实现最简单，但块边界会有伪影，质量风险高。

---

## 5. Case A 详细分析

### 5.1 改动内容

Case A 保留 C32/D64/3 组原主干，仅把直接复数谱输出改为复数 residual mask：

- `M = (1 + tanh(real_head)) + j·tanh(imag_head)`
- `S_hat = M × X`
- head 小非零初始化，初始接近恒等映射。

### 5.2 引入或保留的问题

| 问题 | 来源 | 严重程度 | 说明 |
|---|---|---|---|
| STFT/ISTFT 无算子 | `torch.stft` / `torch.istft` | 硬阻塞 | ANDT 无 FFT 类算子 |
| 复数张量构造 | `torch.complex` | 硬阻塞 | 无直接映射，需展开 |
| data×data mask 乘法 | `spec * mask` | 高 | 非对称量化明确不支持；对称量化需验证 |
| 频率 MHSA + RoPE | `LocoformerMHSA` | 硬阻塞 | data×data matmul、Softmax、RoPE 均无官方路径 |
| 时间 Gated FLA | `FocusedLinearAttention` | 硬阻塞 | RMSGroupNorm、幂运算、data×data matmul |
| FiLM 运行时广播 | `FiLM.forward` | 高 | gamma/beta 为运行时数据 |
| LayerNorm | `nn.LayerNorm` | 中 | 支持 but Soft-SIMD |
| 动态输入尺度 | `mixture.std` | 中 | 流式不可直接实现 |
| 动态输入长度 | `mixture.shape[-1]` | 高 | 违反固定 shape |
| 离线非因果结构 | 整体设计 | 高 | 与流式目标冲突 |

**Case A 相对基础模型新增的算子风险：** 主要是复数 mask 与输入谱的 data×data 乘法。tanh、Conv2d、Linear 本身都在支持列表内，没有新增类别；但复数乘法必须展开为实数 Multiply/Add 才可能被解析。

### 5.3 替代方案与优缺点

**A-1：外置 STFT/ISTFT，芯片输出实数双通道 mask**

- 结构：实数谱输入 → 卷积/GRU 主干 → 输出 [mr, mi] → 外部复数乘法 + ISTFT。
- 优点：
  - 去掉图内 FFT 和复数乘法；
  - mask 输出为实数固定 shape；
  - 保留 Case A 的 residual mask 设计思想；
  - 外部高精度乘法减少量化误差。
- 缺点：
  - 外部需要实现复数乘法和 ISTFT；
  - mask 输出两个通道，传输量比幅值 mask 大；
  - 与原端到端训练目标略有差异，需要验证。
- 推荐度：高（第一版部署首选）。

**A-2：图内展开复数乘法，芯片完成 mask 应用**

- 结构：输出 mr/mi 后，在图内用实数 Multiply/Add 完成 `mr·x_r − mi·x_i`、`mr·x_i + mi·x_r`，输出增强谱。
- 优点：
  - 芯片完成更多计算，减少外部工作；
  - 算子仅为 Multiply/Add，官方支持；
  - 数学上与原模型完全等价。
- 缺点：
  - data×data Multiply 限制为对称量化；
  - 量化误差直接作用于最终谱；
  - 需要逐层输出对比定位误差。
- 推荐度：中，作为第二阶段实验。

**A-3：改为实数幅值 mask**

- 结构：只输出 m，外部做 `|X|·m` + 原相位重建。
- 优点：
  - 输出减半，量化简单；
  - 没有 data×data 复数乘法；
  - 部署最轻。
- 缺点：
  - 相位不修正，极端混响/相消场景能力受限；
  - 与 A 的复数 residual mask 研究问题不一致；
  - 可能损失部分分离质量。
- 推荐度：中低，作为对照或极轻量版本。

**A-4：保留频率 MHSA 但降级为固定窗口局部注意力**

- 结构：限制频率注意力窗口，去掉 RoPE，拆成最小单元图验证。
- 优点：
  - 保留部分频率全局建模；
  - 固定窗口后 shape 固定。
- 缺点：
  - data×data matmul 根本限制仍在；
  - 大量 Soft-SIMD，性能不可控；
  - 实现和验证成本高。
- 推荐度：低，只作为诊断。

**A-5：A 结构 + GRU/TCN 时间轴替换**

- 结构：保留复数 residual mask，但时间轴从 FLA 换成 GRU 或因果 TCN。
- 优点：
  - 时间轴落到官方支持算子；
  - 保留 A 的 mask 设计和初始化优点；
  - 可与 A 的离线结果对比时间建模损失。
- 缺点：
  - 频率 MHSA 仍是阻塞，还需替换；
  - 精度可能下降，需要重训；
  - GRU 每频点计算的参数/耗时需控制。
- 推荐度：中高，作为从 A 到可部署模型的过渡。

### 5.4 Case A 优先动作

1. 把 STFT/ISTFT 移出模型接口，输入改为固定 [T,F,2]，输出改为 [T,F,2] mask。
2. 用单元图单独验证 tanh + residual mask 的量化稳定性。
3. 若 mask 乘法留在芯片，先导出只含复数乘法展开的最小图，用 full8bit 对称量化验证。
4. 并行训练 A-deploy 版本：卷积频率轴 + GRU/TCN 时间轴 + 显式条件 Concat。

---

## 6. Case B 详细分析

### 6.1 改动内容

Case B 在 A 基础上：

- 宽度深度提升：C32/D64/3组 → C64/D128/4组；
- FFN 从全通道 Conv1d/ConvTranspose1d 替换为：

`Linear(C,2D) → SwiGLU → depthwise Conv1d(D,D,k=5) → Linear(D,C)`

- 参数量 678,274，仍低于 1M 预算。

### 6.2 引入或保留的问题

| 问题 | 来源 | 严重程度 | 说明 |
|---|---|---|---|
| 全部 A 的共同问题 | 继承 | 硬阻塞 | STFT/ISTFT、MHSA、FLA、FiLM 等 |
| DepthwiseConv1D k=5 | `DepthwiseSwiGLUFFN` | 高 | 名义支持但官方明确“不适合 NPU 加速” |
| 大量 permute/reshape | 轴向折叠 | 中高 | Transpose 带 `*`，部分 Soft-SIMD |
| 更宽更深主干 | C64/D128/4组 | 中 | 放大 MHSA/FLA/FiLM 的张量规模和 Soft-SIMD 压力 |
| SwiGLU 的 data×data Multiply | `F.silu(a)*b` | 中 | a/b 都是运行时数据，对称量化需验证 |
| chunk(2,-1) | Split | 低 | 官方支持，但多输出需检查解析 |

**相对积极的一面：** Linear、Split、SiLU、Multiply、Conv1d 都在官方清单内；B 的轻量 FFN 在“算子类别”上比原 SwiGLUConvDeconv1d 更贴近官方支持，主要风险从算子类别转移到 DepthwiseConv 的加速性能和 Transpose 数量。

### 6.3 替代方案与优缺点

**B-1：保留轻量 FFN，单独导出单元图实测**

- 做法：导出 `Linear → Split → SiLU×gate → DWConv(k=5) → Linear` 的最小单元，在 ANDT 中解析，检查 `model_parser.json` 和逐层耗时。
- 优点：
  - 直接回答“DepthwiseConv 到底走不走加速”；
  - 不需要改训练结构；
  - 结果可用于决定后续所有 FFN 设计。
- 缺点：
  - 需要准备固定 shape 的单元图和校准输入；
  - 单元图结果受 shape/量化影响，不能完全外推。
- 推荐度：高（应最先做）。

**B-2：k=5 改 k=3**

- 优点：
  - 参数和计算减少；
  - 局部上下文仍保留；
  - 若 k=5 导致低效，k=3 可能改善。
- 缺点：
  - DepthwiseConv 的低加速问题不一定随 k 变化；
  - 感受野变小，精度可能受影响；
  - 需要重训验证。
- 推荐度：中。

**B-3：depthwise Conv 改成普通 1×1 Conv / Linear 堆叠**

- 优点：
  - 完全避开 depthwise 低加速问题；
  - Linear/Conv 混合通道，映射到 CIM 路径更可靠；
  - 算子最干净。
- 缺点：
  - 失去轴向局部上下文；
  - 参数量上升（从 Dk 到 D²），可能突破 1M 预算；
  - 需要重训。
- 推荐度：中，取决于单元图实测结果。

**B-4：depthwise Conv 改成 group ≤5 的 grouped Conv**

- 优点：
  - group ≤5 是官方明确支持范围；
  - 保留部分通道分组结构；
  - 参数量介于 depthwise 和 full conv 之间。
- 缺点：
  - 需要重排通道，可能引入 Transpose；
  - 语义与原 depthwise 不同，需要重训；
  - group=5 时参数仍可能偏高。
- 推荐度：中低。

**B-5：去掉轴向折叠，把时间/频率轴统一为 Conv2d 局部卷积**

- 做法：不做 B×F 或 B×T 折叠，直接在 [T,F,D] 上用 3×3 或 (1,k)/(k,1) Conv2d。
- 优点：
  - 大幅减少 Transpose/Reshape；
  - Conv2d 官方支持且更符合 HWC 布局；
  - 流式 padding 语义更清楚。
- 缺点：
  - 感受野从一维轴向变成二维局部，建模方式变化；
  - 参数量取决于 kernel 设计；
  - 与 B 的轴向 FFN 研究问题不同。
- 推荐度：中高，可部署线可优先考虑。

**B-6：降低宽度和深度，保留轻量 FFN**

- 优点：
  - 减少 Soft-SIMD 压力和片上内存；
  - 更容易满足延迟和 L1/L2 限制；
  - 训练更快。
- 缺点：
  - B 的研究问题（省参数后更宽更深是否更好）会改变；
  - 精度可能下降。
- 推荐度：部署线采用，研究线不采用。

### 6.4 Case B 优先动作

1. 先做 B-1 单元图验证，确定 DepthwiseConv1D k=5 的实际映射。
2. 导出整个 B 的 FFN 子图（不含注意力），统计 Transpose/Slice 数量。
3. 若 Depthwise 低效，优先试 B-5（Conv2d 统一局部卷积），其次 B-3。
4. 把 B 的轻量 FFN 思想移植到可部署基线，但先降宽度到 C32/D64。

---

## 7. Case C 详细分析

### 7.1 改动内容

Case C 与 B **完全相同的结构和初始权重**，区别只在训练监督：

- 同一混音分别给两人的声纹；
- 第一次 `e_target → y`；
- 第二次 `e_interferer → x-y`；
- 目标缺席时第二路参考为完整干扰；
- 两条监督输出取均值训练；
- 等价处理 40k 条件目标，计算预算大于 A/B。

### 7.2 引入或保留的问题

| 问题 | 来源 | 严重程度 | 说明 |
|---|---|---|---|
| 全部 B 的问题 | 结构相同 | 硬阻塞 | STFT/ISTFT、MHSA、FLA、FiLM、DWConv 等 |
| 双路前向的显存/计算 | 训练策略 | 训练期问题 | 推理时单路，不影响部署图 |
| 声纹条件依赖更强 | 训练目标 | 中 | 若部署时条件融合低效，C 的收益更难兑现 |
| 训练/推理图不一致 | 监督设计 | 低 | 推理仍是单条件前向，图本身与 B 相同 |

**关键点：** C 没有引入新的部署算子。它的部署风险与 B 完全一致；额外风险在于 C 的收益依赖声纹条件路径的有效性，而 FiLM 广播正是当前最不确定的条件注入方式。如果 FiLM 在部署版本中被替换成 Concat + Conv，需要单独验证 C 的双声纹监督是否仍有效。

### 7.3 替代方案与优缺点

**C-1：推理结构沿用 B，只保留单条件前向**

- 优点：
  - 训练得到的 C 权重可以直接用于单条件推理；
  - 不引入新的部署算子；
  - 与 B 的部署路径完全共享。
- 缺点：
  - FiLM 等共同阻塞仍在；
  - 需要验证训练时的双路监督对单路推理的影响。
- 推荐度：高（部署图层面）。

**C-2：把双声纹监督迁移到可部署结构（Conv/GRU + Concat 条件）**

- 优点：
  - 保留 C 的研究收益（减少忽略声纹的捷径）；
  - 条件融合改为官方算子；
  - 可与可部署 B 对照。
- 缺点：
  - 需要重训；
  - FiLM 换成 Concat 后条件作用变弱，C 的收益可能减弱；
  - 计算预算仍高于单目标版本。
- 推荐度：高（研究线）。

**C-3：双输出头同时预测目标与干扰**

- 做法：一次前向输出目标 mask 和干扰 mask，训练时分别监督。
- 优点：
  - 训练时不需要同一混音两次前向；
  - 推理时可以只用目标头；
  - 结构对称，便于正则化（如两 mask 之和约束）。
- 缺点：
  - 参数量增加；
  - 与 C 的“同一权重、不同条件”设计不同，声纹依赖的验证结论不能直接迁移；
  - 干扰头在推理时浪费，除非保留做一致性检查。
- 推荐度：中。

**C-4：声纹对比学习/说话人辅助损失放在芯片外**

- 做法：训练时用外部 speaker encoder 辅助，部署时只保留主模型。
- 优点：
  - 部署图不增加算子；
  - 训练时条件约束更强。
- 缺点：
  - 训练流程复杂化；
  - 与当前 C 的 paired supervision 不是同一个实验问题。
- 推荐度：低中。

### 7.4 Case C 优先动作

1. 先等 B/C 训练结果，确认 paired supervision 的质量收益是否显著。
2. 若收益显著，把该监督策略迁移到“可部署 B”（卷积/GRU 主干 + Concat 条件）。
3. 单独验证 Concat 条件下双声纹监督是否仍减少忽略声纹的捷径。
4. 记录 C 的额外计算预算，避免与 A/B 做同计算量比较时误导。

---

## 8. 其他诊断变体的算子风险

`fla_tse_diagnostics/model.py` 中的变体虽然不是当前三 case 主线，但后续可能被引用，需要提前标注部署风险。

| 变体 | 新增结构 | 主要风险 |
|---|---|---|
| `condition` / `wide_condition` / `transfer_random` / `transfer_warm` | speaker projection 后加 `nn.LayerNorm(condition_dim, elementwise_affine=False)` | LayerNorm 带 `*`，Soft-SIMD；运行时 data 归一化 |
| `full_condition` | `UnitSpeakerScale`，保留完整 256D embedding | FiLM 输入通道从 32 升到 256，运行时广播压力更大 |
| `unet` / `unet_profile` | 多尺度上下采样 | Resize/ConvTranspose 名义支持，但多尺度拼接和 Transpose 数量高 |
| `unet_512` / `unet_profile_512` | n_fft=512 | 频点数更多，时间/频率轴折叠后的中间 tensor 更大 |
| `unet_profile_512_fp32` | 注意力强制 FP32 | 8bit 量化路径不匹配，部署风险极高 |
| `unet_profile_512_gru` | GRU 替换时间轴 | **GRU 官方支持，是少数部署友好改动**；但每频点独立 GRU 的耗时需评估 |
| `direct` | 直接谱输出 | 无 mask 乘法，但仍有 STFT/ISTFT、MHSA、FLA、FiLM 阻塞 |

---

## 9. 建议的优化路线

### 9.1 第一阶段：建立部署最小闭环

目标不是精度，而是把工具链和接口跑通。

1. 定义外部 STFT 参数（建议沿用 256/128，16k）。
2. 模型输入改为固定 shape 的实数谱 [T,F,2]。
3. 声纹 embedding 作为第二输入，外部预展开为 [B,T,D] 或图内显式 Concat。
4. 主干采用：Conv2d + ReLU + Add 残差 + Concat 条件 + 因果/固定窗口时间卷积。
5. 输出复数 residual mask 的实部/虚部两个通道。
6. 复数乘法和 ISTFT 外置。
7. 量化配置：full8bit 对称，stream mode 参考样例。
8. 用 ANDT 导出并检查 `model_parser.json`，确认没有异常插入的 Tile/Transpose/Slice。

**验收标准：** ANDT 解析通过、逐层无不可映射算子、流式仿真与框架输出逐帧误差在可接受范围。

### 9.2 第二阶段：逐项引入官方支持的高级结构

在最小闭环上逐项替换，每项单独 A/B：

1. **GRU 时间轴**：参考 `unet_profile_512_gru`，必要时频带分组降低计算。
2. **轻量 SwiGLU FFN**：先导出 B 的单元图确认 DWConv 映射；若低效则改 Conv2d 局部卷积或 1×1 堆叠。
3. **LayerNorm**：只在确认 Soft-SIMD 耗时可接受时使用；优先保留 ReLU/Conv 的无归一化或 BatchNorm 方案。
4. **Mask 乘法回芯片**：复数乘法展开成实数 Multiply/Add，对称量化逐层验证。
5. **FiLM 预展开**：如果 Concat 条件精度不足，尝试每帧 gamma/beta 序列的显式 FiLM。

**验收标准：** 每引入一个结构，记录精度变化、逐层耗时、L1/L2 峰值、CIM/SIMD 占比。

### 9.3 第三阶段：研究线与部署线并行

- **研究线**：继续 A/B/C 和诊断变体，目标是精度和机制验证，不因部署约束 prematurely 改设计。
- **部署线**：以最小闭环为基线，逐步吸收 GRU、轻量 FFN、显式条件融合等官方支持结构。
- **桥接实验**：把 C 的 paired supervision 迁移到部署线，验证条件建模收益是否保留。

### 9.4 量化与验证清单

- [ ] 输入/输出 shape 全部固定，时间轴语义明确。
- [ ] 校准数据覆盖噪声、目标/非目标说话人、静音、远近场、边界帧。
- [ ] full8bit 对称量化基线通过。
- [ ] data×data Multiply（mask、SwiGLU、FiLM）逐项验证。
- [ ] LayerNorm / Softmax / Transpose 的 Soft-SIMD 占比记录。
- [ ] DepthwiseConv 单元图实测。
- [ ] GRU 流式状态和初始 hidden=0 行为验证。
- [ ] 流式仿真按真实帧移逐帧对比。
- [ ] L1/L2、权重缓冲、逐层耗时报告存档。
- [ ] 实板 benchmark 与仿真结果分开记录。

---

## 10. 总结

当前 A / B / C 的研究价值和部署可行性是脱节的。三个 case 都保留了 FiLM、频率 MHSA/RoPE 和时间 Gated Focused FLA，而这些模块没有一个能直接落在 ANDT 官方支持路径上；真正硬阻塞的 STFT/ISTFT 也在图里。

具体来说：

- **Case A** 的新增风险集中在复数 mask 的 data×data 乘法；tanh/Conv/Linear 本身支持，最优先的替代方案是外置复数乘法和 ISTFT，芯片只输出实数双通道 mask。
- **Case B** 的轻量 FFN 在算子类别上反而是更贴近官方清单的部分，主要新增风险是 DepthwiseConv1D 的低加速和轴向折叠带来的大量 Transpose；应先用单元图实测，再决定保留 DWConv、改 k=3、改 Conv2d 还是改 1×1 堆叠。
- **Case C** 没有新增部署算子，风险与 B 完全一致；它的额外问题是收益依赖声纹条件路径，而 FiLM 广播正是最不确定的环节，迁移到 Concat 条件后需要重新验证 paired supervision 的收益。

后续优化应把“研究精度线”和“可部署线”分开：研究线继续当前实验；部署线从外部 STFT/ISTFT + 实数谱输入 + 显式条件 Concat + Conv/GRU 主干 + 实数 mask 输出的最小闭环开始，再逐步引入 GRU、轻量 FFN、图内复数乘法等已验证结构。这样既能继续推进 FLA 研究问题，也能避免出现精度好但完全无法导出的结果。

## 11. 本仓库每个 Case 的逐项清单

下面把当前 `config/cases` 中实际存在的模型配置逐个列出。`exp10`～`exp13` 是实验配置，它们复用或改变基础 Case；因此既要单独记录实验意图，也要继承其基础模型的算子风险。

### Case 00：`case00_small_gridnet.yaml`

- **结构**：`variant=gridnet`，`fc_dim=16`，2 层，频率轴使用原始 GridNet LSTM，时间轴同样使用原始 LSTM。
- **新增/核心算子**：相对 mono 基线引入二维时频 GridNet、频率/时间 LSTM、Conv2D、LayerNorm、reshape/permute；前后仍有 STFT/iSTFT 和复数谱路径。
- **不支持或高风险**：STFT/iSTFT/复数；LayerNorm 只能按 channel 且可能 Soft-SIMD；`permute/reshape` 受固定 4D/HWC 约束；波形长度动态。
- **优化方案**：
  1. 外置 STFT/iSTFT，保留 LSTM+Conv2D。优点是改动最小、可作为浮点对照；缺点是仍需验证 LSTM 的芯片流式状态和 LayerNorm 软件开销。
  2. 用 Conv2D+ReLU/残差替代 LSTM。优点是映射简单、吞吐稳定；缺点是长时依赖下降，需要扩张卷积堆叠。
  3. 时间轴改为官方 GRU。优点是序列算子明确支持；缺点是仍需按频带分组控制计算量。

### Case 01：`case01_wide_gridnet.yaml`

- **结构**：与 Case 00 相同，但 `fc_dim=64`、3 层、hidden=88，属于宽深版本。
- **新增问题**：没有新增算子类型，但显著放大 LSTM、LayerNorm、Transpose 和片上 SRAM/权重带宽压力；更容易超过内存或触发软件路径。
- **优化方案**：
  1. 先缩回 Case 00 尺寸，再逐层增加宽度。优点是容易定位资源拐点；缺点是需要多轮编译。
  2. 宽度保留、层数减半。优点是保留通道表达能力；缺点是时间/频率深度不足。
  3. 改为 Conv2D bottleneck（1×1 降维→3×3→1×1 升维）。优点是权重和 MAC 可控；缺点是可能损失跨轴建模能力。

### Case 02：`case02_small_mix_skip.yaml`

- **结构**：Case 00 + `raw_projection=Conv2D(2,16,1)`，将输入谱的 real/imag 作为 raw skip。
- **新增算子**：`torch.cat(real, imag)`、1×1 Conv2D、Add。
- **不支持/高风险**：real/imag 仍依赖复数 STFT；raw skip 是运行时 data×data 路径前的额外张量搬运；Concat/Transpose 可能走 Soft-SIMD。
- **优化方案**：
  1. 外部直接提供 `[real, imag]` 两通道特征，芯片内只保留 Conv2D+Add。优点是消除复数拆分；缺点是增加前端接口带宽。
  2. 把 skip 改成固定 bias 或可学习 1×1 权重。优点是避免额外输入；缺点是失去原始谱补偿能力。
  3. 用单路幅度/对数幅度输入。优点是图更简单；缺点是相位信息损失明显。

### Case 03：`case03_wide_mix_skip.yaml`

- **结构**：Case 02 的 wide 版本（64 通道、3 层、hidden=88）。
- **新增问题**：算子类型与 Case 02 相同，但 skip 的额外带宽与宽主干叠加，资源风险最高。
- **优化方案**：优先采用 Case 02 的方案 1；其次采用 bottleneck 方案压缩 raw skip。优点是保留 skip；缺点是增加 1×1 层和量化校准点。

### Case 04：`case04_small_film.yaml`

- **结构**：Case 00 + `speaker_projection: Linear(256,32)+Tanh`；每个时频块的两处 LayerNorm 后执行 FiLM：`x*(1+gamma)+beta`。
- **新增算子**：Linear、Tanh、Chunk/Split、运行时 Multiply/Add、embedding 到 `[B,T,F,C]` 的广播。
- **硬问题**：gamma/beta 与 feature 都是 runtime data，ANDT 只明确 initializer→data 广播；data×data Multiply 和动态广播不能默认映射。
- **优化方案**：
  1. embedding 在主控预扩展到固定 `[T,F,32]` 后 Concat，再用 1×1 Conv。优点是形状显式；缺点是带宽和内存增加。
  2. 将 FiLM 改为 embedding 生成固定权重的条件线性层，在每帧由主控更新。优点是芯片只做 data×weight；缺点是主控参与更多、模型表达力下降。
  3. 训练时把条件注入改成 Additive bias（只保留 beta）。优点是去掉 data×data Multiply；缺点是条件调制能力较弱。

### Case 05：`case05_wide_film.yaml`

- **结构**：Case 04 的 wide/deep 版本（64 通道、3 层、hidden=88）。
- **新增问题**：继承 FiLM 的运行时广播硬风险，同时放大 Linear、LayerNorm、Transpose 和内存压力。
- **优化方案**：先将条件维度从 32 降到 8/16，再做显式 Concat+1×1 Conv；优点是减少带宽和参数，缺点是条件区分能力可能下降。若芯片实测 Soft-SIMD 占比高，直接改为 Case 08/09 的 GRU 时间路径更稳。

### Case 06：`case06_small_t_fla.yaml`

- **结构**：Case 00 的时间轴 LSTM 替换为 `FocusedLinearAttention`，外加 `sigmoid(gate(x))*attention(x)`。
- **新增算子**：q/k/v projection、softplus、幂/范数归一化、门控 Sigmoid、data×data MatMul、head reshape/transpose。
- **硬阻塞**：`q@kv`、`q@k` 等不是 data×weight；门控是 data×data Multiply；RMS/归一化和动态 scale 无直接官方映射。
- **优化方案**：
  1. 换官方 GRU（Case 08）。优点是可流式、算子边界清楚；缺点是可能降低全局建模能力。
  2. 换因果 dilated Conv1D/Conv2D。优点是硬件友好；缺点是需要更深网络获得长感受野。
  3. 保留 FLA 仅作为 GPU 研究模型，不进入 ANDT 图。优点是保留研究价值；缺点是无法形成芯片部署结果。

### Case 07：`case07_wide_t_fla.yaml`

- **结构**：Case 06 的 wide/deep 版本。
- **新增问题**：继承 Case 06 的硬阻塞，并把每个 data×data 注意力矩阵扩大到 `fla_dim=256, heads=4`；内存和软件算子延迟风险最大。
- **优化方案**：不建议继续加宽。优先迁移到 Case 09 的 bidirectional LSTM 或 GRU/TCN；优点是能保留较宽通道，缺点是双向结构不天然适配严格因果流式。

### Case 08：`case08_small_t_bilstm.yaml`

- **结构**：时间轴替换为双向 LSTM，hidden=20，输出 Linear(40→16)。
- **新增算子**：Bidirectional LSTM、Linear、Concat/Transpose。
- **兼容性**：LSTM 是 ANDT 明确支持的循环层；但双向需要未来帧，不能直接用于低延迟因果 stream mode；隐藏状态和时间布局需固定。
- **优化方案**：
  1. 改单向 forward LSTM。优点是符合流式因果；缺点是精度可能下降。
  2. 改 GRU。优点是参数更少、状态更简单；缺点是表达能力需重训验证。
  3. 保留双向用于离线版本，另训单向部署版本。优点是兼顾效果与部署；缺点是维护两套权重和指标。

### Case 09：`case09_wide_t_bilstm.yaml`

- **结构**：Case 08 的 wide/deep 版本（64 通道、3 层、hidden=88、双向 hidden=64）。
- **新增问题**：算子类型可映射，但参数量、状态缓存、Transpose 和双向未来依赖会带来延迟/内存风险。
- **优化方案**：采用“宽度保留、hidden 减半”的折中；优点是减少状态和权重，缺点是表达力下降。若要求严格流式，直接切单向 LSTM/GRU。

### Exp10：`exp10_film_finetune.yaml`

- **实验意图**：Case 04 small FiLM 的低学习率微调（`lr=1e-4`），模型算子图不变。
- **新增算子**：无新增；完整继承 Case 04 的 FiLM 广播、LayerNorm、STFT/iSTFT 和复数风险。
- **优化建议**：不要把微调配置误认为部署优化；先解决图兼容，再用同一初始化做量化微调。优点是可复用收敛结果；缺点是量化感知微调成本增加。

### Exp11：`exp11_film_waveform.yaml`

- **实验意图**：Case 04 small FiLM + `normalized_waveform_weight=1.0`，增加波形域损失。
- **新增算子**：训练损失侧增加 waveform/STFT 相关算子；推理模型图与 Case 04 相同。
- **问题**：训练侧可以使用任意 PyTorch 算子，但不能据此认为推理芯片支持 STFT/iSTFT。该实验不会消除模型图硬阻塞。
- **优化建议**：将波形损失保留为训练约束，同时导出实数特征模型；优点是保持音质监督，缺点是训练和部署边界更复杂。

### Exp12：`exp12_film_bilstm.yaml`

- **结构**：FiLM + 双向 LSTM，small 尺寸。
- **新增算子**：在 Case 04 风险之上新增 Bidirectional LSTM、Linear 和未来帧依赖。
- **优化方案**：部署版改单向 LSTM 或 GRU；FiLM 改显式 Concat+1×1 Conv。优点是同时消除两个主要风险；缺点是需要重新训练和重新做条件融合消融。

### Exp13：`exp13_medium_deep_film.yaml`

- **结构**：FiLM，`fc_dim=32`、4 层、hidden=48。
- **新增问题**：没有新增算子类别，但层数增加会重复执行 FiLM runtime broadcast、LayerNorm 和 LSTM，Soft-SIMD 与 SRAM 压力累积。
- **优化方案**：优先减层到 2 层并用 dilation/残差补感受野；优点是部署资源更可控，缺点是深层特征能力降低。若必须保留 4 层，应逐层检查 ANDT parser 和资源报告，不应仅凭参数量判断可部署。

## 12. 后续优化路线（按优先级）

1. **建立可部署最小闭环**：外置 STFT/iSTFT；固定 `T/F/C`；实数双通道输入；Conv2D+ReLU+Add；embedding 用显式 Concat；输出实数 mask；full8bit 对称量化。
2. **把 Case 00 作为浮点参考**：先确认固定 shape、HWC、单向状态和逐层数值一致，再迁移 Case 08/09 的单向 LSTM 或 GRU。
3. **做条件融合消融**：FiLM → Additive bias → Concat+1×1 Conv，分别统计 parser、CIM/SIMD、延迟、内存和 TSE 指标。
4. **逐项恢复高级结构**：仅在最小图可编译后，单独测试 LayerNorm、depthwise、双向 LSTM、局部 attention；每次只引入一种变化。
5. **保留研究线与部署线**：Case 06/07 FLA 继续用于 GPU 精度/论文研究；部署线采用 Conv/GRU/LSTM，不把训练可运行等同于芯片可运行。
6. **每个 Case 建立机器可读清单**：记录配置、导出 shape、算子计数、Soft-SIMD 节点、权重/激活内存、量化方式、端到端延迟、SI-SDR/PESQ/PVAD 指标和失败原因。

## 13. 结论

ANDT 的“支持算子列表”足以指导模型重构，但不能替代逐层解析和实板测量。当前仓库的主要问题不是某一个激活函数，而是模型边界仍是训练友好的动态波形/复数时频图：STFT/iSTFT、复数 mask、运行时广播和 FLA 的 data×data 注意力都超出可靠部署范围。最现实的下一步是以 Case 00 为浮点参考，建立“外部音频前端 + 实数固定时频输入 + Conv/GRU/LSTM + 显式条件融合”的部署基线，再通过单变量实验逐步恢复质量。
