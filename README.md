# speech_tftse：整理后的 TF-TSE 基线

从同级 `TF_TSE/` 已整理的代码中精选复制，日期 2026-11-10。原目录保留，Python 和配置没有改写；复制清单及 SHA-256 见 `SOURCE_MANIFEST.json`。

当前基线：**DCUNet 风格前端 → 改造的 TFGridNet 后端，298,238 参数，16 kHz 双麦输入、外置 256 维声纹、单路目标输出**。本次仅整理文件，没有改成单麦。声纹网络不计入该 TSE 参数量。

## 目录和阅读顺序

1. `config/config_tfgridnet_all.yaml`：原基线结构和训练配置。
2. `models/model_TFGridNet_k1_stage2_multi_v5.py`：完整模型；先看 Tar_Model.forward，再看 GridNetV2Block。
3. `models/local/dcunet.py`：前端卷积和归一化。CConv2d 实际使用实数 wav_conv，构造时额外增加一路条件特征通道。
4. `dataset/data_wsj0_rir_pyroom_v7.py`：旧在线混音训练集和固定开发集。
5. `train_dp_multi.py` → `trainer/trainer_dp_v1.py`：数据加载、训练、验证、保存权重。
6. `utils/loss/losses.py`：SI-SNR 与有效长度感知频谱损失；总损失组合在 trainer 中。
7. `infer.py`：离线推理、权重加载、声纹输入。

保留的配套文件：

- `models/model_TFGridNet_k1_stage2_multi_v5_bn.py`：仅替换后端两处归一化的 BatchNorm 变体。
- `config/config_aishell2_bundle.yaml`、`dataset/bundle_data.py`：已有离线音频包的训练适配。
- `config/config_batchnorm2d_corrected_20261104.yaml`：开启 padding_aware_loss 的 BatchNorm 配置。
- `emb_extract/`：可选外部 ResNet34_yk 声纹提取；使用已有 embedding 时无需推理运行该网络。
- `utils/feature.py`：STFT/iSTFT；`utils/wham_room.py`、`whamr_utils.py`：旧在线空间混音依赖。
- `scripts/cache_bundle_embeddings.py`：缓存离线数据包的注册声纹。
- `tests/`：已有基线、数据、归一化、训练精度和有效长度检查。

## 当前损失

普通训练分支：负 SI-SNR + FFT512/256/128 三种分辨率的实部、虚部、幅度 L1（共九项）+ 0.01 × pw 能量损失。

padding_aware_loss 分支：有效长度内负 SI-SNR + 带有效帧掩码的多分辨率频谱 L1，不使用 pw。验证主要使用负 SI-SNR。目标由声纹指定，没有 PIT。

## 运行

从本目录运行；先安装适合硬件平台的 torch/torchaudio，再安装依赖：

```bash
python -m pip install -r requirements.txt
# 需要旧在线混音训练时：
python -m pip install -r requirements-train.txt
```

旧房间仿真依赖 pyroomacoustics 0.7.3 和旧 SciPy 接口，推荐在兼容的 Linux/Python 3.10 环境配置。

```bash
python train_dp_multi.py --config config/config_tfgridnet_all.yaml
python infer.py --checkpoint /path/to/tse.pth.tar --mix /path/to/two_channel.wav --embedding /path/to/embedding.npy --output /tmp/target.wav
python -m unittest discover -s tests -p 'test_*.py'
```

训练前必须准备真实语料、声纹缓存、噪声/房间参数以及开发集，或按 bundle 配置准备离线数据包。配置中的外部绝对路径和相对数据路径保持来源原值，需要按实际环境调整。若从注册语音现场提声纹，还需单独提供匹配的声纹权重。

未复制：虚拟环境、缓存、训练/评测结果、数据集、大型权重、ONNX 导出、FLA 实验、新版视听造数据快照和历史计划。这是精简源码项目，不是包含数据和预训练权重的即用运行包。

注意：bundle 中把单麦复制成双通道只是旧模型接口适配，不等于已经改成真正的单麦架构。整段标准差、GroupNorm 与居中 STFT 也意味着当前基线不是完整流式实现。

## T0：真正单麦变体（2026-11-10）

新模型 `models/model_tfgridnet_mono.py` 接收 `[B,1,N]` 波形、`[B,1,256]` 声纹，输出 `[B,N]`；实测 **298,138 参数**。数据加载使用 `mono_policy: mono`，拒绝双麦，不复制音频通道。原双麦模型与三份原配置保持原样。

| 配置 | 瓶颈时间布局 | 训练loss |
|---|---|---|
| `config/config_mono_legacy.yaml` | 保留来源错序 | 来源公式及正值截断，供行为对照 |
| `config/config_mono_fixed.yaml` | 修正F/T还原 | 与mono_legacy相同，仅布局不同 |
| `config/config_mono_unified.yaml` | 修正 | 关闭正值截断，有效长度loss，零参考缺席分支 |

legacy/fixed的bundle mask依然只是padding，不是VAD；这两份配置仅保留来源loss行为，并非已复现旧在线空间造数据训练。统一配置用于后续研究；loss各项权重暂保留1作起点，尚未做真实数据量级调优。有效样本内精确全零参考表示目标缺席，采用波形L1及谱L1；其他样本用负SI-SNR及谱L1。缺席RMS独立记录，不进入活动SI-SNR均值。

先准备16k无混响单麦数据、同一麦位置且对齐的目标参考、独立同人注册；编辑新配置的 `bundle_*` 路径。SCP格式为 `样本ID 绝对音频路径`，mix/ref/aux的ID集合一致；缓存用 `torch.save({'embeddings': {id: tensor[1,256]}})`，应来自同一固定声纹网络/预处理。代码不能仅凭音频验证说话人划分、注册独立性或参考物理对齐，这些由数据清单保证。

```bash
python train_dp_multi.py --config config/config_mono_unified.yaml
python infer.py --config config/config_mono_unified.yaml --checkpoint /path/to/mono.pth.tar --mix /path/to/mono.wav --embedding /path/to/embedding.npy --output /tmp/target.wav
```

模型内部补齐STFT hop并裁回原长，短输入也支持；标准差仍基于补齐波形，padding可能影响模型特征，当前不是流式实现。全静音幅度使用下限避免平方根梯度奇异。新checkpoint保存输入/布局契约，推理和恢复训练拒绝布局不匹配或缺失契约的单麦checkpoint。

双麦权重必须显式转换，脚本只映射首层的两个音频kernel求和，保留条件kernel；其他key/shape严格检查，输出迁移报告，不复用优化器状态：

```bash
python scripts/migrate_dual_to_mono.py --checkpoint /path/to/dual.pth.tar --output /tmp/mono_legacy_init.pt
```

迁移结果为legacy布局。若微调，在 `config_mono_legacy.yaml` 中设置 `init_from: /tmp/mono_legacy_init.pt`、保持 `continue_from: ''`；恢复完整训练则用含优化器状态的 `continue_from`，二者不可同时设置。fixed配置默认从头训练。映射数值等价只针对两路完全相同的旧输入，不能证明真实双麦→单麦无损，也不保证实际提取效果。

单麦训练目录额外保存依赖源码快照和SHA-256清单。完整设计与边界见 `docs/research/2026-11-10-tf-tse-audit-design.zh-CN.md`；本次实现与验证记录见 `docs/research/2026-11-10-t0-implementation.zh-CN.md`。尚无真实数据训练、选人质量或芯片实时性结论。

## 10个独立服务器case

已加入小档约300k、大档约800k的原后端、原混音旁路、逐块FiLM、时间Gated FLA、时间BiLSTM五组对照，共10个case，全部<1M。case00/01为各档锚点，其他方向分别比较，不叠加。参数表、共享数据配置、训练/续训/整段评测命令见[服务器实验手册](docs/research/2026-11-10-server-cases.zh-CN.md)。

```bash
cp config/cases/data.example.yaml config/cases/data.server.yaml
# 编辑共享数据路径后，先做同预算短跑：
python scripts/run_case.py --case case00 --data-config config/cases/data.server.yaml --device cuda --batch-size 2 --duration 4 --max-steps 2000
# 无数据检查全部case：
python scripts/inspect_cases.py --case all --output case_profile.json
```

BF16/静音梯度及服务器入口已做CPU合成检查，CUDA与真实质量仍需服务器验证。FLA源码和LICENSE保存在本项目内，不依赖另一个项目目录。
