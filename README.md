# speech_tftse

[English README](README.en.md)

这是一个面向**目标说话人提取（target speaker extraction, TSE）**的 TensorFlow-TSE/TFGridNet 风格研究仓库。模型从混合语音和目标说话人的声纹条件中恢复目标语音，当前源码同时保留：

- 原始双麦输入的 TF-TSE 基线；
- 真正单麦输入的 T0 变体；
- offline bundle 数据集、AISHELL-2 noisy case 和多组结构消融；
- 训练、续训、评估、试听和实验 provenance 记录。

仓库是**源码与实验记录项目**，不是包含语料、预训练权重、缓存和服务器环境的即用数据包。真实数据、checkpoint、训练日志和大体积试听文件应放在仓库外部；代码只保存配置模板、轻量研究报告、必要的测试和可复现元数据。

## 当前状态

| 项目 | 当前约定 |
| --- | --- |
| 采样率 | 16 kHz，声纹提取器也要求 16 kHz |
| 目标说话人 | 一个目标说话人；不使用 PIT |
| 声纹 | ResNet34_yk 生成的 256 维 embedding，通常形状为 `[1, 256]` |
| 双麦基线 | 混合输入 `[B, 2, N]`，参考目标和注册语音为单通道 |
| 单麦 T0 | 混合输入 `[B, 1, N]`，声纹 `[B, 1, 256]`，输出 `[B, N]` |
| 输出 | 与输入目标参考同长的单通道波形 |
| 主要频域 | 512 点 STFT；训练损失另使用 512/256/128 多分辨率项 |
| 设备 | CPU 可做 smoke/test；正式训练与质量评估通常需要 CUDA/ROCm |
| 实时性 | 当前使用居中 STFT 和整段标准差，尚不是完整流式实现 |

模型参数量会随配置变化。原始小型 TFGridNet 后端约 298k 参数，`models/model_tfgridnet_mono.py` 的 unified 单麦变体约 298,138 参数；声纹网络不计入 TSE 主网络参数量。参数量、checkpoint 合约和实际训练结果应以对应配置和实验报告为准。

## 快速开始

### 1. 获取源码并创建环境

```bash
git clone https://github.com/jengzang/TF-tse-small.git
cd TF-tse-small
python3 -m venv .venv
. .venv/bin/activate
```

先根据 CPU、CUDA 或 ROCm 平台安装匹配版本的 `torch` 和 `torchaudio`，再安装本仓库依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

只有需要旧的在线空间混音训练时才安装额外依赖：

```bash
python -m pip install -r requirements-train.txt
```

`requirements-train.txt` 会再次包含基础依赖，并固定 `pyroomacoustics==0.7.3`。旧 `WhamRoom` 使用 0.8 以前的 `Source.get_rir` 接口，因此推荐 Python 3.10/Linux 和可用的 C++ 编译环境。若只做 offline bundle 推理、测试或已有数据包训练，通常不需要安装它。

### 2. 做不依赖真实语料的检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/inspect_cases.py --case all --output /tmp/tftse-case-profile.json
```

第一条命令验证数据契约、模型形状、损失、推理、case 注册和训练入口。第二条命令在 CPU 上构造 case，输出参数量和短时 profile；它不代表 CUDA 速度、长时间训练收敛或语音质量。

### 3. 运行离线推理

使用已有 256 维声纹：

```bash
python infer.py \
  --config config/config_tfgridnet_all.yaml \
  --checkpoint /path/to/checkpoint.pth.tar \
  --mix /path/to/two_channel_16k.wav \
  --embedding /path/to/embedding.npy \
  --output /tmp/target.wav
```

也可以从目标说话人的单通道 16 kHz 注册语音提取声纹：

```bash
python infer.py \
  --config config/config_tfgridnet_all.yaml \
  --checkpoint /path/to/checkpoint.pth.tar \
  --mix /path/to/two_channel_16k.wav \
  --enroll /path/to/enrollment_16k_mono.wav \
  --speaker-checkpoint /path/to/model_19.pkl \
  --output /tmp/target.wav
```

`--embedding` 与 `--enroll` 必须二选一。推理会检查 16 kHz、输入通道数、embedding 有限性、checkpoint 合约和输出长度；输入文件应为非空、有限值的 WAV/可由 `soundfile` 读取的音频。

## 仓库结构

```text
.
├── config/                 # 主配置、mono 配置、case 和服务器数据模板
├── dataset/                # legacy corpus、offline bundle、AISHELL-2 noisy 数据集
├── models/                 # TSE 模型、mono 变体、case 后端和 vendored FLA
├── trainer/                # 训练循环、精度上下文、checkpoint 保存
├── utils/                  # STFT、声纹、音频缓存、loss、空间混音辅助代码
├── emb_extract/            # 可选 ResNet34_yk 声纹提取实现
├── scripts/                # 数据准备、训练队列、评估、迁移和报告脚本
├── tests/                  # 单元/契约/smoke 测试
├── docs/research/          # 按日期保存的实验报告、摘要和 provenance JSON
├── docs/superpowers/       # 设计文档和执行计划
├── train_dp_multi.py       # 主训练入口
├── infer.py                # 离线推理入口
├── requirements.txt        # 基础依赖
└── requirements-train.txt  # 旧在线混音额外依赖
```

推荐阅读顺序：

1. `config/config_tfgridnet_all.yaml`：默认双麦基线的完整配置；
2. `models/model_TFGridNet_k1_stage2_multi_v5.py`：`Tar_Model.forward`、STFT 布局和 GridNet block；
3. `models/local/dcunet.py`：前端卷积、归一化和条件通道；
4. `dataset/bundle_data.py`、`dataset/corpus_mix.py`：离线 bundle 和在线 corpus 的样本契约；
5. `train_dp_multi.py`、`trainer/trainer_dp_v1.py`：数据加载、初始化、续训、保存和源码快照；
6. `utils/loss/losses.py`：SI-SNR、有效长度和多分辨率频谱损失；
7. `infer.py`、`scripts/evaluate_case.py`：推理和整段评估；
8. `docs/research/`：实验结果与数据/权重 provenance。

## 模型与数据契约

### 双麦基线

默认配置使用 `models.model_TFGridNet_k1_stage2_multi_v5.Tar_Model`。混合输入是 `[B, 2, N]`，参考目标为 `[B, N]`，声纹条件为 `[B, 1, 256]` 或等价的 batch 形式。模型输出单路目标波形，长度由输入裁剪/补齐逻辑决定。`mono_policy: duplicate` 可以把离线 bundle 的单麦混合复制成双通道，但这只是旧接口适配，不等于模型已经改成真正单麦。

普通分支训练损失由负 SI-SNR、多分辨率 FFT（512/256/128）的实部、虚部和幅度 L1，以及 `0.01 ×` 波形能量项组成。验证通常以负 SI-SNR 为主。目标说话人由声纹指定，因此不做 permutation invariant training。

### 单麦 T0

`config/config_mono_legacy.yaml`、`config/config_mono_fixed.yaml` 和 `config/config_mono_unified.yaml` 使用 `models/model_tfgridnet_mono.Tar_Model`。其中：

- 输入混合为 `[B, 1, N]`，不会复制成双麦；
- 声纹为 `[B, 1, 256]`；
- 输出为 `[B, N]`；
- `mono_policy: mono` 会拒绝双通道混合；
- unified 配置开启 `padding_aware_loss` 和 `target_absent_loss`，对有效长度计算损失，并单独记录目标缺席样本的输出 RMS；
- legacy/fixed 保留来源 loss 行为，用于布局/行为对照，不代表已经完成真实在线混音复现。

模型内部会为 STFT hop 补齐输入，再裁回原长；短输入可处理，但居中 STFT、整段标准差和 padding 影响意味着它仍不是完整流式模型。双麦 checkpoint 转单麦只能用显式迁移脚本，不能直接加载：

```bash
python scripts/migrate_dual_to_mono.py \
  --checkpoint /path/to/dual.pth.tar \
  --output /tmp/mono_legacy_init.pt
```

迁移脚本只映射首层两个音频 kernel 的求和并保留条件 kernel，严格检查其他 key/shape，输出迁移报告，不复用优化器状态。该映射只在两路旧输入完全相同的情况下具有数值等价意义，不能证明真实双麦到单麦无损。

### Checkpoint 合约

训练目录通常位于 `chkpt/<config.name>/`，日志位于 `logs/<config.name>/`。训练入口会保存实际配置、模型源码快照和 SHA-256 清单；运行进程已经加载的代码不会因为之后重新同步源码而自动变化，所以结果报告必须记录运行时 revision、配置和 checkpoint hash。

- `continue_from`：恢复完整训练状态，包含模型、优化器、epoch/step 等可恢复状态；
- `init_from`：只加载模型权重，优化器和训练计数从新实验开始；
- 两者互斥；若实验目录已存在且没有 `continue_from`，训练入口会拒绝覆盖；
- 单麦 checkpoint 包含输入/布局契约，推理和恢复训练会拒绝契约缺失或不匹配的文件。

## 数据准备

仓库支持两条主要数据路径。两条路径都要求 16 kHz，实际路径需要在本机或服务器配置文件中替换，不能直接使用仓库作者环境中的绝对路径。

### Legacy corpus / 在线混音

默认配置使用 `dataset_type` 未设置时的 legacy 数据入口，依赖多个语料的 speaker-to-utterance 映射、预计算声纹、噪声 SCP、房间/ RIR 参数和固定 dev/test SCP。模板中的数据位置包括：

- `data/AISHELL1/`、`data/AISHELL2/`、`data/Librispeech/`、`data/aidatatang/`、`data/3D-Speaker/`；
- `data/wham_noise/`、`data/room/`、`data/rir_noise/`；
- `train_spk2utts_path`、`train_embd_path`、`dev_*_scp` 和 `test_*_scp`。

可用 `scripts/prepare_corpora.py` 从 LibriSpeech、AISHELL-1/2 等目录建立 speaker-disjoint index、声纹缓存和固定 dev/test bundle。默认参数包含服务器路径，使用前必须显式覆盖：

```bash
python scripts/prepare_corpora.py \
  --output-dir /path/to/corpus_bundle \
  --libri-root /path/to/LibriSpeech \
  --aishell1-root /path/to/aishell1/wav \
  --aishell2-scp /path/to/aishell2_scp \
  --aishell2-root /path/to/aishell2/wav \
  --speaker-checkpoint /path/to/model_19.pkl \
  --device cuda:0
```

脚本会检查 16 kHz 单通道、speaker split 不重叠、注册语音至少 2.9 秒和 CPU/GPU embedding 一致性。它不能替代人工确认说话人身份、数据授权和参考语音物理对齐。

### Offline bundle

bundle 用三个 ID 对齐的 SCP 表示每个样本：

```text
sample_id /absolute/path/to/mix.wav
sample_id /absolute/path/to/ref.wav
sample_id /absolute/path/to/enrollment.wav
```

`mix`、`ref`、`aux` 的 ID 集合必须完全一致。双麦 bundle 的 mix 应为 `[2, N]`，参考和注册语音为单通道；单麦 bundle 的 mix 应为 `[1, N]`，并使用 `mono_policy: mono`。音频必须非空、有限且为 16 kHz；mix/aux 不能是全静音。

embedding cache 通常由 `torch.save` 保存，最小结构为：

```python
{"embeddings": {"sample_id": tensor_of_shape_[1,256]}}
```

推荐额外保存 speaker checkpoint、index hash、采样率和 `ref_duration` 元数据。使用已有 embedding 时不需要在每个样本加载声纹网络；使用 `enrollment_mode: waveform` 时则需要从 aux WAV 现场提取。

bundle 路径模板：

```bash
cp config/cases/data.example.yaml config/cases/data.server.yaml
# 编辑 data.server.yaml 中的 bundle_* 路径和 ref_duration
```

配置中的 `bundle_root` 可以重写固定 marker 前缀，适用于服务器挂载点变化；除此之外，相对路径按数据配置文件所在目录解析。

## 训练、续训和实验 case

### 默认基线训练

确认 `config/config_tfgridnet_all.yaml` 中所有数据路径有效后：

```bash
python train_dp_multi.py --config config/config_tfgridnet_all.yaml
```

训练默认 400 epochs、16 kHz、10 秒片段、batch size 8，并将结果写入 `chkpt/baseline/dcunet_tfgridnet_v5_run01/`。如果同名目录已有实验而配置没有 `continue_from`，程序会拒绝覆盖；请改 `name` 或显式恢复既有 checkpoint。

### 单麦 unified 训练

先准备真正单通道、mix/ref 对齐的 bundle 和 `[1,256]` embedding cache，再编辑 `config/config_mono_unified.yaml` 的 `bundle_*` 路径：

```bash
python train_dp_multi.py --config config/config_mono_unified.yaml
```

legacy/fixed 配置用于行为对照；unified 配置用于有效长度和目标缺席分支研究。尚无仅凭 CPU smoke 或参数量得出的真实质量、选人质量或芯片实时性结论。

### 服务器 case

`config/cases/case00_*.yaml` 至 `case09_*.yaml` 是小档/大档 TFGridNet、mix skip、FiLM、时间 FLA 和时间 BiLSTM 等独立对照。共享数据路径模板为：

```bash
cp config/cases/data.example.yaml config/cases/data.server.yaml
# 编辑 data.server.yaml
python scripts/run_case.py \
  --case case00 \
  --data-config config/cases/data.server.yaml \
  --device cuda \
  --precision bf16 \
  --batch-size 2 \
  --max-steps 2000
```

正式 case 可加入 `--epochs`、`--resume`、`--init-from`、`--evaluate-test`、`--samples-per-epoch` 和后端开关。完整参数见：

```bash
python scripts/run_case.py --help
python scripts/run_server_queue.py --help
```

`--init-from` 是模型 warm start，优化器重新开始；`--resume`/配置中的 `continue_from` 用于恢复训练。每个 case 独立命名并保留配置/源码快照，不要覆盖已有 checkpoint 或改变队列中任务的语义。

### 评估已有 checkpoint

对 mono bundle 做整段评估：

```bash
python scripts/evaluate_case.py \
  --config config/config_mono_unified.yaml \
  --checkpoint /path/to/temp_best.pth.tar \
  --mix-scp /path/to/test/mix.scp \
  --ref-scp /path/to/test/ref.scp \
  --aux-scp /path/to/test/aux.scp \
  --embedding-cache /path/to/test/embeddings.pt \
  --output /tmp/tftse-eval.json \
  --device cuda \
  --precision fp32
```

评估报告会保存输入/输出 SI-SNR、SI-SNR improvement、gain、waveform MAE、目标缺席输出 RMS，以及配置、manifest 和 checkpoint 的 SHA-256。SIR/SNR 分组指标依赖 metadata JSON，不能通过事后挑选测试样本或 checkpoint 替代固定协议。

## 损失、评估与结果解释

- 普通分支：负 SI-SNR + 多分辨率频谱的实部/虚部/幅度 L1 + 小权重波形能量项；
- `padding_aware_loss`：只在有效长度内计算负 SI-SNR 和带 mask 的频谱项；
- `target_absent_loss`：目标完全缺席的样本不进入活动 SI-SNR 平均，而是单独记录输出 RMS/相对混合能量；
- 评估优先使用固定 dev/test manifest 和 validation-best checkpoint；禁止用 test 指标选 checkpoint 后再声称独立 test 结果；
- SIR、噪声 SNR、目标活动/缺席、gain 和暂停区间应分开报告，不能只报告一个总体均值。

已有报告集中在 [`docs/research/`](docs/research/)；重要入口包括：

- [server cases 实验手册](docs/research/2026-11-10-server-cases.zh-CN.md)；
- [T0 单麦实现与验证](docs/research/2026-11-10-t0-implementation.zh-CN.md)；
- [TF-TSE 数据/实现审计设计](docs/research/2026-11-10-tf-tse-audit-design.zh-CN.md)；
- [AISHELL-2 noisy 进展](docs/research/noisy_progress_20261117/report.zh-CN.md)；
- [请求的 SIR/SNR 对比](docs/research/requested_sir_20261118/report.zh-CN.md)；
- [原始模型 reference 设计](docs/superpowers/specs/2026-11-17-original-reference-design.md)。

试听 WAV、HTML 和临时 JSON 只作为本地或服务器产物保存；仓库中的历史小型报告不代表所有生成产物都应提交。

## 验证与故障排查

### 推荐验证顺序

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/inspect_cases.py --case all --output /tmp/tftse-case-profile.json
git diff --check
git status --short
```

GPU 训练前先用短跑和少量样本检查：配置能解析、SCP ID 对齐、embedding cache 完整、checkpoint contract 一致、mixed precision 不产生 NaN。CPU smoke 通过只说明代码路径和输入契约基本成立，不说明真实质量。

### 常见问题

**找不到数据或配置中的绝对路径失效**：复制对应的 `*.example.yaml`，把所有 `data/...` 和服务器绝对路径替换成当前机器路径；不要直接修改历史研究报告中的路径。

**SCP ID 不一致**：对 `mix/ref/aux` 做集合比较，不能只按文件排序配对。`dataset/bundle_data.py` 会拒绝重复 ID、空 SCP 或集合不一致。

**embedding shape 错误**：推理的 `.npy` 允许 `(256,)`、`(1,256)` 或 `(1,1,256)`；bundle cache 的单项必须是 `(1,256)`，且值有限、范数非零。

**双麦/单麦混用**：双麦配置和单麦配置的 `input_channels`、`mono_policy`、模型类及 checkpoint contract 必须一致。不要把双麦 checkpoint 直接用于单麦模型；先运行迁移脚本。

**同名实验目录已存在**：训练入口默认保护已有结果。改 `name` 开新实验，或填写正确的 `continue_from` 恢复；不要删除旧 checkpoint 来“解决”冲突。

**pyroomacoustics 安装失败**：这是旧在线空间混音路径的额外依赖。确认 Python 3.10、编译器和 SciPy 兼容性；offline bundle/推理路径可以不安装它。

**结果与当前源码不一致**：训练进程启动后会加载源码和配置快照。检查 checkpoint 目录中的 snapshot/hash、实验报告记录的 Git revision 和服务器部署记录；不要假设重新同步代码会改变已经运行的进程。

## 服务器同步与 Git 约定

本地 checkout 是主源树。训练服务器只接收明确部署的代码/configuration，服务器上的紧急修改必须带回本地后再提交。每次部署或实验应记录：

- Git commit 和关键文件 SHA-256；
- 实际使用的配置、模型源码、依赖版本和设备；
- checkpoint 起始 revision、是否 warm start/continue；
- 运行开始结束时间、队列状态和结果报告路径。

不要提交凭据、数据集、缓存、checkpoint、大型生成音频、训练日志或虚拟环境。`artifacts/` 用于本地试听/进度产物，已在 `.gitignore` 中忽略；历史已跟踪的小型报告按原提交保留。每个 coherent、已验证的实现或实验尽快单独提交，提交前检查 diff 并运行相应测试。

## 相关文档

- [开发与实验 provenance 规则](AGENTS.md)
- [中文 agent 工作提示](AGENT_PROMPT.zh-CN.md)
- [设计文档与执行计划](docs/superpowers/)
- [研究报告索引](docs/research/)
- [GitHub repository](https://github.com/jengzang/TF-tse-small)

## 许可与第三方代码

仓库内 `models/vendor/fla/` 保存了随项目使用的 FLA 源码及其 `LICENSE`/`SOURCE.json`。使用第三方数据集、ResNet34_yk 声纹权重和旧 TF-TSE 代码时，请分别遵守其原始许可和所在机器的访问政策；本 README 不授予数据集或模型权重的再分发权。
