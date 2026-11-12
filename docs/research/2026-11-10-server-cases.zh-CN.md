# 单麦TF-TSE：10个服务器实验case

日期：2026-11-10。状态：代码和合成检查完成，未进行真实语料训练，无质量排名。本轮不修改来源TF_TSE或FLA仓库，不需要原工程Python源码路径，也不启动远端任务。

## 1. Case清单

统一：16k、单麦输入[B,1,N]、外置声纹[B,1,256]、单目标输出[B,N]；修正前端时间布局，原前端/多尺度声纹拼接保持；FFT512/窗512/hop320、center=True。统一padding-aware loss、关闭正值截断、精确零参考走目标缺席分支，不用PIT。

小档后端C16/H32/2块；大档C64/H88/3块。原后端/混音旁路/FiLM的时间LSTM仍单向；FLA和时间BiLSTM使用未来上下文。整网均为离线模型，不能把单向时间LSTM等同于整网因果。

以下为真实实例化和本地FP32、CPU arm64单线程测量，输入batch1、1秒16k、预热2次/重复3次；所有参数均计入，全部可训练。算力列是支持算子的MAC估算，非完整模型MAC。

| Case / 配置文件名 | 独立方向 | 总参数 | MAC估算 / G | 中位耗时 / ms |
|---|---|---:|---:|---:|
| case00_small_gridnet | 小档原后端 | 298,138 | 0.719 | 37.91 |
| case01_wide_gridnet | 大档原后端 | 812,794 | 7.386 | 125.34 |
| case02_small_mix_skip | 小档原混音旁路 | 298,186 | 0.720 | 37.31 |
| case03_wide_mix_skip | 大档原混音旁路 | 812,986 | 7.387 | 123.87 |
| case04_small_film | 小档逐块FiLM | 310,586 | 0.719 | 37.48 |
| case05_wide_film | 大档逐块FiLM | 846,362 | 7.386 | 125.68 |
| case06_small_t_fla | 小档时间Gated FLA | 294,074 | 0.780 | 80.41 |
| case07_wide_t_fla | 大档时间Gated FLA | 846,970 | 9.167 | 403.19 |
| case08_small_t_bilstm | 小档时间BiLSTM | 297,754 | 0.712 | 39.70 |
| case09_wide_t_bilstm | 大档时间BiLSTM | 857,530 | 7.959 | 140.50 |

完整逐模块参数、运行环境、MAC覆盖范围和内存口径在[原始JSON](2026-11-10-case-profile.cpu.json)。配置位于config/cases/下同名.yaml。大档812,794–857,530参数，均在800k±10%内；小档294,074–310,586参数，均在300k±10%内。模型构造会拒绝总参数>=1,000,000，不以冻结参数隐藏预算。

MAC计LSTM/Linear/Conv1d/Conv2d/dense ConvTranspose2d及FLA核心矩阵乘法，1乘加=1MAC，不含FFT、归一化、门激活及逐元素运算。转置卷积包含边缘裁去的乘积，是上界估算。CPU RSS是进程累计高水位，包含导入和此前case，不能比较独占激活内存；独立内存比较请每case单独启动进程。CUDA内存/耗时需要服务器实际重测。本地FLA耗时明显高于原时间LSTM，不预设它更快或效果更好。

## 2. 每种改法具体改变什么

- **case00/01 原后端**：case00与T0统一loss基线结构/数值一致；case01扩大后端宽度、隐藏维度和层数，前端不变。容量比较以它们为各档锚点。
- **case02/03 原混音旁路**：标准差归一化后STFT的原始实/虚谱，经2→C的1×1投影，加到原后端声纹融合输出。它绕过前端谱加工，增加48/192参数。投影零初始化，初始为原基线；需要同时检查目标细节与干扰泄漏。
- **case04/05 原条件+逐块FiLM**：保留原条件路径；增加共享256→32声纹投影，以及每块独立F/T的32→2C仿射头。FiLM放在既有LayerNorm之后、序列模块之前，避免立即被同一个norm抵消。仿射头零初始化，初始为恒等；同seed下复制原主干初始化。增加12,448/33,568参数。
- **case06/07 时间Gated FLA**：只替换后端时间LSTM及输出投影，频率BiLSTM、前端、GridNet norm与外层残差保留。FocusedLinearAttention分别用attention_dim64/256、heads2/4、depthwise kernel7、focusing_factor3、不加RoPE/FFN。新adapter返回sigmoid(gate(x))*attention(x)增量，没有内部残差，外层只加一次。gate作用于已有inter_norm后的输入，与上游完整Linear_Attn_Template不同。BF16时该adapter内部FP32，其余路径遵从autocast，避免focused幂/归一化数值不稳；性能比较须记录这个精度选择。
- **case08/09 时间BiLSTM上下文对照**：仅后端时间LSTM改双向（每方向H20/H64），相近参数。与FLA同时看，避免把未来上下文收益全部解释为FLA机制收益；它不保证各分支完全相同的感受野或计算量。

本轮不叠加旁路+FiLM+FLA，不做完整后端替换或前端删减。先用这些case判断有效方向，再组合。所有case从头训练；checkpoint契约含variant/宽度/层数等，禁止不同case静默混用，T0的checkpoint也不会被自动当case checkpoint加载。

参数公式更正：GridNet每块参数为 **15HC+12H²+24H+6C**，前轮估算漏了3HC。C64/H96/3块实测889,402而非834,106。本轮因此用H88，实际812,794；不能继续引用旧估算。

## 3. 上传和环境

上传完整项目或解压提供的speech_tftse_cases_20261110.tar.gz，进入speech_tftse目录。包不含数据、权重、.venv或训练结果，含新增case、vendor源码/LICENSE和文档。

先安装适合服务器CUDA/ROCm版本的torch/torchaudio，再运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_cases.py'
python scripts/inspect_cases.py --case all --device cuda --seconds 4 --repeats 5 --output case_profile.cuda.json
```

新case只需要基础依赖，FLA是本项目内的torch实现，不依赖外部FLA仓库、flash-attn、rotary扩展或旧房间仿真。运行全部历史测试需具备旧测试依赖，原环境中46项通过；本次只用CPU验证，不声称CUDA/ROCm已验证。

## 4. 一次配置数据路径

```bash
cp config/cases/data.example.yaml config/cases/data.server.yaml
```

编辑data.server.yaml的train/dev mix/ref/aux SCP和embedding缓存路径。共享配置只接受这些路径及ref_duration，避免各case无意改采样率或模型；相对SCP路径相对data.server.yaml解析。SCP行格式是“样本ID 音频绝对路径”，mix/ref/aux ID集合严格一致。bundle_root支持旧bundle标记路径重定位，训练与评测用同一解析逻辑。

所有音频16k，mix/ref单声道等长且时间对齐，参考为同一麦位置目标。第一阶段无混响。按说话人分train/dev/test；注册与目标独立、优先不同录音/会话。模型不可能从文件长度判断身份独立性或物理参考对齐，需由数据清单保证。训练/验证数据和声纹缓存应对所有case完全相同，记录清单哈希。

缓存格式：torch.save({'embeddings': {sample_id: float32_tensor[1,256]}, 'metadata': {...}})。使用固定声纹网络/权重和预处理，推荐记录speaker_checkpoint_sha256、aux_scp_sha256、采样率及注册截取策略。现有项目提供scripts/cache_bundle_embeddings.py；有现成缓存时盒端训练/推理不需要运行声纹模型。缓存不能用另一套声纹模型输出悄悄替代。

目标缺席参考应精确全零；很轻的非零目标仍算活动。mask仅标记padding有效长度，不是VAD。数据应覆盖受控SIR强/等/弱、不同重叠率、单独目标、暂停和完全缺席。

## 5. 统一短跑和正式训练

先每case相同的2,000步上限、4秒段、batch2、seed3407检查收敛与选人，再决定正式预算：

```bash
python scripts/run_case.py --case case00 --data-config config/cases/data.server.yaml --device cuda --batch-size 2 --duration 4 --epochs 400 --max-steps 2000 --seed 3407
```

case可用case00..case09或完整名字。换case编号即可训练其他方向。若服务器支持且希望用BF16，统一给所有case加--precision bf16；默认FP32。显存有限时统一减batch-size；不要只改变某个case的有效batch后直接比较。

依次运行全部case（任一失败即停止）：

```bash
set -e
for c in case00 case01 case02 case03 case04 case05 case06 case07 case08 case09; do
  python scripts/run_case.py --case "$c" --data-config config/cases/data.server.yaml --device cuda --batch-size 2 --duration 4 --epochs 400 --max-steps 2000 --seed 3407
done
```

输出：chkpt/cases/<完整case名>/seed3407/和对应logs目录；保存实际解析配置config.yaml、逐epoch权重、temp_best.pth.tar、依赖源码快照及SHA-256。--max-steps是全局优化步数上限；epochs与步数先到者停止，比较时检查checkpoint的global_step确实一致。同seed在模型构造后重新设置采样随机流，避免参数初始化消耗随机数导致数据顺序不同。正式比较还需相同有效音频小时、优化器/调度与多seed（例如3407/3408/3409），不能用短跑随机差异排名。

恢复同一个case并增加步数上限，例如：

```bash
python scripts/run_case.py --case case06 --data-config config/cases/data.server.yaml --device cuda --batch-size 2 --duration 4 --epochs 400 --max-steps 10000 --seed 3407 --resume chkpt/cases/case06_small_t_fla/seed3407/epoch1.pth.tar
```

用实际存在的epoch权重路径。恢复global_step、优化器、scheduler、历史最优验证loss与Python/NumPy/torch/CUDA RNG；不同模型契约会被拒绝。原配置与快照保留，续训记录写resumes/run_*/。恢复以保存的epoch边界进行；若步数上限截断了一个epoch，不承诺恢复其剩余batch或worker随机状态的逐batch完全重放。旧checkpoint没有global_step时不允许有步数预算的续训。

## 6. 整段固定测试集评测

训练dev评估会受固定裁剪长度限制，最终比较请用独立固定test清单，以下脚本整段运行、不裁短、不复制麦通道：

```bash
python scripts/evaluate_case.py --config chkpt/cases/case06_small_t_fla/seed3407/config.yaml --checkpoint chkpt/cases/case06_small_t_fla/seed3407/temp_best.pth.tar --mix-scp /data/mono/test_mix.scp --ref-scp /data/mono/test_ref.scp --aux-scp /data/mono/test_aux.scp --embedding-cache /data/mono/test_embeddings.pt --device cuda --output results/case06.seed3407.json
```

结果保存每条活动目标输入/输出SI-SNR、SI-SNRi、幅度gain_db和波形MAE；缺席单列输出RMS与相对混音能量dB，不进入普通SI-SNR。输出包含配置/权重/清单哈希，方便核对来源。mean gain_db可能正负抵消，目标独说时还要看逐条绝对偏差和MAE。

可加--metadata /data/mono/test_metadata.json做分SIR/场景统计与标注暂停区评测，格式示例：

```json
{
  "weak_A": {"sir_db": -5, "scenario": "overlap"},
  "solo_A": {"scenario": "target_only"},
  "pause_A": {"scenario": "target_pause", "target_active_intervals_samples": [[0, 16000], [32000, 48000]]},
  "absent_A": {"scenario": "absent"}
}
```

活动区间是采样点，按时间排序、互不重叠，区间外记录pause_output_rms；未提供活动标注时不假装有VAD。相似说话人、错误注册也可用scenario标记。做换声纹测试时，在同一mix路径建立A/B两个样本ID，各自提供独立注册embedding及A/B目标参考；比较两份输出分别匹配所选目标，不能仅检查输出发生变化。

必须实际查看弱目标、相似说话人、错误注册/目标缺席、暂停泄漏与目标独说保真。没有真实测试证据前，任何case都只称候选。

## 7. 来源与本地验收

models/vendor/fla/linear_attn.py与LICENSE为参考仓库upstream快照的字节一致复制，来源及SHA-256见同目录SOURCE.json；没有修改原版。只使用其中FocusedLinearAttention；新的gate/norm/残差适配放在models/model_tfgridnet_cases.py，不称复现阿里完整发布模型，也未引入其权重。

本次46项测试通过：原双麦/T0回归、十case有限前向/梯度与<1M预算、BF16与静音、旁路/FiLM初始等价、逐norm条件位置、FLA未来依赖和增量残差、契约拒绝混用、同seed数据流、训练/迁移/评测CLI、固定step停止、错误/正确resume与快照保护、bundle_root重定位和全长度指标。它们只证明这些检查通过，不证明实际提取效果或芯片实时性。
