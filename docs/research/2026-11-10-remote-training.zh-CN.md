# 服务器混合语料训练

## 已确认协议

单麦、16 kHz、无混响，监督为干净目标。干扰先按目标 RMS 配平，再按 `max(N(5,10²), -5)` dB 的 SIR 缩放；最后仅用共同增益避免混音削波。目标语料 LibriSpeech/AISHELL-1/AISHELL-2 按约 1:1:1 采样。注册录音与所有混音源录音不同，同一说话人的注册录音从混音候选中排除。说话人跨训练、验证、测试无交集。

训练场景：70%双人重叠、10%独说、10%目标暂停、10%目标缺席。固定验证和测试各512条，SIR 网格 -5/0/5/10/20 dB，含同混音换注册目标的成对案例。测试集不参与选优，验证集用于 best checkpoint。正式训练结束后自动用验证最优权重运行固定test，并保存test_metrics.json。

源语料只读；新索引、声纹缓存和固定评测音频保存在服务器项目 `data/corpora_dry/`。AISHELL-2 使用已审计的1393/399/199说话人分割；若某说话人所有注册候选均短于2.9秒，会明确记录排除原因，不把说话人移到其他分割。

## 运行

服务器使用 `/opt/conda/bin/python`，两张 K500SM_AI，BF16。原上传项目67个核对文件与本地一致；新增接入代码随后同步。69项本地单元测试通过，服务器梯度累积与数据入口6项专项测试通过。

`config/cases/server.data.yaml` 是服务器路径配置。`scripts/run_server_queue.py` 维护双GPU队列，持久化 status.json、各次尝试日志及30秒资源采样。每个 case 先3轮×4096条4秒混音，试训任务优先调度；所有待试训case已上卡后，空闲卡可让已通过的case进入50轮×32000条正式训练。正式训练从本 case 的试训 best 初始化；中断恢复从最近完整 epoch 恢复。

小模型microbatch64×累积1，大模型32×2，宽FLA16×4，均有效batch64。OOM时microbatch减半、累积翻倍；首轮未产出checkpoint时保留失败目录后重新开始。单case数值异常会记录失败，其余case继续。ROCm环境避免CUDA_VISIBLE_DEVICES筛选引发lazy seed错误，使用显式cuda:0/cuda:1与单卡DataParallel。

启用底层MIOpen/cuDNN后端（已有合成基准明显更快），persistent workers、预取、锁页内存、异步传输及批量验证。每轮metrics.jsonl包含耗时、吞吐、峰值显存、验证SI-SNR和缺席输出RMS。FLA聚焦幂/归一化及投影仍为FP32，以保持数值稳定，外围采用BF16。

真实数据初步smoke：小基线128条首轮31.46秒（含首次编译），验证512条9.74秒；宽FLA128条首轮65.17秒，验证31.23秒。此处样本太少且含编译，不直接外推正式epoch；应使用完整试训轮日志更新估计。前两次smoke使用初版数据索引，仅用于运行检查，不计正式对比。

## 已启动队列

服务器项目目录下：

- 调度进程 PID 9167（启动时记录，恢复后会变化）。
- 状态与日志：`runs/dry_bf16_20261110/`。
- 权重与逐轮指标：`chkpt/cases/dry_bf16_20261110/{pilot,formal}/case*/`。
- 数据：`data/corpora_dry/`，4874个独立注册声纹；训练4071人、验证511人、测试292人；AISHELL-2开发集D0487因没有2.9秒注册录音被记录排除。
- 音频本地SSD缓存：`/workspace/speech_tftse_audio_cache/`；原文件字节复制、原子落盘，多case复用。剩余空间少于20GiB时回退源盘读取。缓存不参与数据协议或目标定义。
- 两进程最初各8个数据worker，观察冷读瓶颈后从完整epoch恢复并改为各16个worker，固定评测集并行生成。模型文件与配置等59项同步哈希完全一致，见server-sync-after.json。

恢复命令（同一队列目录，活跃队列锁会拒绝重复启动）：

```sh
/opt/conda/bin/python scripts/run_server_queue.py --data-config config/cases/server.data.yaml --output runs/dry_bf16_20261110 --name dry_bf16_20261110 --gpus 0 1 --num-workers 16
```

首轮完整pilot实测（8 workers、首次缓存）：小基线257.85秒训练+6.66秒验证，宽基线422.72秒训练+12.89秒验证，对32000条正式epoch的初估分别33.7、55.3分钟。小模型第二轮253.09+6.05秒，仍受冷读影响。30秒利用率快照15次均值31.5%/72.7%，不是纯算子效率。16 workers恢复试验及后续case的缓存复用需继续观察，不能声称已经满载。宽FLA仅完成128条运行检查，完整epoch时间尚待pilot。

## 20:27性能设置更新

实测卷积算法选择为主要瓶颈，已启用backend autotune，短跑吞吐小基线2.08×、宽基线1.50×。训练队列PID30796已恢复，完整epoch进度分别6/50、2/50；后续case使用同一后端设置。算法允许非确定性，具体比较与复现方式见2026-11-10-performance-tuning.zh-CN.md。恢复命令应增加 `--backend-autotune` 到run_case；队列已自动传入，无需给run_server_queue额外参数。
