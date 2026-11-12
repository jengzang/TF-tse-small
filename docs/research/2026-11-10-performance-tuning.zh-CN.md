# BF16训练性能核查与调整

2026-11-10晚，用户要求继续提高卡使用率。实际CPU配额为12核；原先32核是进程可见的CPU affinity，不是可持续CPU配额。

## 实测定位

使用`scripts/profile_training_steps.py`，真实CorpusMixDataset及统一loss，CUDA同步分段计时，预热2步后统计8步。数据为已缓存原始录音，原始文件与采样协议不变。首次probe与正式训练共卡，后续exclusive对照暂停正式队列，每张卡只有一个基准进程，两张卡可各跑自己的基准。

初始batch32 profile中，卷积反向占约52%的GPU时间，其中一个BF16权重梯度内核占51%。预热后的next(loader)平均等待约0.2毫秒，loss约10毫秒。此段的主要瓶颈是卷积算法选择；不能用这一段推断所有冷读场景都没有IO瓶颈。

对照结果（秒/微批次，正式batch）：

| 模型 | microbatch/累积 | 原设置 | 新设置 | 吞吐倍率 |
|---|---|---:|---:|---:|
| 小GridNet | 64/1 | 3.742 | 1.800 | 2.08× |
| 宽GridNet | 32/2 | 3.129 | 2.088 | 1.50× |

小模型反向由3.194秒降到1.247秒；宽模型由2.357秒降到1.315秒。FP32卷积反而变慢，未采用。channels_last与算法搜索叠加的额外收益不够明确，也未采用。

## 已应用设置

保持BF16 autocast，启用MIOpen算法搜索：backend_acceleration=true、backend_benchmark=true、backend_deterministic=false。CLI为`--backend-acceleration --backend-autotune`。默认旧配置仍保留deterministic=true、benchmark=false；队列显式启用新设置。算法允许非确定性，固定seed不再保证逐位一致。

网络、参数数量、训练数据、采样seed、loss、有效batch64和训练轮数均保持。恢复配置与源码另存resumes目录，原配置/原快照保留；后续metrics.jsonl新增runtime字段记录后端与worker设置。

70项本地回归通过，正式batch短跑完成前后向及非有限梯度检查。20:27恢复双卡队列，PID30796：小基线从正式epoch6 checkpoint恢复，正在epoch7；宽基线从epoch2恢复，正在epoch3。未完成的轮次从该轮开头重跑。后续case也自动使用新后端设置。

按短跑吞吐估算32000条/epoch：小模型约15分钟、宽模型约35分钟，仍需完整epoch验证，冷读与系统负载可能增加耗时。GPU利用率是阶段性快照，不保证恒定100%；评价主要使用samples/s和每epoch时间。FLA的内部FP32与RNN/attention开销不同，不能直接套用这两个倍率。

原始分段计时及均值见`2026-11-10-training-performance.json`。

## 恢复后的真实训练确认

20:30:39，小基线正式epoch7完成101个microstep，含首步预热平均1.905秒/microstep；20:31:13，宽基线正式epoch3完成101个microstep，含预热平均2.237秒/microstep。日志loss与梯度检查正常，实际训练已复现短跑的加速趋势，完整epoch耗时仍待该轮结束。
