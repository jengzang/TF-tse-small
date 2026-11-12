# 四卡小模型同卡并行试验

用户要求进一步提高利用率。四卡初次短profile（与正式训练同卡）显示缓存预热后的取数等待约0.2ms，主要GPU耗时转向MIOpen RNN、复制、矩阵乘及大量小算子。不能仅凭低瞬时利用率认定IO瓶颈。

额外同卡小模型探针：batch64，丢弃2步预热后40步，平均2.088秒/批、30.65样本/秒；同时原case00的iter101→201用188.918秒，即约1.889秒/批、33.88样本/秒。两段存在重叠但不完全相同，合计吞吐约64.5样本/秒只是初步估计，并非严格稳态对照；单路此前稳态约35.5样本/秒。未据此声称长期1.8倍或恒定满载。

调度器新增可选`--share-small`，至多两个普通small case共享一张卡，宽模型与所有FLA均保持独占。判断基于已测峰值显存：普通small各约23～25GB，双任务可放入约64GB卡；宽与FLA不纳入共享以避免OOM。所有case仍有各自模型/优化器、seed、有效batch64与固定样本预算。小模型单轮可能变慢，目标是提高整组实验合计吞吐，后续需要用各case实际metrics检验。

2026-11-10 23:10左右控制停止旧四卡队列，从最近完整epoch恢复；当轮未保存部分重跑。新队列PID5835，四卡六任务：

| GPU | case |
|---|---|
| 0 | 00小基线 + 04小FiLM |
| 1 | 01宽基线 |
| 2 | 02小混音旁路 + 08小双向LSTM |
| 3 | 03宽混音旁路 |

每进程4个数据worker，总24个，匹配服务器24核CPU配额；持续使用SSD缓存与BF16/autotune。保留OOM降batch逻辑。共享策略记录在status.json的runtime_policy中。

71项本地回归通过，服务器调度专项7项通过。探针原始记录：runs/dry_bf16_20261110/four_gpu_probe.json及four_gpu_dual_small.json。

恢复命令：

```sh
/opt/conda/bin/python scripts/run_server_queue.py --data-config config/cases/server.data.yaml --output runs/dry_bf16_20261110 --name dry_bf16_20261110 --gpus 0 1 2 3 --num-workers 4 --share-small
```

23:10:51～53，六个正式训练进程均完成首个真实训练步骤，loss有限，无OOM。case00/01仍分别从epoch16/6恢复；case02/03/04/08从各自pilot best开始。初期显存占用GPU0/2约79%/87%，宽模型GPU1/3约58%。此时瞬时利用率45%/91%/68%/73%，不能当成持续满载证明；需观察完整epoch合计吞吐。
