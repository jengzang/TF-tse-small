# 四卡服务器迁移

用户要求停止原两卡实例，所有后续工作迁移到四卡实例。2026-11-10 23:01停止旧队列后确认无run_case/run_server_queue训练进程，两卡GPU占用和显存均为0。

新实例SSH用户：yangzheng123.2120167689715888129，ksai.scnet.cn端口58222。密码不落盘。新实例24核CPU配额、4张加速卡，torch2.4.1与原版本一致。依赖可用。

`/public/home/yangzheng123/project/speech_tftse`为两个实例共同可见的共享持久目录；模型、数据索引、声纹缓存、固定dev/test、日志及源码均可直接读取，不依赖旧实例/workspace。新实例成功torch.load全部12个最新checkpoint（10个pilot+2个formal），确认模型、Adam优化器、调度器、步数及Python/NumPy/torch随机状态齐全；SHA见four-gpu-checkpoints.json。

独立恢复备份位于共享目录`backups/four_gpu_migration_20261110/`，保留每个实验latest/best/config/metrics共48个文件并逐个SHA核对；manifest.json保存原路径及哈希。原全部历史checkpoint与source_snapshot/resumes/logs也保持在原路径。

停止时已完成：小基线正式16epoch（8000 optimizer steps）、宽基线正式6epoch（3000 steps）；10case pilot各3epoch均完成。恢复小基线epoch17、宽基线epoch7；中断未完成轮次从开头重跑。新增GPU2/3分别开始case02/03正式训练，从各自pilot best初始化。

调度器CLI支持任意数量不重复GPU ID。共享队列status保留原尝试记录、迁移记录及原身份快照status.before-four-gpu.json，身份GPU列表更新为0/1/2/3。新队列PID270（重启可能变），每进程8数据worker，四进程共享24核CPU。保持BF16、后端autotune、有效batch64、每case50×32000预算及测试协议。

新实例本地音频SSD缓存会重新预热，原实例SSD缓存是可重建副本，不是唯一数据或训练结果。剩余本地空间约515GiB，缓存20GiB余量保护保持。

恢复命令：

```sh
cd /public/home/yangzheng123/project/speech_tftse
/opt/conda/bin/python scripts/run_server_queue.py --data-config config/cases/server.data.yaml --output runs/dry_bf16_20261110 --name dry_bf16_20261110 --gpus 0 1 2 3 --num-workers 8
```

新实例队列单元测试6项通过；启动后须继续检查四case的首个训练步骤与checkpoint恢复记录。

23:03:07～09，四路均完成首个真实训练反向及优化步骤：case00 epoch17、case01 epoch7、case02/03 epoch1，loss有限。首步包含编译和新SSD缓存预热，约37～39秒，不代表稳态速度。原服务器任务已停止，可关停原实例。
