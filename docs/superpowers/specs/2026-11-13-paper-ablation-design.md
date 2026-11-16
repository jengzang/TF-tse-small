# 文献启发的22项消融及持续后备队列

用户授权自主实现更多方法/消融，准备充足后备任务避免GPU空闲；不需要重复审批。保留现有4个训练进程和旧队列身份。只改本工程，不触碰TF_TSE/FLA来源仓库。严格提取网络所有参数<1M，16k单麦干声，BF16，原三语料、注册排除及说话人划分不变。

来源核查：
- S-LTSS: https://link.springer.com/article/10.1186/s13636-023-00317-3 ，分层循环、条件归一化、声纹/提取联合优化。本批用新实现的分层时间LSTM、后置CLN以及附加声学身份头作机制消融；外部预训练声纹网络保持固定，不把附加头称作原论文完整联合训练。
- SEF-PNet: https://arxiv.org/abs/2501.11274 ，官方 https://github.com/isHuangZiling/SEF-PNet ，MIT。阅读了LCA/IFI源文件。自写精简上下文模块，用LayerNorm替代BN，以支持batch1和全局池化，零初始化最终映射并用2*sigmoid保持起始增益为1。原文LCA用sigmoid/BN。原始注册语音路线共享频谱投影并作时间交叉注意/两步融合，再生成256维内部条件；属于TF-TSE适配，非原SEF-PNet复现，原论文8k而本批16k。
- CIGAN: https://www.sciencedirect.com/science/article/pii/S1746809426006191 ，搜索服务可读摘要/方法片段，直接全文403。模型约1.03M超本项目限制。新建小条件谱判别器+梯度反转作单优化器对抗消融，非其完整域GAN/唤醒词/RNN-T复现。判别器和身份头也计入总参数，不能用训练专用排除来规避预算。

实验P14..35（同seed5513，从零，除指定结构外保持小FiLM主干）：
14 fresh_film_control
15 local_context
16 global_context
17 local_global_context
18 post_cln（替代FiLM）
19 plain_three_lstm
20 hierarchical_three_lstm
21 identity_w01
22 identity_w1
23 adversarial_w001
24 adversarial_w005
25 adversarial_identity（与24只差身份项）
26 absence_w01
27 absence_w1
28 waveform_w1
29 waveform_w3
30 raw_pool
31 raw_context
32 raw_iterative_context
33 raw_iterative_context_lca
34 lca_identity（与17只差身份项）
35 medium_lca（与既有exp13中等深度配置比较，非等参数小模型结论）

训练：每case试训3轮×4096；通过数值/验证/运行检查后正式50轮×32000，4秒，每个case有效batch64。正式从pilot最佳权重起步，Adam重置，数据epoch_offset=3，避免正式前3轮重播pilot样本；参数count、初始化、源快照记录。不同case相同seed/预算，原既有结果只作参考，以14为新矩阵控制。主损失仍presence-aware SI-SNR+谱L1，新增项独立配置；身份头用干净目标和输出共同拟合外部缓存embedding，缺席不计算身份或对抗项；条件判别器用反转的假谱梯度和正常真谱梯度，critic和generator共享Adam但通过GRL分离梯度方向，系数明确。

评估：固定dev选优，固定test只报告。每次正式结束自动报告overlap/target-pause的7档SIR(-20,-10,-5,0,5,10,20)、输入/输出/SI-SNRi/音量偏差；目标缺席、独说、暂停独立统计，禁止跨SIR总均值用于用户汇报。pair错误声纹由固定paired样本覆盖。raw模式使用同一注册文件最后2.9秒，不读取目标源作注册，所有raw读取包含空音频/非有限/全静音检查。

队列（2026-11-14修复）：`scripts/run_paper_queue.py`使用独立持久化`status.v2.json`和排他锁；扫描本项目run_case显式设备，保守地每卡一个训练任务，要求24GiB可用显存，不终止原任务。优先完成全部pilot，成功后自动进入formal50轮并评估验证最优权重。试训数据epoch_offset=0，正式=3。重启有epoch权重则恢复，无权重的冲突目录归档，监督进程退出时保留子任务，重启接管。OOM减半microbatch、倍增累积直到batch1，其他错误记录日志后继续后备任务。旧失败status.json保留作审计；不得把启动失败称作实验效果。
