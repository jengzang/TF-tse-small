# 2026-11-19 15:15 左右服务器进度与同集补测

noisy01 active log-RMS已完成正式25/50轮，正在26轮；验证最佳23轮已复制冻结并补测。BatchNorm2D仍在该50轮之后串行等待，未开始正式pilot。旧noisy01..05及原架构对照均50轮完成，有test_metrics。noisy02最佳45轮，noisy05最佳49轮。

补测使用scripts/compare_requested_sir_20261118.py的prepare/evaluate，FP32 GPU0；相同64组语音每组五档SIR(-5,0,5,10,20)，各档SNR 5/10/15/20各16组；52条缺席。新noisy01权重快照冻结在服务器runs/progress_20261119/noisy01_snapshot。summary.json保留参数量、epoch、权重SHA256及分档SI-SNRi/输出增益/缺席抑制，逐条结果本地artifacts/progress_20261119和服务器runs/progress_20261119。未按测试结果修改训练或选择权重。

noisy01输出衰减由旧版40~43dB恢复到6~9dB，尚未恢复到目标音量；SI-SNRi在-5/0改善，5基本持平，10/20下降，缺席抑制由46.82降到12.98dB。不能声称所有指标改善。noisy05是非论文启发的宽FiLM方向，50轮完成；SI-SNRi在SIR5/10/20优于本次noisy01恢复版，但目标仍衰减约15~17dB。

当前卡任务：GPU0 paper27_absence_w1、GPU1 paper25_adversarial_identity、GPU2 paper26_absence_w01、GPU3 noisy01_active_gain。利用率快照波动较大，不能声称四卡持续满载。noisy01近期正式每轮约56~64分钟，余24~25轮约23~27小时（估算依赖负载）。原scale_followup队列仍等待GPU3释放，未覆盖原checkpoint。
