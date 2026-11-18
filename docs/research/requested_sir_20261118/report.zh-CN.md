# 2026-11-18 五档 SIR 对照补测

五个有干扰场景列为 SI-SNRi (dB)；目标缺席列为 10log10(Pmix/Pout)，正值越大抑制越强。禁止把目标独说或缺席样本混入 SI-SNRi。

AISHELL-2：同 64 组测试语音重混为 -5/0/5/10/20 dB，每档噪声 SNR 5/10/15/20 各16组，另52条原固定缺席样本。16k单麦无混响，FP32 GPU推理；目标、干扰、注册、噪声片段在五档间一致。原测试 SIR/重构残差误差 <0.01 dB，重混后 SIR 误差 <0.0001 dB。此为补充压力测试，-5/10/20超出新训练 SIR U[0,5]。

|模型/验证最佳轮次|SIR -5|SIR 0|SIR 5|SIR 10|SIR 20|目标缺席|
|---|---:|---:|---:|---:|---:|---:|
|senior_original_reference / 49|2.166|3.340|3.408|2.855|3.668|33.915|
|historical_batchnorm2d_external / 70|1.892|2.301|2.376|2.161|1.519|7.450|
|noisy03_paper17_local_global_context / 42|2.029|3.618|3.807|3.305|4.005|30.185|
|noisy04_paper16_global_context / 42|2.247|3.523|3.682|3.350|4.230|30.879|
|noisy01_exp11_film_waveform / 43|2.220|3.493|3.761|3.302|3.396|46.815|

历史BN为 TF_TSE/chkpt/bundle/batchnorm2d_corrected_20261104 验证最佳epoch70，旧AISHELL2+WHAMR有混响预混训练，**本次为新AISHELL2干声+MUSAN跨混音协议测试**。原训练1393说话人与当前测试全部198位声源/注册说话人交集为0；105个噪声文件基名与历史训练重叠，不能声称该历史BN测试噪声完全独立。服务器只找到原20k train和3k dev，没有同协议独立test，未将dev冒充test。历史BN各层实现和训练条件与新实验不同，不能把差异单独归因于GN/BN。

noisy03按已完成新协议实验的验证最佳output SI-SNR选取（6.9494；noisy04 6.9045，noisy01 6.8677），未按本次补测选权重。新噪声模型均完成50轮；该表不包含新启动的scale followup结果。

|目标输出相对干净参考RMS增益dB|SIR -5|SIR 0|SIR 5|SIR 10|SIR 20|
|---|---:|---:|---:|---:|---:|
|senior_original_reference|-27.391|-29.482|-30.051|-29.891|-29.538|
|historical_batchnorm2d_external|0.340|-1.777|-2.493|-2.511|-2.444|
|noisy03_paper17_local_global_context|-23.592|-25.744|-26.306|-26.298|-26.058|
|noisy04_paper16_global_context|-24.194|-26.321|-26.878|-26.828|-26.544|
|noisy01_exp11_film_waveform|-40.208|-42.276|-42.907|-43.070|-43.154|

低输出幅度会直接提高缺席抑制数值；不可脱离目标增益评判模型。尤其noisy01已有normalized_waveform_weight=1.0仍有约-43dB的衰减，说明启用同类损失本身尚无解决幅度问题的实证，scale新实验需另外验证。

三库（LibriSpeech/AISHELL1/AISHELL2）固定test：仅overlap+target-pause，五档样本数90/90/90/70/73，另40条缺席。以下复用已有逐条结果重算，exp11当前最佳checkpoint SHA256与存储测试哈希一致。

|模型|SIR -5|SIR 0|SIR 5|SIR 10|SIR 20|目标缺席|
|---|---:|---:|---:|---:|---:|---:|
|senior_epoch54|2.352|1.677|0.677|-1.775|-9.590|8.701|
|exp11_film_waveform|7.051|4.528|3.804|3.218|1.643|7.289|

exp11：单麦、修正时间轴、FiLM声纹调制、归一化波形损失权重1.0；完成100轮、验证最佳77轮。该模型在已完成三库实验中dev最佳输出SI-SNR为10.8769。原模型epoch54为师兄历史权重，仅作跨协议参考，历史训练数据划分未核实，不称为三库训练的同协议基线。没有找到三库训练的GN→BN历史记录。

复现：scripts/compare_requested_sir_20261118.py，commit 94c3c5b，服务器运行源码SHA256 09265fab45e1edbbf415886ecd56e7ba8ad0dde42b6fc78e6a4a00fd418200cc；服务器2项重混/统计测试通过（2.72秒）。逐样本结果在服务器 runs/requested_sir_20261118/ 和本地 artifacts/requested_sir_20261118/；Git仅保存小型汇总与源码。
