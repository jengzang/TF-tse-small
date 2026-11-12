# T0 实现与验证记录

日期：2026-11-10。用户已确认16k、保留来源行为、独立布局修正与统一loss对照。

## 已实现

- 新增单麦子类models/model_tfgridnet_mono.py。只输入[B,1,N]，输出[B,N]；明确声纹[B,1,256]和有限/非零检查；首层只有1音频+1条件通道。batch1不吞麦维。内部补齐至少640、320整数倍，再裁回输入原长；使用既有STFT配置512/320/512，并约束iSTFT一致。
- legacy/fixed布局显式选择。fixed正确从[B×F,T,C]还原[B,F,T,C]再转通道优先；legacy保留原行为。保留前端、声纹和完整GridNet主干，未开始T1。
- 移除单麦forward中被覆盖的mag_phase iSTFT及未用计算；幅度sqrt前加1e-8下限保护静音梯度。这是三个单麦配置共有的数值改动，非布局对照变量。波形标准差仍包含padding，不称逐样本长度完全不敏感。
- BundleDataset新增mono模式，返回[1,N]，拒绝双麦输入。原reject/duplicate模式保留。ref同裁剪同补齐，mask保持有效长度意义。
- infer.py依据配置input_channels=1接收真单麦；默认双麦接口保留。trainer验证按3D或原4D接口正确取第0麦作输入SI-SNR基线。
- 三份独立配置mono_legacy/fixed/unified。前两者只相差时间布局，保留来源loss公式/截断；unified另关闭截断、启用padding-aware与缺席处理，不能将其收益全部归因布局。
- unified训练：活动样本负SI-SNR，精确零参考样本有效长度内预测L1；两类按样本平均，再加有效长度多分辨率谱L1。缺席不调用普通SI-SNR。谱/静音权重目前为1，是真实量级审计前的起点。验证仅活动样本计入SI-SNR/SI-SNRi；缺席记录RMS，调度loss为活动负SI-SNR/缺席L1混合均值，不能把这个loss本身当全体SI-SNR。
- checkpoint保存model_contract(input_channels,time_layout)，模型推理/恢复拒绝错布局或缺失契约。双麦旧checkpoint加载行为不变。迁移脚本独立严格映射首层；init_from只加载模型，重置优化器/epoch；continue_from仍恢复完整训练。
- 单麦训练保存source_snapshot依赖源码及哈希；实际配置仍另存config.yaml。没有修改两个来源仓库。当前目录无Git元数据，未提交或创建PR。

## 验证证据

使用 /Users/jengzang/CodeProject/audio-intelligence/TF_TSE/.venv/bin/python，在当前项目根运行：

```
python -m unittest discover -s tests -p 'test_*.py'
```

34项通过（原23项+新增11项）；验证包括：单麦298,138参数；batch1/2；1/701/1601等输入长度完整保留；静音有限输出/梯度；错误维度/零embedding拒绝；标签检查布局；首层映射重复输入数值等价及错误key/shape拒绝；迁移CLI；布局契约拒绝混用；有效长度缺席loss梯度；极小非零目标不误判缺席；数据/三配置；混合活动/缺席的训练保存与独立CLI推理；真实训练入口读取临时合成单麦音频包并warm-start/验证/保存源码快照。

新增测试先观察缺失实现/错误行为失败，再实现并回归。随机权重和临时合成音频仅证明接口与数值路径；不证明身份选择、音质或真实训练收敛。旧依赖弃用和旧文件未关闭警告仍存在，不影响本次测试通过。

## 参数和后续

三个新配置均预期同为298,138总参数/可训练参数，原双麦298,238。前端编码解码17,183，其余模块与审计表一致。本轮无FLA源码引入，无真实语料或权重下载，无远端训练或大环境安装。

真实训练前仍需数据清单、固定声纹权重/缓存来源和硬件预算。需从头训练legacy/fixed控制组，再切统一loss；统一loss下稳定后进入T1。目标选择、弱目标、相似说话人、错误注册/缺席和单独目标保真必须在真实未见说话人测试集验收，不能由本轮单元测试代替。
