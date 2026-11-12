# 单麦多case服务器实验 Implementation Plan

**Goal:** 交付十个可独立训练/推理/检查的case，所有盒端参数<1M。
**Architecture:** 两个预算档（16/32/2和64/88/3），每档五种独立改法：原GridNet、原混音旁路、逐块FiLM、仅时间Gated FLA、时间BiLSTM上下文控制。统一修正布局、16k及loss，复用前端。原T0及双麦行为保留，新增轻量扩展点。
**Tech Stack:** torch、hyperpyyaml、unittest；FLA仅vendored torch算子，不新增大依赖。

用户已同意实现多个方向，按前轮建议实现；不用重新等待架构批准。当前无Git元数据，不创建提交/worktree。使用writing-plans与TDD，在当前会话依序实现，最终只读审查。

- [x] 新增tests/test_cases.py：先断言十case注册、参数上限、形状/梯度、基线等价、旁路与FiLM注入、时间FLA未来依赖与单残差、严格checkpoint契约。运行发现缺失case模块后再实现。
- [x] 为models/model_tfgridnet_mono.py新增无参数identity扩展点：后端融合和后端执行；旧forward数值不变。新增models/model_tfgridnet_cases.py实现五种case，构造中断言<1M。
- [x] 将只读FLA upstream/linear_attn.py与LICENSE字节一致复制到models/vendor/fla；保存来源/哈希，禁止修改参考源码。新适配器仅使用FocusedLinearAttention，保持GridNet外层norm/残差，不叠加LSTM。
- [x] 新增config/cases十份自包含模型配置和共享data.example.yaml。case00/01作为锚点；02/03混音；04/05FiLM；06/07FLA；08/09时间BiLSTM。不同case契约不互相静默加载。
- [x] scripts/run_case.py：选择case、共享数据覆盖、seed/device/precision/batch/epochs/max_steps、保存实际解析配置，调用现有训练入口。给trainer增加可选max_steps全局上限，保持旧配置默认行为；快照覆盖新增源码。
- [x] scripts/inspect_cases.py：无数据检查参数/分模块、输入输出、支持算子MAC估算、CPU/CUDA时间、内存口径；导出JSON。tests覆盖CLI和临时合成音频训练/推理往返。
- [x] 全量unittest、十case梯度、至少BF16冒烟、参数表、来源哈希；生成服务器操作手册与case清单。无真实效果结论，无远端训练。

后续用户已授权远端BF16训练，见2026-11-10-remote-pilot.md。
