# 双语项目 README 设计

## 背景

仓库已经包含双麦 TF-TSE 基线、真正单麦变体、离线 bundle 数据适配、服务器 case 队列和多轮实验记录，但现有 README 主要是中文历史整理笔记，缺少面向新读者的入口、命令索引和数据契约说明。当前工作区还存在本地试听及进度产物，不能随源码推送。

## 目标

1. 把 `README.md` 改成完整的中文开发者文档。
2. 新增 `README.en.md`，与中文文档保持同一结构和命令语义。
3. 明确当前实现的模型输入输出、采样率、声纹格式、双麦/单麦边界和 checkpoint 约束。
4. 为常见工作流提供可复制的命令：环境安装、CPU smoke、训练、续训、推理、case profile、双麦到单麦迁移。
5. 链接已有研究报告和实验手册，说明报告、checkpoint、数据、日志的 provenance 关系。
6. 将本地生成的 `artifacts/` 加入 `.gitignore`，保留工作区文件但避免新产物进入 Git。

## 文档结构

两份 README 使用相同章节顺序：

1. 项目定位、当前状态、快速导航和中英文链接。
2. 模型与数据契约：16 kHz、声纹维度、双麦/单麦形状、输出、损失和非流式限制。
3. 仓库目录和阅读顺序。
4. 环境安装与依赖分层：基础推理/测试依赖和旧在线混音训练依赖。
5. 数据准备：legacy corpus、offline bundle、SCP、embedding cache、路径替换和数据划分责任。
6. 验证：单元测试、case CPU profile、短跑和 GPU/真实质量验证边界。
7. 训练与恢复：主配置、mono 配置、`continue_from` 和 `init_from` 的区别、输出目录与源码快照。
8. 推理与评估：embedding/enrollment 两种输入、checkpoint 加载、SIR/SNR 对比和试听产物。
9. 实验队列、服务器同步、提交和 provenance 约定。
10. 常见问题、限制、已有研究资料索引和引用入口。

## 命令与准确性约束

- 所有命令从仓库根目录运行，并使用仓库中现有的脚本名和配置名。
- 命令示例中的数据、checkpoint 和服务器路径使用占位符；不得写入本机私有绝对路径或凭据。
- README 不宣称未完成的真实数据训练、CUDA 性能、实时性或语音质量结论。
- 对依赖真实数据或 GPU 的步骤显式标注前置条件；CPU 检查与完整训练结果分开描述。
- 文档链接只指向已跟踪的文件或明确说明的外部仓库。

## `.gitignore` 处理

只新增 `artifacts/` 规则，不删除已有规则，也不删除工作区中的本地文件。已经被 Git 跟踪的历史试听文件继续保留；新增的 WAV、HTML、进度 JSON 等生成物默认不再显示为未跟踪文件。

## 验证

实现后执行：

1. `python -m unittest discover -s tests -p 'test_*.py'`。
2. `python scripts/inspect_cases.py --case all --output /tmp/tftse-case-profile.json`（若当前环境缺少模型依赖，则记录实际阻塞原因）。
3. 检查 README 中的本地链接、代码块路径和 `git diff --check`。
4. 检查 `git status --short`，确认新生成的 `artifacts/` 不再作为未跟踪文件出现。
