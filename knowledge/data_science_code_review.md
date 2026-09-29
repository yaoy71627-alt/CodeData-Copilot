# Data Science Code Review

- 任何 scaler、imputer、PCA、特征选择或重采样步骤都应在数据划分之后，仅使用训练集拟合。
- 验证集和测试集只能调用 transform/predict/score；model.fit(X_test, y_test) 是明确的数据边界违规。
- 目标列必须从特征矩阵中显式排除，避免 target 同时出现在 X 与 y。
- 分类任务划分应评估 stratify 的必要性；所有随机划分与常见随机模型应记录 random_state。
- 模型评价必须使用未参与拟合的数据，并选择与业务代价和任务类型相匹配的多个指标。
- 训练集上的高分不能替代验证/测试集评价；调参期间反复查看测试集会造成选择偏差。
- 交叉验证是稳定性评估工具，不是所有项目的强制要求；时间序列和分组数据需要匹配的数据划分方式。
- sklearn Pipeline 可以把预处理与模型拟合边界封装在训练流程中，降低手工顺序错误风险。
- Notebook 应从干净内核按顺序运行；乱序执行与保存的 error output 都是可复现性证据。
