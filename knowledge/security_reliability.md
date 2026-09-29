# Security and Reliability Review

- 源代码中的 API key、token、password 等字符串凭据应立即轮换，并迁移到环境变量或受控 secrets 管理。
- 报告、日志和 AI Prompt 中的凭据证据必须脱敏，不能复制完整 secret。
- eval、exec 与 os.system 会扩大动态执行风险；优先采用显式解析、映射或参数列表形式的 subprocess。
- subprocess 只有在 shell=True 等明确证据存在时才应标记为 shell 风险，普通参数列表调用不应误报。
- bare except 或 except Exception 后 pass 会吞掉错误上下文；应捕获具体异常、记录上下文或重新抛出。
- open 文件句柄应优先使用 with 上下文管理器，确保异常路径也能关闭资源。
- 机器相关的绝对路径会降低可移植性；使用配置、环境变量或项目相对路径。
- 安全规则只报告静态证据，不声称已经发生攻击或数据泄露。

