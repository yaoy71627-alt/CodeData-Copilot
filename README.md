# CodeData-Copilot

**AI-Powered Data Science Project Auditor**

面向开源数据科学项目的智能质量诊断与优化助手。系统可导入本地项目或公开 GitHub 仓库，自动分析代码风险、评估数据质量，并生成 AI 驱动的 Review、修复建议和标准 Patch。

## 核心能力

- 本地项目上传：支持 `.py`、`.ipynb`、`.csv`
- GitHub 项目导入：分析 `.py`、`.ipynb`、`.csv`、`.md`
- AST 静态代码质量分析
- CSV 数据质量分析
- 项目类型、语言、依赖库和结构理解
- 基于本地知识库与 Qwen 的 AI Review
- 修改前/修改后代码建议，等待用户确认
- 标准 unified diff `.patch` 文件导出
- 完整 Markdown / PDF Review 报告导出
- 七阶段分析进度与友好失败提示

系统不会运行上传项目中的代码，也不会直接修改用户项目。

## Local Run

建议在 Windows 上使用 Python 3.10 或 3.11：

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

GitHub 导入还需要系统已安装 `git`，并能访问公开 GitHub 仓库。

## Environment Variables

复制配置示例并填写本地 API Key：

```cmd
copy .env.example .env
```

编辑 `.env`：

```text
DASHSCOPE_API_KEY=你的阿里云百炼API_Key
```

没有 API Key 时，项目概览、代码检查、数据检查、健康评分和确定性修复建议仍可使用，只有在线 AI Review 不可用。

应用从系统环境变量或本地 `.env` 读取 `DASHSCOPE_API_KEY`。真实 `.env` 和 API Key 已加入 `.gitignore`，不要提交到 GitHub。

启动应用：

```cmd
python -m streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。

也可以直接双击项目根目录的 `run_app.bat`，或在 VS Code 的 **Run and Debug** 中选择 **Run CodeData-Copilot**。两种方式都会自动使用项目根目录作为工作目录，无需手动 `cd`。

## 使用方式

1. 在 Overview 的 Project Input 中选择 `Local Project` 或 `GitHub Repository`。
2. 本地模式可一次上传多个 `.py`、`.ipynb`、`.csv` 文件；GitHub 模式输入公开仓库根地址。
3. 分析完成后，通过工作区侧栏查看 Project、Code Analysis、Data Quality、AI Review、Repair 和 Report。
4. 在 Repair 页面检查修改前代码、修改后代码和说明；确认后下载 `fix_issue.patch`。
5. 在 Report 页面下载同源的 Markdown 或 PDF 完整报告。

GitHub 模式的单文件上限为 100 MB，clone 超时为 300 秒。`.git`、虚拟环境、`node_modules`、模型及 checkpoint 目录会被忽略；不支持、过大或疑似二进制文件会显示路径和跳过原因。导入内容只存放在一次性临时目录，分析完成后清理。

## 页面结构

- **Overview**：本地/GitHub 导入、项目状态与结构概览
- **Code Review**：AST 代码风险、文件定位与修复建议
- **Data Quality**：缺失值、重复值、字段类型和相关性分析
- **AI Review**：结合项目证据与知识库生成六部分诊断报告
- **Repair**：展示修改前后代码、修改说明及 Patch 下载
- **Report**：下载 Markdown 与中文 PDF 完整报告

旧 Workflow 页面及其静态解析状态均已移除。项目健康评分仅基于已导入代码、数据文件及其检查结果。

## 知识增强

Agent 会按检测问题进行轻量关键词检索，无需向量数据库。知识库包括：

- `knowledge/data_quality.md`
- `knowledge/code_quality.md`
- `knowledge/data_leakage.md`
- `knowledge/feature_engineering.md`
- `knowledge/model_evaluation.md`
- `knowledge/ml_best_practice.md`
- `knowledge/reproducibility.md`
- `knowledge/notebook_quality.md`
- `knowledge/sklearn_pandas_best_practice.md`

AI Review 固定包含：项目总结、发现的问题、问题原因、影响分析、优化建议、修改代码示例。

## 主要模块

| 文件 | 功能 |
|---|---|
| `app.py` | V2 Streamlit 产品界面与分析编排 |
| `file_loader.py` | 本地多文件校验、编码处理和临时目录管理 |
| `github_loader.py` | GitHub URL 校验、clone、过滤、跳过记录与错误分类 |
| `code_analyzer.py` | Notebook/Python AST 代码检查 |
| `data_analyzer.py` | CSV 数据质量检查 |
| `project_overview.py` | 项目类型、语言、依赖库、结构及摘要生成 |
| `agent.py` | 知识检索、Qwen Review 与确定性修复建议 |
| `patch_generator.py` | 标准 unified diff Patch 生成 |
| `project_score.py` | 可解释的项目健康评分 |
| `report.py` | 统一报告数据、Markdown 与中文 PDF 生成 |

`demo_data/test.ipynb` 和 `demo_data/train.csv` 可用于快速测试。
