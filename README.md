# CodeData-Copilot

**面向 Python 机器学习项目的代码与数据智能审查系统**

CodeData-Copilot 面向课程实验、科研原型及开源机器学习项目，融合静态代码分析、数据质量检查、跨文件关联和大语言模型辅助解释，帮助使用者定位代码缺陷、数据异常和建模流程风险。系统以可追溯的检测证据为基础生成审查结果，并支持人工复核和报告导出。

## 核心功能

- **多来源项目导入**：支持本地文件、ZIP 压缩包与公开 GitHub 仓库；主要分析 Python 脚本、Jupyter Notebook、CSV 数据及相关项目说明。
- **代码与 Notebook 静态分析**：基于 Python AST 和 Notebook 解析识别部分语法问题、数据泄漏线索、建模评价风险、可复现性及代码质量问题。
- **数据质量分析**：检查缺失值、重复记录、字段类型、异常分布和相关性等，并区分统计观察与需要处理的潜在问题。
- **跨文件关联与流程识别**：结合可识别的数据读写路径、变量传递和建模操作，整理代码与数据之间的关系；对无法确认的关联保留不确定性。
- **结构化证据与 AI Review**：以统一的 Finding 记录问题位置、证据、严重程度和验证状态；从本地 Markdown 知识准则中检索相关内容，通过千问模型生成辅助解释，并对模型输出进行校验。不符合条件的解释由确定性摘要补充。
- **修复建议与报告导出**：对支持的问题提供修复方向，对候选修改开展静态校验；满足条件时支持 Patch 导出，并可生成 Markdown、中文 PDF 审查报告。

## 审查流程

```text
项目导入 → 文件解析 → 代码与数据检查 → 跨文件关联
        → Finding 证据整理 → 知识增强与 AI Review
        → 解释校验与修复建议 → 报告及可用 Patch 导出
```

## 环境与启动

项目基于 Python 开发，使用 Streamlit 提供交互界面。已有验证记录的 Python 环境版本为 3.12；具体依赖范围见 `requirements.txt`。

```powershell
# 克隆正式分支
git clone -b master https://github.com/yaoy71627-alt/CodeData-Copilot.git
cd CodeData-Copilot

# 安装基础运行依赖
python -m pip install -r requirements.txt

# 启动系统
python -m streamlit run app.py
```

Windows 用户也可使用仓库中的 `run_app.bat` 启动。若使用 GitHub 仓库导入功能，需预先安装 Git，并能够访问目标公开仓库。

### 可选：配置 AI Review

系统通过阿里云百炼兼容接口调用千问模型。需要使用 AI Review 时，在项目根目录创建 `.env` 文件，并配置自己的 API Key：

```dotenv
DASHSCOPE_API_KEY=your_api_key_here
```

仓库提供 `.env.example` 作为示例。请勿将真实密钥提交至 GitHub；不使用模型接口时，基础静态审查功能仍可独立使用，增强解释能力则受到限制。

## 测试与验证

项目使用四类案例开展阶段性验证：

| 测试案例 | 主要验证目的 |
| --- | --- |
| Titanic 人工注入问题案例 | 已知问题的识别与定位 |
| Cardekho 误报控制案例 | 合法数据现象与实际风险的区分 |
| Telco 开源项目 | 多文件项目审查与建模流程分析 |
| IBM Employee Attrition 开源项目 | Notebook、接口兼容性及模型评价风险审查 |

仓库内 `validation/validation_manifest.json` 记录的一轮自动化测试结果为：**538 项测试通过、17 个子测试通过、1 项跳过、0 项失败**。跳过项为未启用的在线付费模型端到端测试。该结果反映对应测试范围内的通过情况，**不等同于系统在任意机器学习项目中的总体检测准确率**。

测试代码及部分测试资源位于 `tests/`；运行完整测试可能还需安装测试与案例依赖，相关环境版本可参考 `validation/validation_manifest.json`。

## 主要目录

```text
app.py                       Streamlit 应用入口
code_analyzer.py             代码分析
data_analyzer.py             数据质量分析
data_flow_analyzer.py        数据流程分析
consistency_validator.py     解释及结果一致性校验
repair_validator.py          修复建议校验
report.py                    报告生成
knowledge/                   本地知识准则
assets/fonts/                中文 PDF 所需字体及许可说明
tests/                       自动化测试与部分案例资源
validation/                  测试记录及验证清单
requirements.txt             基础运行依赖
run_app.bat                   Windows 启动脚本
```

## 适用范围与说明

系统以静态分析和可获得的数据证据为基础，不以执行待审查项目为常规前提；对于动态控制流、外部服务、复杂业务规则等情况，可能出现漏检或误报。AI 解释与静态修复校验均不能替代人工复核、实际运行测试或模型效果评估。

项目使用的第三方代码、测试资源和字体应分别遵守其适用的许可条件。正式进行代码复用与再分发时，请核对仓库中的项目许可证及相关第三方许可文件。
