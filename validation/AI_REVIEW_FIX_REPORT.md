# AI Review 与页面展示修复记录

## 1. 修复内容

- 在结构化解释入口核对问题类别、字段、操作、位置及已重建事实。拒绝明确错配的解释，按当前 Finding 补齐；不把普通谨慎解释一律拒绝。
- 对已知 dropna 删除量、划分行数提供限定到文件和操作位置的简短上下文。数量、比例及用途不符的陈述被剔除；不能将其他问题或其他文件的已知数字借用为依据。
- 缓存复用与最终报告再次校验已接受的解释；不可信旧正文不能绕过重新渲染。拒绝原因记录 Finding ID，不记录被拒绝的敏感原文。
- 首页同一次 AI 增强事件统一为一条提示，细节折叠展示；部分成功不再写入完全失败状态。成功刷新不会继续展示旧的 AI 警告，真实文件分析或 PDF 故障仍保留。
- 展示编号为稳定的 P01 / F01 序号；完整 ID 保留在内部报告、关联、去重及机器数据中。页面、Markdown、PDF 使用同一份映射，支持一个问题关联多份证据。
- 页面和导出通过 outcome 使用 success / partial_success / failed（未启用或无需调用单列）。底层 status、failure_code、错误和请求诊断继续区分超时、鉴权、格式或内容验证故障。只有完整且经校验的 AI 注释计入 AI 已解释数量。
- 未改动分析器、知识库、真实案例、GitHub 导入和修复/Patch 模块；最终补丁仍经过现有修复验证及重新生成流程。

## 2. 修改源码与配置

agent.py、app.py、consistency_validator.py、qwen_runtime.py、report.py。

新增 review_semantics.py、ui_review_helpers.py；补回缺失的 pytest.ini，仅注册既有 e2e 标记。

## 3. 测试变更

新增 tests/test_review_binding_ui.py，覆盖错配、正常解释、量化事实、位置、缓存旧文本、去重、统计、短编号、一对多证据、真实 PDF 文本和 Streamlit 首页 rerun。

更新 tests/test_review_accuracy.py、tests/test_final_report_consistency.py、tests/test_full_regression_acceptance.py：仅将展示层的旧完整 ID 断言更新为对应的短编号，内部 ID 仍按完整 ID 校验。

上传包原先缺少 6 个既有测试资源，导致全量收集失败。已从前次受控快照逐字节恢复：batch_d/truth.json、IBM CSV、Telco README、Telco raw/train/test CSV。现有同名夹具逐一比对一致，未覆盖差异内容；原始用户案例未修改。恢复文件及 SHA256 见 validation_manifest.json。

## 4. 实际最终验证

- 针对性：149 passed。
- 完整套件：538 passed，17 subtests passed，0 failed，1 skipped，0 warnings。
- 唯一跳过项：tests/test_qwen_online.py，需要显式 --online 与进程环境中的 Key。本次没有进行收费 API 调用。
- 已配置原始 Titanic 问题案例和 Cardekho 正常案例，并运行 IBM、Telco 受控快照回归。未跳过原始两案例回归。
- 四案例 Markdown、PDF 和五个 Streamlit 页面均通过现有验收；四案例 PDF 为 18 / 11 / 19 / 24 页。抽查了 Titanic 首页与解释页、Cardekho 解释页的实际渲染，中文和短编号可读。
- 四案例既有受控真值子集：Titanic TP=10/FP=0/FN=0；Cardekho TP=2/FP=0/FN=0；IBM TP=2/FP=0/FN=0；Telco TP=7/FP=0/FN=0。这是既定标注子集，不是整个项目的识别准确率。
- 初始失败已保留于开发过程记录：先修复新增导入顺序及建议/状态元数据保留问题；更新有意改变展示行为对应的旧断言；恢复缺失夹具。最终通过来自新的完整进程，未删除或跳过失败测试。

复现：安装项目依赖及 pytest、pypdf 等已有测试依赖；为原始案例指定 CODEDATA_TITANIC_NOTEBOOK、CODEDATA_TITANIC_CSV、CODEDATA_CARDEKHO_NOTEBOOK、CODEDATA_CARDEKHO_CSV，然后在项目根目录运行 `python -m pytest -q -ra`。原始两案例未打包进项目，请使用自己的原始文件，哈希受既有真值约束。不要将 `.env` 或密钥加入报告或归档。

## 5. 影响与兼容

知识检索、有限批次 Qwen、预算限制及异常降级继续保留。保留先前未提交改动，没有执行提交或推送。没有更改原始完整 ID，也没有改动分析器和 Patch 验证模块。已有报告缓存通过策略版本重新投影，避免继续使用旧编号或旧解释。

## 6. 限制

语义校验是基于问题类别、字段、位置和限定事实的静态约束，不能证明任意自然语言解释在业务上完全正确。未运行真实千问在线调用；Streamlit 测试使用 AppTest，不能代替部署环境下的人工浏览器验收。修复建议与 Patch 不因本轮文字和显示修复获得新的运行验证资格。

测试日志、JUnit 和运行版本/文件哈希记录随包提供。

最新输入确认：CodeData-Copilot(1)(2).zip 与前一上传包的全部非敏感源码和资源逐文件 SHA256 一致，Git 元数据也一致；本报告最终测试结果适用于该最新包。
