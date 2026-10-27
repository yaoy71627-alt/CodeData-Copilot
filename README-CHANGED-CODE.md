# 累计新增及修改代码文件

基准：CodeData-Copilot-master (2).zip；最终：CodeData-Copilot-Batch-D.zip。本包包括 A—D 的累计变更，保留原相对路径。

共 44 个代码文件：修改 12 个，新增 32 个。未变化代码不重复输出。

## 使用说明

本包是代码提取包，不是可独立运行的完整项目。没有包含字体、知识库、数据集及全部配套配置；需要完整运行时请使用 CodeData-Copilot-Batch-D.zip。不要删除原项目中未列出的文件。
.env、.git、历史报告和验证产物未纳入对比或导出；不会覆盖本地密钥。

tests/fixtures/ 下的源代码为固定第三方案例快照，不是对其上游仓库的修改；已附许可与来源文件。未执行这些案例代码。

本次仅比较、提取与校验，未再次修改源代码，也未重新运行功能测试。逐文件状态与前后哈希见 CODE_CHANGE_MANIFEST.json。

## 文件清单

| 文件 | 状态 | 类型 |
| --- | --- | --- |
| `agent.py` | 修改 | application_or_tool |
| `app.py` | 修改 | application_or_tool |
| `code_analyzer.py` | 修改 | application_or_tool |
| `consistency_validator.py` | 修改 | application_or_tool |
| `data_flow_analyzer.py` | 修改 | application_or_tool |
| `file_loader.py` | 修改 | application_or_tool |
| `github_loader.py` | 修改 | application_or_tool |
| `project_archive.py` | 新增 | application_or_tool |
| `project_lineage.py` | 新增 | application_or_tool |
| `project_overview.py` | 修改 | application_or_tool |
| `project_score.py` | 修改 | application_or_tool |
| `qwen_runtime.py` | 新增 | application_or_tool |
| `regression_benchmark.py` | 新增 | application_or_tool |
| `repair_validator.py` | 修改 | application_or_tool |
| `report.py` | 修改 | application_or_tool |
| `scripts/check_qwen_connection.py` | 新增 | application_or_tool |
| `scripts/render_regression_pdfs.py` | 新增 | application_or_tool |
| `scripts/run_batch_d.py` | 新增 | application_or_tool |
| `static_review_rules.py` | 新增 | application_or_tool |
| `tests/conftest.py` | 新增 | test_code |
| `tests/fixtures/batch_b/ibm/employee-attrition.ipynb` | 新增 | reference_fixture |
| `tests/fixtures/batch_b/telco/src/analysis.py` | 新增 | reference_fixture |
| `tests/fixtures/batch_b/telco/src/data_download.py` | 新增 | reference_fixture |
| `tests/fixtures/batch_b/telco/src/eda_script.py` | 新增 | reference_fixture |
| `tests/fixtures/batch_b/telco/src/pre_process_script.py` | 新增 | reference_fixture |
| `tests/fixtures/batch_c/telco/Makefile` | 新增 | reference_fixture |
| `tests/test_batch_b_cases.py` | 新增 | test_code |
| `tests/test_batch_b_rules.py` | 新增 | test_code |
| `tests/test_batch_c_archives.py` | 新增 | test_code |
| `tests/test_batch_c_integration.py` | 新增 | test_code |
| `tests/test_batch_c_lineage.py` | 新增 | test_code |
| `tests/test_batch_d_acceptance.py` | 新增 | test_code |
| `tests/test_batch_d_runner.py` | 新增 | test_code |
| `tests/test_final_report_consistency.py` | 新增 | test_code |
| `tests/test_full_regression_acceptance.py` | 新增 | test_code |
| `tests/test_knowledge_resilience.py` | 新增 | test_code |
| `tests/test_pdf_smoke.py` | 新增 | test_code |
| `tests/test_qwen_batch_a.py` | 新增 | test_code |
| `tests/test_qwen_online.py` | 新增 | test_code |
| `tests/test_regression_benchmark.py` | 新增 | test_code |
| `tests/test_repair_reliability_p1.py` | 新增 | test_code |
| `tests/test_review_accuracy.py` | 新增 | test_code |
| `tests/test_review_coverage.py` | 新增 | test_code |
| `tests/test_titanic_flow_p0.py` | 修改 | test_code |
