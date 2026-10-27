"""Mock-only Qwen configuration, compact batches, schema and failure semantics."""
import copy
import json
import socket
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import agent
import qwen_runtime as qr
from consistency_validator import build_analysis_facts, validate_agent_consistency


@pytest.fixture
def facts():
    code = [{"file": "model.py", "issues": [
        {"rule_id": f"CODE_RISK_{i}", "type": "Static Review Risk", "line": i + 1,
         "severity": "High" if i == 2 else "Low", "confidence": "Medium",
         "evidence": "模型输入范围需要复核。", "source_snippet": "model.fit(X_train, y_train)"}
        for i in range(3)]}]
    return build_analysis_facts(code, [], {"code_units": [{"code": "UNNEEDED_SOURCE" * 20000}]})


def prompt_facts(kwargs):
    raw = kwargs["messages"][1]["content"].split("结构化 Finding 与 analysis_facts（唯一事实依据）：\n")[1]
    return json.JSONDecoder().raw_decode(raw)[0]


def annotation(fid):
    return {"finding_id": fid, "cause": "请核对建模输入与预处理的执行路径。",
            "potential_impact": "实际影响取决于执行路径，需要运行对照确认。",
            "recommendation": "按原始证据检查输入边界并保留复核记录。"}


def response(payload, request_id="req-safe-001"):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return NS(choices=[NS(message=NS(content=text), finish_reason="stop")],
              usage=NS(prompt_tokens=300, completion_tokens=100, total_tokens=400), _request_id=request_id)


def successful(**kwargs):
    return response({"explanations": [annotation(f["id"]) for f in prompt_facts(kwargs)["findings"]]})


def run(facts, create, previous_review=None):
    code = [{"file": f["file"], "issues": [f]} for f in facts["formal_code_issues"]]
    client = NS(chat=NS(completions=NS(create=create)))
    with patch.object(agent, "get_client", return_value=client):
        return agent.run_agent(code, [], {"analysis_facts": facts}, knowledge_context="", analysis_facts=facts, previous_review=previous_review)


def test_compact_request_preserves_ids_and_omits_repeated_project_material(facts):
    original = copy.deepcopy(facts)
    create = Mock(side_effect=successful)
    review = run(facts, create)
    create.assert_called_once()
    prompt = create.call_args.kwargs["messages"][1]["content"]
    assert "UNNEEDED_SOURCE" not in prompt and '"data_flow"' not in prompt
    assert len(prompt) < 6000
    assert review.ai_status["status"] == "success"
    for f in facts["findings"]:
        assert prompt.count(f["id"]) == 1
    batch = review.ai_status["batches"][0]
    assert batch["input_characters"] == sum(len(m["content"]) for m in create.call_args.kwargs["messages"])
    assert batch["usage"]["total_tokens"] == 400 and batch["request_id"] == "req-safe-001"
    assert "UNNEEDED_SOURCE" not in json.dumps(review.ai_status)
    assert facts == original
    validated, _ = validate_agent_consistency(review, facts)
    assert validated.ai_status == review.ai_status


@pytest.mark.parametrize("payload", ["bad json", "", [], {}, {"explanations": "wrong"},
    {"explanations": []}, {"explanations": [annotation("F-unknown")]},
    {"explanations": [{"finding_id": "F-unknown", "cause": 5}]}])
def test_invalid_model_content_never_claims_success(facts, payload):
    create = Mock(return_value=response(payload))
    review = run(facts, create)
    assert review.ai_status["status"] == "invalid_ai_response"
    assert review.coverage["ai_explained"] == 0
    assert "完整静态诊断" in review
    assert all(f["id"] in review for f in facts["findings"])
    assert create.call_count == 3
    assert review.ai_status["batches"][0]["retry_count"] == 2
    assert review.ai_status["batches"][0]["failure_reasons"]


def test_partial_schema_and_fact_rejected_explanations_have_honest_status(facts):
    fid = facts["findings"][0]["id"]
    partial = run(facts, Mock(return_value=response({"explanations": [annotation(fid)]})))
    assert partial.ai_status["status"] == "partial_success"
    assert partial.ai_status["accepted_findings"] == 1
    invented = annotation(fid)
    invented["potential_impact"] = "模型准确率显著下降。"
    invalid = run(facts, Mock(return_value=response({"explanations": [invented]})))
    assert invalid.ai_status["status"] == "invalid_ai_response"
    assert "模型准确率显著下降" not in invalid


def test_batches_prioritize_important_findings_and_keep_partial_results(facts, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_BATCH_SIZE", "1")
    create = Mock(side_effect=lambda **kw: successful(**kw) if create.call_count == 1 else (_ for _ in ()).throw(TimeoutError("private payload")))
    review = run(facts, create)
    assert create.call_count == 4
    assert prompt_facts(create.call_args_list[0].kwargs)["findings"][0]["severity"] == "High"
    assert review.ai_status["status"] == "partial_success"
    assert review.ai_status["accepted_findings"] == 1
    assert review.ai_status["errors"][0]["status"] == "timeout"
    assert review.ai_status["batches"][1]["retry_count"] == 2
    assert "private payload" not in json.dumps(review.ai_status)
    assert all(f["id"] in review for f in facts["findings"])


@pytest.mark.parametrize("budget", ["batch", "tokens", "time"])
def test_job_budgets_stop_without_retry(facts, monkeypatch, budget):
    create = Mock(side_effect=successful)
    if budget == "batch":
        monkeypatch.setenv("DASHSCOPE_BATCH_SIZE", "1")
        monkeypatch.setenv("DASHSCOPE_MAX_BATCHES", "1")
    elif budget == "tokens":
        monkeypatch.setenv("DASHSCOPE_TOKEN_BUDGET", "1000")
    else:
        monkeypatch.setattr(agent.time, "monotonic", Mock(side_effect=[0, 121, 122]))
    review = run(facts, create)
    assert create.call_count == (1 if budget == "batch" else 0)
    assert review.ai_status["errors"][0]["status"] == "budget_exceeded"
    assert review.ai_status["status"] == ("partial_success" if budget == "batch" else "budget_exceeded")
    assert all(f["id"] in review for f in facts["findings"])


def test_streaming_json_and_usage_are_assembled_and_closed(facts, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_STREAM", "true")
    closed = Mock()
    class Stream:
        close = closed
        def __iter__(self):
            payload = json.dumps({"explanations": [annotation(f["id"]) for f in facts["findings"]]}, ensure_ascii=False)
            for text in (payload[:31], payload[31:]):
                yield NS(choices=[NS(delta=NS(content=text), finish_reason=None)])
            yield NS(choices=[], usage=NS(prompt_tokens=50, completion_tokens=90, total_tokens=140), request_id="req-stream")
    create = Mock(return_value=Stream())
    review = run(facts, create)
    assert review.ai_status["status"] == "success"
    assert review.ai_status["batches"][0]["usage"]["total_tokens"] == 140
    assert create.call_args.kwargs["stream"] and create.call_args.kwargs["stream_options"]["include_usage"]
    closed.assert_called_once()


@pytest.mark.parametrize("status,code,expected", [
    (401, "InvalidApiKey", "authentication_failed"), (429, "Throttling", "rate_limited"),
    (400, "InvalidParameter", "request_format_error"), (422, "BadFormat", "request_format_error"),
    (403, "AccessDenied", "permission_or_model"), (404, "ModelNotFound", "permission_or_model"),
    (402, "Any", "balance_or_quota"), (403, "Arrearage", "balance_or_quota"),
    (500, "Any", "api_failed"),
])
def test_service_error_classification_has_safe_diagnostics(status, code, expected):
    exc = RuntimeError("sk-never-log-this-private-payload")
    exc.status_code, exc.body = status, {"error": {"code": code, "message": "private payload"}}
    exc.response = NS(headers={"x-request-id": "req-diagnostic"})
    result = agent.classify_ai_error(exc)
    assert result["status"] == expected and result["http_status"] == status
    assert result["request_id"] == "req-diagnostic"
    assert "private" not in json.dumps(result)


def test_dns_network_and_config_are_distinct():
    dns = ConnectionError("private")
    dns.__cause__ = socket.gaierror("private")
    assert agent.classify_ai_error(dns)["status"] == "dns_failed"
    assert agent.classify_ai_error(ConnectionError("private"))["status"] == "network_failed"
    assert agent.classify_ai_error(qr.QwenConfigError("private"))["status"] == "configuration_error"


def test_regional_settings_and_client_arguments(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("DASHSCOPE_KEY_REGION", "ap-southeast-1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-plus")
    monkeypatch.setenv("DASHSCOPE_TIMEOUT", "75")
    settings = qr.load_settings()
    assert settings.region == "singapore" and settings.key_region_checked
    with patch("openai.OpenAI") as constructor:
        agent.get_client()
    assert constructor.call_args.kwargs["base_url"] == settings.base_url
    assert constructor.call_args.kwargs["timeout"] == 75
    assert constructor.call_args.kwargs["max_retries"] == 0
    monkeypatch.setenv("DASHSCOPE_KEY_REGION", "beijing")
    with pytest.raises(qr.QwenConfigError):
        qr.load_settings()


@pytest.mark.parametrize("key,value", [
    ("DASHSCOPE_BASE_URL", "https://evil.example/compatible-mode/v1"),
    ("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com:invalid/compatible-mode/v1"),
    ("DASHSCOPE_TIMEOUT", "nan"), ("DASHSCOPE_STREAM", "maybe"),
    ("DASHSCOPE_MAX_BATCHES", "100"), ("DASHSCOPE_BATCH_SIZE", "1.5"),
])
def test_invalid_configuration_fails_before_network(key, value, monkeypatch):
    monkeypatch.setenv(key, value)
    with pytest.raises(qr.QwenConfigError):
        qr.load_settings()


def test_dictionary_headers_credentials_never_enter_prompt(facts):
    finding = copy.deepcopy(facts["findings"][0])
    finding["evidence"] = '''headers={"Authorization":"Bearer hiddenCredential123"}; config={'api_key':'dictionaryKey987'}; x='sk-longSecretPlaceholder987' '''
    text = qr.make_prompt([finding], facts, "", {})
    assert all(secret not in text for secret in ("hiddenCredential123", "dictionaryKey987", "sk-longSecretPlaceholder987"))
    assert "[REDACTED]" in text


def test_connectivity_script_requires_explicit_flag(capsys):
    from scripts.check_qwen_connection import main
    with patch.object(agent, "get_client") as client:
        assert main([]) == 2
    client.assert_not_called()
    assert "未调用 API" in capsys.readouterr().out


def test_connectivity_preflight_is_tiny_and_logs_no_secret():
    from scripts.check_qwen_connection import check_connection
    create = Mock(return_value=response("OK"))
    with patch.object(agent, "get_client", return_value=NS(chat=NS(completions=NS(create=create)))):
        result = check_connection()
    assert result["status"] == "success"
    assert create.call_args.kwargs["max_tokens"] == 16
    assert len(create.call_args.kwargs["messages"][0]["content"]) < 20
    assert "OK" not in json.dumps(result)  # Response text itself is not logged.


def test_partial_review_can_resume_only_missing_findings(facts, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_BATCH_SIZE", "1")
    first = Mock(side_effect=lambda **kw: successful(**kw) if first.call_count == 1 else (_ for _ in ()).throw(TimeoutError()))
    previous = run(facts, first)
    assert previous.ai_status["status"] == "partial_success"
    second = Mock(side_effect=successful)
    completed = run(facts, second, previous_review=previous)
    assert second.call_count == 2
    assert completed.ai_status["status"] == "success"
    assert completed.ai_status["reused_findings"] == 1
    reused = set(previous.coverage["accepted_annotations"])
    resent = {f["id"] for call in second.call_args_list for f in prompt_facts(call.kwargs)["findings"]}
    assert not reused & resent


def test_changed_facts_invalidate_partial_review_cache(facts):
    fid = facts["findings"][0]["id"]
    previous = run(facts, Mock(return_value=response({"explanations": [annotation(fid)]})))
    changed = copy.deepcopy(facts)
    changed["model_features"] = ["different_input"]
    create = Mock(side_effect=successful)
    result = run(changed, create, previous_review=previous)
    assert result.ai_status["reused_findings"] == 0
    assert {f["id"] for f in prompt_facts(create.call_args.kwargs)["findings"]} == {f["id"] for f in facts["findings"]}


def test_knowledge_excerpt_preserves_whole_rules():
    knowledge = "### Source\n- First complete rule\n- " + "lengthy" * 1000
    excerpt = qr.compact_knowledge(knowledge, 100)
    assert excerpt == "### Source\n- First complete rule"
    assert "lengthy" not in excerpt


@pytest.mark.parametrize("kind", ["bad_json", "valid", "transport_failure"])
def test_each_attempt_reserves_budget_exactly_once(facts, monkeypatch, kind):
    monkeypatch.setenv("DASHSCOPE_BATCH_SIZE", "1")
    def create(**kwargs):
        if kind == "transport_failure":
            raise TimeoutError("private transport payload")
        if kind == "bad_json":
            return response("not JSON")
        return successful(**kwargs)
    review = run(facts, Mock(side_effect=create))
    batches = review.ai_status["batches"]
    expected = sum(max(b["input_characters"] + b["max_output_tokens"],
                       b.get("usage", {}).get("total_tokens", 0)) for b in batches)
    assert review.ai_status["budget_reserved_tokens"] == expected
    assert len(batches) == (1 if kind == "transport_failure" else 3)
    assert review.ai_status["status"] == {"bad_json": "invalid_ai_response",
        "valid": "success", "transport_failure": "timeout"}[kind]
