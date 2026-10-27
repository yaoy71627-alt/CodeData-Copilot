"""Explicit, tiny online preflight; never prints keys, response bodies or prompts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_connection():
    from agent import get_client, classify_ai_error
    from qwen_runtime import load_settings, request_id_from
    try:
        settings = load_settings()
        response = get_client().chat.completions.create(
            model=settings.model, messages=[{"role": "user", "content": "Reply OK."}],
            max_tokens=16, temperature=0, stream=False,
            timeout=min(settings.timeout, 30), extra_body={"enable_thinking": False})
        choices = getattr(response, "choices", [])
        if not choices or not choices[0].message.content:
            return {"status": "invalid_ai_response", "reason": "连通性请求没有返回文本。"}
        return {"status": "success", "reason": "短消息连通性通过；这不代表完整项目审查通过。",
                "region": settings.region, "key_region_checked": settings.key_region_checked,
                "request_id": request_id_from(response)}
    except Exception as exc:
        return classify_ai_error(exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="Allow one small, potentially billable request")
    args = parser.parse_args(argv)
    if not args.online:
        print("未调用 API。请在本地配置同地域环境后，显式添加 --online 执行短消息测试。")
        return 2
    result = check_connection()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
