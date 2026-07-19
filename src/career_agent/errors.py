"""Convert model-provider errors into safe, actionable user messages."""
from __future__ import annotations


def friendly_error_message(error: Exception) -> str:
    """Return a user-facing message without exposing provider internals."""
    status_code = getattr(error, "status_code", None)
    if status_code is None and (response := getattr(error, "response", None)):
        status_code = getattr(response, "status_code", None)

    if status_code in {401, 403}:
        return "模型 API 认证失败，请检查 OPENAI_API_KEY 和 OPENAI_BASE_URL。"
    if status_code == 402:
        return "模型 API 余额不足，请检查服务商控制台的 Billing 页面。"
    if status_code == 429:
        return "模型 API 请求过于频繁，请稍后重试。"
    if isinstance(error, TimeoutError):
        return "连接模型服务超时，请检查网络后重试。"
    if isinstance(error, ConnectionError):
        return "无法连接模型服务，请检查网络和 OPENAI_BASE_URL。"

    message = str(error).lower()
    if "insufficient balance" in message:
        return "模型 API 余额不足，请检查服务商控制台的 Billing 页面。"
    if "api key" in message or "authentication" in message:
        return "模型 API 认证失败，请检查 OPENAI_API_KEY 和 OPENAI_BASE_URL。"
    if "timeout" in message:
        return "连接模型服务超时，请检查网络后重试。"
    if "connection" in message or "network" in message:
        return "无法连接模型服务，请检查网络和 OPENAI_BASE_URL。"

    return "模型服务暂时不可用，请稍后重试；如持续发生，请查看终端日志。"
