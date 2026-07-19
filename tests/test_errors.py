from career_agent.errors import friendly_error_message


class ProviderError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_friendly_error_message_for_insufficient_balance():
    assert friendly_error_message(ProviderError(402)) == (
        "模型 API 余额不足，请检查服务商控制台的 Billing 页面。"
    )


def test_friendly_error_message_for_invalid_api_key():
    assert friendly_error_message(ProviderError(401)) == (
        "模型 API 认证失败，请检查 OPENAI_API_KEY 和 OPENAI_BASE_URL。"
    )


def test_friendly_error_message_for_network_failure():
    assert friendly_error_message(ConnectionError()) == (
        "无法连接模型服务，请检查网络和 OPENAI_BASE_URL。"
    )
