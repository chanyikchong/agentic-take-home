import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.exceptions import ModelHTTPError

from solutions.pipelines import run_agent, _is_rate_limit_error, AgentResult


class TestIsRateLimitError:
    def test_model_http_error_429_is_rate_limit(self):
        err = ModelHTTPError(status_code=429, model_name="test-model")
        assert _is_rate_limit_error(err) is True

    def test_model_http_error_500_is_not_rate_limit(self):
        err = ModelHTTPError(status_code=500, model_name="test-model")
        assert _is_rate_limit_error(err) is False

    def test_model_http_error_400_is_not_rate_limit(self):
        err = ModelHTTPError(status_code=400, model_name="test-model")
        assert _is_rate_limit_error(err) is False

    def test_generic_exception_is_not_rate_limit(self):
        err = RuntimeError("something broke")
        assert _is_rate_limit_error(err) is False

    def test_value_error_is_not_rate_limit(self):
        err = ValueError("429 in message but not a rate limit")
        assert _is_rate_limit_error(err) is False


def _make_mock_agent(side_effect=None, return_value=None):
    """Create a mock agent that simulates pydantic-ai Agent behavior."""
    agent = MagicMock()
    agent._prepend_prompt = None
    agent._model_key = "test-model"
    agent._output_type = None
    agent._parse_retries = 1

    mock_result = MagicMock()
    mock_result.output = "test output"
    mock_result.usage.return_value = MagicMock()

    if side_effect is not None:
        agent.run = AsyncMock(side_effect=side_effect)
    elif return_value is not None:
        agent.run = AsyncMock(return_value=return_value)
    else:
        agent.run = AsyncMock(return_value=mock_result)

    return agent, mock_result


class TestSuccessfulCall:
    @pytest.mark.asyncio
    async def test_success_returns_immediately(self):
        agent, mock_result = _make_mock_agent()

        result = await run_agent(agent, "test prompt")

        assert isinstance(result, AgentResult)
        assert result.output == "test output"
        assert agent.run.call_count == 1


class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self):
        agent, mock_result = _make_mock_agent()
        rate_limit_error = ModelHTTPError(status_code=429, model_name="test-model")

        # Fail twice with 429, succeed on third
        agent.run = AsyncMock(side_effect=[
            rate_limit_error,
            rate_limit_error,
            mock_result,
        ])

        with patch("solutions.pipelines.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await run_agent(agent, "test prompt", base_delay=0.01, max_delay=0.05)

        assert isinstance(result, AgentResult)
        assert result.output == "test output"
        assert agent.run.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_backoff_increases_between_retries(self):
        agent, mock_result = _make_mock_agent()
        rate_limit_error = ModelHTTPError(status_code=429, model_name="test-model")

        agent.run = AsyncMock(side_effect=[
            rate_limit_error,
            rate_limit_error,
            mock_result,
        ])

        with patch("solutions.pipelines.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch("solutions.pipelines.random.uniform", return_value=0.0):
                result = await run_agent(agent, "test prompt", base_delay=1.0, max_delay=15.0)

        # First delay: min(1.0 * 2^0, 15) + 0 = 1.0
        # Second delay: min(1.0 * 2^1, 15) + 0 = 2.0
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays[0] < delays[1]


class TestRateLimitExhausted:
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        agent, _ = _make_mock_agent()
        rate_limit_error = ModelHTTPError(status_code=429, model_name="test-model")

        agent.run = AsyncMock(side_effect=rate_limit_error)

        with patch("solutions.pipelines.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ModelHTTPError) as exc_info:
                await run_agent(agent, "test prompt", max_http_retries=3, base_delay=0.01)

        assert exc_info.value.status_code == 429
        assert agent.run.call_count == 3


class TestNonRateLimitErrors:
    @pytest.mark.asyncio
    async def test_500_error_raises_immediately(self):
        agent, _ = _make_mock_agent()
        server_error = ModelHTTPError(status_code=500, model_name="test-model")
        agent.run = AsyncMock(side_effect=server_error)

        with pytest.raises(ModelHTTPError) as exc_info:
            await run_agent(agent, "test prompt")

        assert exc_info.value.status_code == 500
        assert agent.run.call_count == 1

    @pytest.mark.asyncio
    async def test_runtime_error_raises_immediately(self):
        agent, _ = _make_mock_agent()
        agent.run = AsyncMock(side_effect=RuntimeError("unexpected"))

        with pytest.raises(RuntimeError):
            await run_agent(agent, "test prompt")

        assert agent.run.call_count == 1


class TestJsonParseWithinRetryLoop:
    @pytest.mark.asyncio
    async def test_json_parse_retries_work(self):
        agent, _ = _make_mock_agent()
        agent._output_type = MagicMock()  # triggers manual JSON parsing
        agent._parse_retries = 3

        # All calls return invalid JSON strings
        bad_result = MagicMock()
        bad_result.output = "not json"
        bad_result.usage.return_value = MagicMock()
        agent.run = AsyncMock(return_value=bad_result)

        with patch("solutions.pipelines._parse_json_output", side_effect=ValueError("bad json")):
            with pytest.raises(ValueError, match="bad json"):
                await run_agent(agent, "test prompt")

        # Should have made parse_retries calls (no 429, so only inner loop)
        assert agent.run.call_count == 3


class TestRateLimitDuringJsonParse:
    @pytest.mark.asyncio
    async def test_429_during_parse_loop_retries_outer(self):
        agent, mock_result = _make_mock_agent()
        agent._output_type = None  # no manual parsing needed
        rate_limit_error = ModelHTTPError(status_code=429, model_name="test-model")

        # First call: 429, second call: success
        agent.run = AsyncMock(side_effect=[rate_limit_error, mock_result])

        with patch("solutions.pipelines.asyncio.sleep", new_callable=AsyncMock):
            result = await run_agent(agent, "test prompt", base_delay=0.01)

        assert isinstance(result, AgentResult)
        assert agent.run.call_count == 2
