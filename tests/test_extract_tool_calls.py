"""extract_tool_calls handles both message shapes the wild emits."""

from __future__ import annotations

from tracewright import Message, ToolCall, extract_tool_calls, tool_call_divergence


def _msg(role: str, content: str = "", **extra: object) -> Message:
    return Message(role=role, content=content, **extra)  # type: ignore[arg-type]


def test_openai_nested_function_shape() -> None:
    messages = [
        _msg("user", "find rust posts"),
        _msg(
            "assistant",
            "",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "rust"}'},
                }
            ],
        ),
    ]
    calls = extract_tool_calls(messages)
    assert calls == [ToolCall(name="search", arguments={"q": "rust"})]


def test_flat_shape_with_dict_arguments() -> None:
    messages = [
        _msg(
            "assistant",
            "",
            tool_calls=[{"name": "fetch", "arguments": {"url": "https://example.com"}}],
        )
    ]
    calls = extract_tool_calls(messages)
    assert calls == [ToolCall(name="fetch", arguments={"url": "https://example.com"})]


def test_arguments_as_json_string_parsed() -> None:
    messages = [
        _msg(
            "assistant",
            "",
            tool_calls=[{"name": "fetch", "arguments": '{"url": "x"}'}],
        )
    ]
    calls = extract_tool_calls(messages)
    assert calls == [ToolCall(name="fetch", arguments={"url": "x"})]


def test_malformed_json_arguments_yields_empty_dict() -> None:
    # We'd rather lose arg-fidelity than drop the call - the divergence
    # score still captures the name match.
    messages = [
        _msg("assistant", "", tool_calls=[{"name": "fetch", "arguments": "not-json"}]),
    ]
    calls = extract_tool_calls(messages)
    assert calls == [ToolCall(name="fetch", arguments={})]


def test_multiple_messages_preserves_order() -> None:
    messages = [
        _msg(
            "assistant",
            "",
            tool_calls=[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}],
        ),
        _msg("tool", "result", name="a"),
        _msg(
            "assistant",
            "",
            tool_calls=[{"name": "c", "arguments": {}}],
        ),
    ]
    calls = extract_tool_calls(messages)
    assert [c.name for c in calls] == ["a", "b", "c"]


def test_skips_non_assistant_messages() -> None:
    messages = [
        _msg("user", "hi"),
        _msg("system", "you are an agent"),
        _msg("tool", "result", name="search", tool_calls=[{"name": "ignored", "arguments": {}}]),
    ]
    calls = extract_tool_calls(messages)
    assert calls == []


def test_empty_or_none_input() -> None:
    assert extract_tool_calls(None) == []
    assert extract_tool_calls([]) == []
    assert extract_tool_calls([_msg("assistant", "no calls here")]) == []


def test_round_trip_with_divergence_scorer() -> None:
    # Realistic flow: extract baseline calls from a recorded trace,
    # extract candidate calls from a replay, score divergence.
    baseline_messages = [
        _msg(
            "assistant",
            "",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "rust"}'},
                }
            ],
        )
    ]
    candidate_messages = [
        _msg(
            "assistant",
            "",
            tool_calls=[{"name": "search", "arguments": {"q": "rust"}}],
        )
    ]
    baseline = extract_tool_calls(baseline_messages)
    candidate = extract_tool_calls(candidate_messages)
    result = tool_call_divergence(baseline, candidate)
    assert result.passed
    assert result.score == 1.0
