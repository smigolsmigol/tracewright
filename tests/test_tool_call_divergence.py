"""Tool-call divergence scorer tests."""

from __future__ import annotations

from tracewright import ToolCall, tool_call_divergence


def test_exact_replay_scores_one() -> None:
    baseline = [
        ToolCall(name="search", arguments={"query": "rust"}),
        ToolCall(name="fetch", arguments={"url": "https://example.com"}),
    ]
    candidate = [
        ToolCall(name="search", arguments={"query": "rust"}),
        ToolCall(name="fetch", arguments={"url": "https://example.com"}),
    ]
    r = tool_call_divergence(baseline, candidate)
    assert r.passed
    assert r.score == 1.0
    assert r.metadata["name_jaccard"] == 1.0
    assert r.metadata["arg_match_rate"] == 1.0
    assert r.metadata["cardinality"] == 0


def test_both_empty_scores_one() -> None:
    r = tool_call_divergence([], [])
    assert r.passed
    assert r.score == 1.0
    assert r.metadata["name_jaccard"] == 1.0
    assert r.metadata["arg_match_rate"] is None


def test_one_empty_scores_zero() -> None:
    baseline = [ToolCall(name="search", arguments={})]
    r = tool_call_divergence(baseline, [])
    assert not r.passed
    assert r.score == 0.0
    assert r.metadata["cardinality"] == -1
    r2 = tool_call_divergence([], baseline)
    assert not r2.passed
    assert r2.score == 0.0
    assert r2.metadata["cardinality"] == 1


def test_same_names_different_args_halves_score() -> None:
    baseline = [ToolCall(name="search", arguments={"query": "rust"})]
    candidate = [ToolCall(name="search", arguments={"query": "python"})]
    r = tool_call_divergence(baseline, candidate)
    assert r.metadata["name_jaccard"] == 1.0
    assert r.metadata["arg_match_rate"] == 0.0
    assert r.score == 0.0  # 1.0 * 0.0
    assert not r.passed


def test_partial_name_overlap() -> None:
    baseline = [ToolCall(name="search", arguments={}), ToolCall(name="fetch", arguments={})]
    candidate = [ToolCall(name="search", arguments={}), ToolCall(name="parse", arguments={})]
    r = tool_call_divergence(baseline, candidate)
    # multiset intersection = {search} (1), union = {search, fetch, parse} as multiset (3)
    # so jaccard = 1/3
    assert abs(r.metadata["name_jaccard"] - 1 / 3) < 1e-9
    assert r.metadata["arg_match_rate"] == 1.0  # the search call args match


def test_cardinality_mismatch_penalises_via_jaccard() -> None:
    baseline = [ToolCall(name="search", arguments={})]
    candidate = [
        ToolCall(name="search", arguments={}),
        ToolCall(name="search", arguments={}),
    ]
    r = tool_call_divergence(baseline, candidate)
    # multiset: a=[search], b=[search,search] -> intersection 1, union 2 -> 0.5
    assert r.metadata["name_jaccard"] == 0.5
    assert r.metadata["cardinality"] == 1
    assert r.metadata["arg_match_rate"] == 1.0  # the matched pair has equal args


def test_pairing_aligns_in_order() -> None:
    baseline = [
        ToolCall(name="search", arguments={"q": "a"}),
        ToolCall(name="search", arguments={"q": "b"}),
    ]
    candidate = [
        ToolCall(name="search", arguments={"q": "b"}),  # swapped order
        ToolCall(name="search", arguments={"q": "a"}),
    ]
    r = tool_call_divergence(baseline, candidate)
    # in-order pairing: bi=0 -> ci=0 (q=a vs q=b, no match), bi=1 -> ci=1 (q=b vs q=a, no match)
    assert r.metadata["arg_match_rate"] == 0.0
    assert r.metadata["matched_pairs"] == [(0, 0), (1, 1)]


def test_arg_dict_key_order_does_not_matter() -> None:
    baseline = [ToolCall(name="api", arguments={"x": 1, "y": 2})]
    candidate = [ToolCall(name="api", arguments={"y": 2, "x": 1})]
    r = tool_call_divergence(baseline, candidate)
    assert r.score == 1.0
    assert r.passed


def test_disjoint_names_zero_jaccard() -> None:
    baseline = [ToolCall(name="search", arguments={})]
    candidate = [ToolCall(name="parse", arguments={})]
    r = tool_call_divergence(baseline, candidate)
    assert r.metadata["name_jaccard"] == 0.0
    assert r.metadata["matched_pairs"] == []
    assert r.metadata["arg_match_rate"] is None
    assert r.score == 0.0
