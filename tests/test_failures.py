"""Every failure path, exercised at the boundary where it is handled."""

from __future__ import annotations

import json

import httpx
import pytest

from llm import FatalError, OpenAIChatLLM, TransientError
from tools import InvalidArgument, NotFound, Workspace, build_registry


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (tmp_path / "big.txt").write_text("line\n" * 60_000, encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "sub").mkdir()
    return tmp_path


def registry_for(workspace):
    return build_registry(workspace)


def error_of(raw: str) -> dict:
    payload = json.loads(raw)
    assert "error" in payload, payload
    return payload


def test_missing_required_field(workspace):
    payload = error_of(registry_for(workspace).call("read_file", "{}"))
    assert payload["kind"] == "invalid_arguments"
    assert "missing required field 'path'" in payload["details"]


def test_wrong_type_names_the_field(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": 123}'))
    assert payload["details"] == ["'path' must be string, got int"]


def test_boolean_is_not_accepted_as_integer(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "notes.txt", "start_line": true}'))
    assert "'start_line' must be integer, got boolean" in payload["details"]


def test_unexpected_field_lists_what_is_allowed(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "notes.txt", "nope": 1}'))
    assert "unexpected field 'nope'" in payload["details"][0]
    assert "path" in payload["details"][0]


def test_unknown_tool_lists_the_real_ones(workspace):
    payload = error_of(registry_for(workspace).call("delete_everything", "{}"))
    assert payload["kind"] == "unknown_tool"
    assert "list_dir" in payload["hint"]


def test_malformed_json_arguments(workspace):
    payload = error_of(registry_for(workspace).call("read_file", "{path: notes.txt"))
    assert payload["kind"] == "bad_json"


def test_arguments_must_be_an_object(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '["notes.txt"]'))
    assert payload["kind"] == "bad_json"


def test_missing_file_suggests_the_parent_directory(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "sub/notes.txt"}'))
    assert payload["kind"] == "not_found"
    assert "'sub'" in payload["hint"]


def test_directory_read_as_file(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "sub"}'))
    assert payload["kind"] == "invalid_argument"


def test_binary_file_is_refused(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "blob.bin"}'))
    assert payload["kind"] == "invalid_argument"
    assert "binary" in payload["error"]


def test_large_file_is_truncated_but_reports_its_real_size(workspace):
    out = registry_for(workspace).call("read_file", '{"path": "big.txt"}')
    assert out.startswith("big.txt (lines 1-200 of")
    assert "larger than" in out
    assert "more line(s) not shown" in out


def test_start_line_past_the_end(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "notes.txt", "start_line": 99}'))
    assert payload["kind"] == "invalid_argument"
    assert "3 lines" in payload["error"]


def test_invalid_regex_names_the_problem(workspace):
    payload = error_of(registry_for(workspace).call("grep", '{"pattern": "TODO["}'))
    assert payload["kind"] == "invalid_argument"
    assert "invalid regular expression" in payload["error"]


def test_no_matches_is_a_result_not_an_error(workspace):
    out = registry_for(workspace).call("grep", '{"pattern": "zzz_not_present"}')
    assert out.startswith("no matches for")
    assert "error" not in out


def test_path_cannot_escape_the_workspace(workspace):
    payload = error_of(registry_for(workspace).call("read_file", '{"path": "../../etc/passwd"}'))
    assert payload["kind"] == "invalid_argument"
    assert "outside the workspace" in payload["error"]


def test_workspace_raises_directly_too(workspace):
    with pytest.raises(NotFound):
        Workspace(workspace).read_file("nope.txt")
    with pytest.raises(InvalidArgument):
        Workspace(workspace).read_file("../escape.txt")


def _chat_body(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }


def _response(status: int, body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "http://test/chat/completions")
    if body is None:
        return httpx.Response(status, text="upstream boom", request=request)
    return httpx.Response(status, json=body, request=request)


def make_llm(monkeypatch, sequence, max_attempts=3):
    delays: list[float] = []
    llm = OpenAIChatLLM(
        "test-model",
        api_key="test",
        base_url="http://test",
        max_attempts=max_attempts,
        sleep=lambda d: delays.append(d),
    )
    pending = list(sequence)
    calls: list[dict] = []

    def fake_post(_url, json=None):
        calls.append(json)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm._client, "post", fake_post)
    return llm, calls, delays


def test_retries_a_429_then_succeeds(monkeypatch):
    llm, calls, delays = make_llm(
        monkeypatch, [_response(429), _response(429), _response(200, _chat_body("recovered"))]
    )
    response = llm.complete([{"role": "user", "content": "hi"}], None)

    assert response.content == "recovered"
    assert len(calls) == 3
    assert len(delays) == 2
    assert delays[0] < delays[1], "backoff should grow between attempts"


def test_does_not_retry_a_401(monkeypatch):
    llm, calls, delays = make_llm(monkeypatch, [_response(401)])
    with pytest.raises(FatalError):
        llm.complete([{"role": "user", "content": "hi"}], None)
    assert len(calls) == 1
    assert delays == []


def test_gives_up_after_max_attempts(monkeypatch):
    llm, calls, _ = make_llm(monkeypatch, [_response(503), _response(503), _response(503)])
    with pytest.raises(FatalError, match="gave up after 3 attempts"):
        llm.complete([{"role": "user", "content": "hi"}], None)
    assert len(calls) == 3


def test_network_timeout_is_retried(monkeypatch):
    llm, calls, _ = make_llm(
        monkeypatch,
        [httpx.ConnectTimeout("timed out"), _response(200, _chat_body("second try"))],
    )
    assert llm.complete([{"role": "user", "content": "hi"}], None).content == "second try"
    assert len(calls) == 2


def test_malformed_body_is_treated_as_transient(monkeypatch):
    llm, calls, _ = make_llm(monkeypatch, [_response(200, {"nope": True}), _response(200, _chat_body("fixed"))])
    assert llm.complete([{"role": "user", "content": "hi"}], None).content == "fixed"
    assert len(calls) == 2


def test_transient_error_is_raised_when_attempts_run_out(monkeypatch):
    llm, _, _ = make_llm(monkeypatch, [httpx.ReadTimeout("slow")], max_attempts=1)
    with pytest.raises(FatalError):
        llm.complete([{"role": "user", "content": "hi"}], None)
