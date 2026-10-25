"""End-to-end recovery: the model is wrong, reads the error, and corrects itself.

These are the tests that justify the whole design. If tool errors were raised
instead of returned, none of these runs could finish.
"""

from __future__ import annotations

import json

import pytest

from agent import Agent, Tool, ToolRegistry
from llm import ScriptedLLM, call_tool, say
from tools import build_registry


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    return tmp_path


def test_bad_argument_type_is_corrected_in_one_turn(workspace):
    script = [
        call_tool("read_file", {"path": "notes.txt", "start_line": "one"}, "c1"),
        call_tool("read_file", {"path": "notes.txt"}, "c2"),
        say("notes.txt contains alpha, beta and gamma"),
    ]
    agent = Agent(ScriptedLLM(script), build_registry(workspace))
    answer = agent.run("what is in notes.txt?")

    first_error = json.loads(agent.steps[0].results[0][1])
    assert first_error["kind"] == "invalid_arguments"
    assert "'start_line' must be integer, got str" in first_error["details"]

    assert "alpha" in agent.steps[1].results[0][1]
    assert answer.startswith("notes.txt contains")


def test_wrong_path_recovery_walks_back_to_the_root(workspace):
    script = [
        call_tool("read_file", {"path": "sub/notes.txt"}, "c1"),
        call_tool("list_dir", {"path": "sub"}, "c2"),
        call_tool("list_dir", {"path": "."}, "c3"),
        call_tool("read_file", {"path": "notes.txt"}, "c4"),
        say("it was in the root all along"),
    ]
    agent = Agent(ScriptedLLM(script), build_registry(workspace))
    answer = agent.run("find notes.txt")

    assert json.loads(agent.steps[0].results[0][1])["kind"] == "not_found"
    assert "(empty directory)" in agent.steps[1].results[0][1]
    assert "notes.txt" in agent.steps[2].results[0][1]
    assert answer == "it was in the root all along"


def test_bad_regex_is_retried_with_a_valid_one(workspace):
    script = [
        call_tool("grep", {"pattern": "alpha["}, "c1"),
        call_tool("grep", {"pattern": "alpha"}, "c2"),
        say("alpha appears on line 1"),
    ]
    agent = Agent(ScriptedLLM(script), build_registry(workspace))
    agent.run("grep for alpha")

    assert json.loads(agent.steps[0].results[0][1])["kind"] == "invalid_argument"
    assert "notes.txt:1:" in agent.steps[1].results[0][1]


def test_a_crashing_tool_does_not_kill_the_run(workspace):
    def explode(path: str = ".") -> str:
        raise ValueError("internal invariant broken")

    registry = ToolRegistry().add(
        Tool(
            name="explode",
            description="always fails",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
            fn=explode,
        )
    )
    script = [
        call_tool("explode", {}, "c1"),
        say("that tool is broken, I will work around it"),
    ]
    agent = Agent(ScriptedLLM(script), registry)
    answer = agent.run("try the broken tool")

    payload = json.loads(agent.steps[0].results[0][1])
    assert payload["kind"] == "tool_crashed"
    assert "will not help" in payload["hint"]
    assert answer.startswith("that tool is broken")


def test_unknown_tool_returns_a_usable_hint(workspace):
    script = [
        call_tool("search_files", {"query": "alpha"}, "c1"),
        call_tool("grep", {"pattern": "alpha"}, "c2"),
        say("used grep instead"),
    ]
    agent = Agent(ScriptedLLM(script), build_registry(workspace))
    agent.run("search for alpha")

    payload = json.loads(agent.steps[0].results[0][1])
    assert payload["kind"] == "unknown_tool"
    assert "grep" in payload["hint"]
    assert "alpha" in agent.steps[1].results[0][1]
