"""Loop-level behaviour: termination, budgets, and history shape."""

from __future__ import annotations

import pytest

from agent import Agent
from llm import Response, ScriptedLLM, Usage, call_tool, say
from tools import build_registry


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    return tmp_path


def make_agent(script, workspace, **kwargs):
    llm = ScriptedLLM(script)
    return Agent(llm, build_registry(workspace), **kwargs), llm


def test_plain_answer_terminates_immediately(workspace):
    agent, llm = make_agent([say("nothing to do")], workspace)
    assert agent.run("hi") == "nothing to do"
    assert len(llm.calls) == 1
    assert len(agent.steps) == 1


def test_one_tool_call_then_answer(workspace):
    script = [
        call_tool("list_dir", {"path": "."}, "c1"),
        say("the workspace has notes.txt and sub/"),
    ]
    agent, llm = make_agent(script, workspace)
    answer = agent.run("what is here?")

    assert answer.startswith("the workspace has")
    assert len(agent.steps) == 2
    assert "notes.txt" in agent.steps[0].results[0][1]


def test_history_pairs_every_tool_call_with_a_result(workspace):
    script = [
        call_tool("read_file", {"path": "notes.txt"}, "c1"),
        say("done"),
    ]
    agent, llm = make_agent(script, workspace)
    agent.run("read the notes")

    messages = llm.calls[1]["messages"]
    assistant = messages[-2]
    tool = messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "c1"


def test_step_limit_forces_a_final_answer(workspace):
    script = [
        call_tool("list_dir", {"path": "."}, "c1"),
        call_tool("list_dir", {"path": "sub"}, "c2"),
        say("ran out of steps, but here is what I saw"),
    ]
    agent, llm = make_agent(script, workspace, max_steps=2)
    answer = agent.run("keep looking")

    assert answer == "ran out of steps, but here is what I saw"
    assert len(llm.calls) == 3
    assert "step limit of 2" in llm.calls[2]["messages"][-1]["content"]


def test_repeated_identical_call_is_cut_off(workspace):
    same = {"path": "."}
    script = [
        call_tool("list_dir", same, "c1"),
        call_tool("list_dir", same, "c2"),
        call_tool("list_dir", same, "c3"),
        say("I am stuck on list_dir"),
    ]
    agent, llm = make_agent(script, workspace, max_identical_calls=2)
    answer = agent.run("loop forever")

    assert answer == "I am stuck on list_dir"
    assert "identical arguments 3 times" in llm.calls[-1]["messages"][-1]["content"]


def test_different_arguments_are_not_treated_as_repeats(workspace):
    script = [
        call_tool("list_dir", {"path": "."}, "c1"),
        call_tool("list_dir", {"path": "sub"}, "c2"),
        call_tool("list_dir", {"path": "."}, "c3"),
        say("done"),
    ]
    agent, _ = make_agent(script, workspace, max_identical_calls=2)
    assert agent.run("look twice") == "done"


def test_token_budget_stops_the_run(workspace):
    expensive = Response(
        tool_calls=[{"id": "c1", "name": "list_dir", "arguments": '{"path": "."}'}],
        usage=Usage(prompt_tokens=5_000, completion_tokens=5_000),
    )
    script = [expensive, say("budget gone")]
    agent, llm = make_agent(script, workspace, token_budget=1_000)
    answer = agent.run("spend everything")

    assert answer == "budget gone"
    assert "token budget of 1000" in llm.calls[-1]["messages"][-1]["content"]


def test_finalize_withholds_tools_entirely(workspace):
    script = [
        call_tool("list_dir", {"path": "."}, "c1"),
        say("ok"),
    ]
    agent, llm = make_agent(script, workspace, max_steps=1)
    agent.run("one step only")

    assert llm.calls[0]["tools"] is not None
    assert llm.calls[-1]["tools"] is None


def test_empty_model_output_gets_a_placeholder(workspace):
    agent, _ = make_agent([Response(content=None)], workspace)
    assert "no text" in agent.run("say nothing")
