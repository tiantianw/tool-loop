"""Offline demo: a scripted run that fails, recovers, and finishes.

No API key, no network. The failure paths are the interesting part, so they are
reproducible rather than described: anyone reviewing this repo can watch the
loop recover instead of trusting a screenshot.

    python demo.py
    python demo.py --transcript examples/transcript.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from llm import Response, ScriptedLLM, Usage, call_tool, say
from tools import build_registry

TASK = "Summarise what this repository does, and tell me how long agent.py is."


def _turn(response: Response, prompt: int, completion: int) -> Response:
    """Give scripted turns plausible token counts so the trace reads like a real run."""
    response.usage = Usage(prompt_tokens=prompt, completion_tokens=completion)
    return response


def build_demo_agent(root: str | Path = ".") -> tuple[Agent, str]:
    root_path = Path(root)
    total_lines = len((root_path / "agent.py").read_text(encoding="utf-8").splitlines())

    final = (
        f"This repository is a minimal tool-calling agent loop. agent.py is {total_lines} lines "
        "and holds three things: a registry of read-only file tools, a loop that turns tool "
        "errors into tool results so the model can correct itself, and guards against running "
        "forever or repeating the same call. ToolError is declared exactly once, in agent.py; "
        "tools.py subclasses it for the three file tools, and llm.py never sees it at all "
        "because transport failures are handled before the loop runs."
    )

    # Built by concatenation so this file does not match the pattern it searches
    # for -- otherwise the demo finds its own source and reports a wrong count.
    needle = "class " + "ToolError"

    script = [
        # The model guesses a src/ layout that does not exist. The hint sends it
        # one level down, where it fails again, and only then to the root.
        _turn(call_tool("read_file", {"path": "src/agent.py"}, "call_1"), 1180, 42),
        _turn(call_tool("list_dir", {"path": "src"}, "call_2"), 1240, 38),
        _turn(call_tool("list_dir", {"path": "."}, "call_3"), 1300, 44),
        # An unclosed character class. The error names the pattern problem.
        _turn(call_tool("grep", {"pattern": needle + "[", "path": ".", "glob": "*.py"}, "call_4"), 1520, 41),
        # Retried with a valid pattern. One declaration, which is the answer.
        _turn(call_tool("grep", {"pattern": needle, "path": ".", "glob": "*.py"}, "call_5"), 1580, 39),
        # Default max_lines truncates, and the header still reports the true size.
        _turn(call_tool("read_file", {"path": "agent.py"}, "call_6"), 1650, 46),
        _turn(say(final), 3100, 118),
    ]
    return Agent(ScriptedLLM(script), build_registry(root_path), log=print), TASK


def render_transcript(agent: Agent, task: str) -> str:
    """Rebuild a readable trace from what the agent already recorded."""
    out = [f"**Task** — {task}", ""]
    for step in agent.steps:
        out.append(f"#### Step {step.index}")
        out.append("")
        if step.response.content:
            out.append(f"> {step.response.content}")
            out.append("")
        for call in step.response.tool_calls:
            out.append(f"`{call['name']}({call['arguments']})`")
            out.append("")
        for _name, result in step.results:
            body = result if len(result) < 900 else result[:900] + "\n... (cut for readability)"
            out.append("```")
            out.append(body)
            out.append("```")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline scripted demo.")
    parser.add_argument("--root", default=".", help="workspace the tools may read")
    parser.add_argument("--transcript", help="also write a markdown trace to this path")
    args = parser.parse_args()

    agent, task = build_demo_agent(args.root)
    answer = agent.run(task)
    print()
    print("FINAL ANSWER")
    print(answer)

    if args.transcript:
        path = Path(args.transcript)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Transcript\n\n"
            "Produced by `python demo.py --transcript examples/transcript.md` against the\n"
            "repository as committed. The model is a scripted stand-in, so this output is\n"
            "deterministic: every failure below is reproducible, not illustrative.\n\n"
        )
        path.write_text(header + render_transcript(agent, task), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
