"""CLI entry point.

    python main.py --demo
    python main.py "find every TODO in this repo" --root . --model gpt-4o-mini

Without --demo this needs an OpenAI-compatible endpoint: OPENAI_API_KEY, and
optionally OPENAI_BASE_URL and OPENAI_MODEL.
"""

from __future__ import annotations

import argparse
import os
import sys

from agent import Agent
from llm import FatalError, OpenAIChatLLM
from tools import build_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A minimal tool-calling agent.")
    parser.add_argument("task", nargs="?", help="what the agent should do")
    parser.add_argument("--demo", action="store_true", help="run the offline scripted demo; no API key needed")
    parser.add_argument("--root", default=".", help="workspace the tools may read (default: .)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max-steps", type=int, default=12, help="hard cap on loop iterations")
    args = parser.parse_args(argv)

    if args.demo:
        from demo import build_demo_agent

        agent, task = build_demo_agent(args.root)
    else:
        if not args.task:
            parser.error("provide a task, or pass --demo to run offline")
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("set OPENAI_API_KEY, or pass --demo to run offline")
        llm = OpenAIChatLLM(args.model, log=lambda msg: print(f"[llm] {msg}", file=sys.stderr))
        agent = Agent(llm, build_registry(args.root), max_steps=args.max_steps, log=print)
        task = args.task

    try:
        print(agent.run(task))
    except FatalError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
