"""A tool-calling loop small enough to read in one sitting.

The point of this file is the failure handling, so the governing rule is worth
stating up front:

    A tool failure is data, not an exception.

The model can fix a bad path, a bad regex, a bad argument type. It cannot fix
a 429 or an expired key. So tool errors are serialised and handed back as the
tool's result, while transport errors are retried or raised in llm.py. This
loop only guards the failures nobody can fix from inside it: running forever,
spending forever, and repeating itself.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from llm import Response

SYSTEM_PROMPT = """You are a file-inspection agent working inside one workspace.

- Inspect with the tools before answering. Never guess file contents.
- If you do not know the layout, call list_dir first.
- Every tool returns text, or a JSON object with an "error" key. Read "hint":
  it usually names the next call that would work.
- If the same call fails twice for the same reason, stop and say what you
  could not determine.
- When you have enough information, answer in prose and stop calling tools."""


class ToolError(Exception):
    """A failure the model may be able to fix. Becomes a tool result."""

    kind = "tool_error"

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": self.message, "kind": self.kind}
        if self.hint:
            out["hint"] = self.hint
        return out


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """A deliberately small subset of JSON Schema: required keys and primitive types.

    This is not a general validator. It covers the two mistakes models actually
    make -- omitting a field and passing the wrong type -- and it names the
    offending field so the model can correct it in one turn. A dependency would
    buy more coverage than this loop needs.
    """
    problems: list[str] = []
    props = schema.get("properties") or {}

    for key in schema.get("required") or []:
        if key not in args:
            problems.append(f"missing required field {key!r}")

    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            allowed = ", ".join(sorted(props)) or "none"
            problems.append(f"unexpected field {key!r}; allowed fields: {allowed}")
            continue
        want = spec.get("type")
        expected = _JSON_TYPES.get(want or "")
        if expected is None:
            continue
        if want in ("integer", "number") and isinstance(value, bool):
            problems.append(f"{key!r} must be {want}, got boolean")
        elif not isinstance(value, expected):
            problems.append(f"{key!r} must be {want}, got {type(value).__name__}")

    return problems


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


class ToolRegistry:
    """Owns the boundary where a tool call becomes either text or an error payload."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return self

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, raw_arguments: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return _encode({
                "error": f"unknown tool {name!r}",
                "kind": "unknown_tool",
                "hint": f"available tools: {', '.join(sorted(self._tools))}",
            })

        try:
            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
        except (ValueError, TypeError) as exc:
            return _encode({
                "error": f"tool arguments were not valid JSON: {exc}",
                "kind": "bad_json",
                "hint": "emit one JSON object, with no surrounding prose and no code fence",
            })
        if not isinstance(args, dict):
            return _encode({
                "error": f"tool arguments must be an object, got {type(args).__name__}",
                "kind": "bad_json",
            })

        problems = validate_arguments(tool.parameters, args)
        if problems:
            return _encode({
                "error": "invalid arguments",
                "kind": "invalid_arguments",
                "details": problems,
            })

        try:
            return str(tool.fn(**args))
        except ToolError as exc:
            return _encode(exc.payload())
        except Exception as exc:  # a bug in the tool, not in the model's arguments
            return _encode({
                "error": f"{type(exc).__name__}: {exc}",
                "kind": "tool_crashed",
                "hint": "the tool itself failed; repeating the same call will not help",
            })


@dataclass
class Step:
    index: int
    response: Response
    results: list[tuple[str, str]] = field(default_factory=list)


def _assistant_message(response: Response) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in response.tool_calls
        ]
    return message


def _canonical(raw_arguments: Any) -> str:
    try:
        return json.dumps(json.loads(raw_arguments), sort_keys=True)
    except (ValueError, TypeError):
        return str(raw_arguments)


class Agent:
    def __init__(
        self,
        llm,
        registry: ToolRegistry,
        max_steps: int = 12,
        token_budget: int = 60_000,
        max_identical_calls: int = 2,
        log: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.max_identical_calls = max_identical_calls
        self._log = log
        self.steps: list[Step] = []

    def run(self, task: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        repeats: Counter[tuple[str, str]] = Counter()
        spent = 0

        for index in range(1, self.max_steps + 1):
            response = self.llm.complete(messages, self.registry.schemas())
            spent += response.usage.total
            step = Step(index, response)
            self.steps.append(step)
            self._log(f"step {index}: {len(response.tool_calls)} tool call(s), {spent} tokens")

            if not response.tool_calls:
                return response.content or "(the model returned no text)"

            if spent > self.token_budget:
                return self._finalize(messages, f"the token budget of {self.token_budget} ran out at step {index}")

            # Checked before the assistant turn is committed, so the history
            # never ends with tool_calls that have no matching tool messages --
            # several providers reject that shape outright.
            for call in response.tool_calls:
                key = (call["name"], _canonical(call["arguments"]))
                repeats[key] += 1
                if repeats[key] > self.max_identical_calls:
                    return self._finalize(
                        messages,
                        f"{call['name']} was called with identical arguments "
                        f"{repeats[key]} times, so the model is stuck",
                    )

            messages.append(_assistant_message(response))
            for call in response.tool_calls:
                result = self.registry.call(call["name"], call["arguments"])
                step.results.append((call["name"], result))
                self._log(f"  <- {call['name']}: {result.splitlines()[0][:90] if result else '(empty)'}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        return self._finalize(messages, f"the step limit of {self.max_steps} was reached")

    def _finalize(self, messages: list[dict[str, Any]], reason: str) -> str:
        """Close out with whatever the model already knows.

        Tools are withheld rather than merely discouraged: passing None makes a
        tool call impossible, so the model has to produce an answer.
        """
        self._log(f"stopping early: {reason}")
        messages.append({
            "role": "user",
            "content": f"{reason}. Stop calling tools and answer using what you already have.",
        })
        response = self.llm.complete(messages, None)
        self.steps.append(Step(len(self.steps) + 1, response))
        return response.content or f"(stopped early: {reason})"
